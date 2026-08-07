"""共享探针集（probe set）与评估用 DataLoader 工厂。

# 为什么需要共享探针集

实验 A 要回答的是"同一个 δ 在**不同客户端的模型**上自然程度是否不同"。
如果每个客户端在自己的本地测试集上测量，则

    Var_k(Overlap_k) = 模型差异（我们要的） + 评估数据差异（纯噪声）

而第二项在高异构（小 alpha）下会**压倒**第一项 —— 于是即使 δ 完全跨环境不变，
也会得到一条漂亮的"CV 随 alpha 增长"曲线。那条曲线什么都不能证明。

正确做法是**固定数据、只变模型**。本模块从全局测试集抽取一份固定的探针集，
在所有 alpha / seed / mode / 客户端之间**完全共用**。

# 四个子集及其不相交要求

    ref     真实目标类图片        -> kNN 的参照分布
    other   每个非目标类若干张    -> kNN 的对照分布
    query   非目标类图片          -> 施加 delta / 随机噪声的载体
    target  目标类图片            -> 实验 B（Table 24 复现）

``query`` 必须与 ``ref ∪ other`` 不相交，否则 kNN 会匹配到底图完全相同的样本，
overlap 被虚高。同理 ``target`` 必须与 ``ref`` 不相交，否则实验 A 的"正对照"
组等于在自己身上做最近邻，overlap 恒为 1。

# 唯一的例外

``metrics.observable_score`` 使用的类原型**不用**探针集，而用客户端的本地训练
数据 —— 因为它模拟的是防御方在真实部署中能拿到的东西。详见 metrics.py 的说明。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

__all__ = ["make_eval_loader", "get_targets", "ProbeSet", "build_probe_set",
           "assert_disjoint"]


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------
def make_eval_loader(dataset: Dataset, batch_size: int = 256,
                     num_workers: int = 0) -> DataLoader:
    """构造**评估专用** DataLoader。

    两个关键设置，都不是可选项：

    - ``drop_last=False``：原仓库在**测试集**上也用了 ``drop_last=True``
      （main.py:83），这是训练时的约定泄漏到了评估路径。后果是每个客户端静默
      丢掉最多 ``batch_size - 1`` 个样本，且丢弃比例随客户端样本数变化 ——
      在高异构下会给指标注入系统性偏差。
    - ``shuffle=False``：保证确定性。kNN 的邻居集合必须可复现。
    """
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                      drop_last=False, num_workers=int(num_workers))


# ---------------------------------------------------------------------------
# 标签提取
# ---------------------------------------------------------------------------
def get_targets(dataset: Dataset) -> np.ndarray:
    """取出数据集的整型标签数组，尽量避免遍历。

    依次尝试 ``.targets`` / ``.labels``；``Subset`` 会递归并按索引取子集；
    都没有时退化为逐样本遍历（慢，但保证可用）。
    """
    if isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return np.asarray(parent)[np.asarray(dataset.indices, dtype=np.int64)]

    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy().astype(np.int64)
            return np.asarray(value, dtype=np.int64)

    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))],
                      dtype=np.int64)


# ---------------------------------------------------------------------------
# 探针集
# ---------------------------------------------------------------------------
@dataclass
class ProbeSet:
    """四个互不相交的探针子集。

    ``indices`` 中保存的是相对于**原始数据集**的绝对索引，写进
    ``results/probe_indices.json`` 即可完全复现。
    """

    ref: Subset
    other: Subset
    query: Subset
    target: Subset
    target_class: int
    indices: Dict[str, List[int]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def loaders(self, batch_size: int = 256) -> Dict[str, DataLoader]:
        """返回 ``{'ref','other','query','target'}`` 四个评估 loader。"""
        return {
            name: make_eval_loader(getattr(self, name), batch_size=batch_size)
            for name in ("ref", "other", "query", "target")
        }

    def save_indices(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"target_class": self.target_class,
                   "meta": self.meta,
                   "indices": {k: list(map(int, v))
                               for k, v in self.indices.items()}}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def assert_disjoint(*subsets: Subset) -> None:
    """断言若干 ``Subset`` 的索引两两不相交，否则抛 ``ValueError``。

    这个检查进冒烟测试。索引重叠会让 kNN overlap 虚高，是**静默**的错误 ——
    结果看起来完全合理，只是全错。
    """
    seen: Dict[int, int] = {}
    for position, subset in enumerate(subsets):
        indices = getattr(subset, "indices", subset)
        for index in np.asarray(indices, dtype=np.int64).tolist():
            if index in seen:
                raise ValueError(
                    f"探针子集索引重叠：索引 {index} 同时出现在第 {seen[index]} "
                    f"和第 {position} 个子集中")
            seen[index] = position


def build_probe_set(test_dataset: Dataset, target_class: int,
                    cfg: Any, rng_seed: Optional[int] = None) -> ProbeSet:
    """构造共享探针集。

    Parameters
    ----------
    test_dataset:
        全局测试集（``cfg.probe.source`` 目前只支持 ``"test"``）。
    target_class:
        目标类标签。
    cfg:
        完整配置对象；本函数读取 ``cfg.probe`` 下的字段。
    rng_seed:
        覆盖 ``cfg.probe.seed``。**正常情况下不要传** —— 探针集必须与实验 seed
        解耦，否则不同 seed 的 run 之间不再是 apples-to-apples 的比较。
        仅供单元测试使用。

    Notes
    -----
    抽样使用独立的 ``np.random.RandomState``，不触碰全局 numpy RNG，
    因此调用本函数**不会**扰动数据划分的随机流。
    """
    probe_cfg = cfg["probe"] if isinstance(cfg, dict) and "probe" in cfg else cfg
    seed = int(rng_seed if rng_seed is not None else probe_cfg["seed"])
    n_ref = int(probe_cfg["n_ref"])
    n_other_per_class = int(probe_cfg["n_other_per_class"])
    n_query = int(probe_cfg["n_query"])
    n_target = int(probe_cfg["n_target"])

    targets = get_targets(test_dataset)
    classes = np.unique(targets)
    if target_class not in classes:
        raise ValueError(
            f"目标类 {target_class} 不存在于数据集中；实际类别: {classes.tolist()}")

    rng = np.random.RandomState(seed)

    # --- 目标类：先切 ref，再切 target，保证不相交 ---
    target_pool = np.where(targets == target_class)[0]
    target_pool = rng.permutation(target_pool)
    need_target = n_ref + n_target
    if len(target_pool) < need_target:
        raise ValueError(
            f"目标类样本不足：需要 n_ref({n_ref}) + n_target({n_target}) = "
            f"{need_target}，实际只有 {len(target_pool)}")
    ref_idx = target_pool[:n_ref]
    target_idx = target_pool[n_ref:n_ref + n_target]

    # --- 非目标类：先按类切 other，剩下的池子里切 query ---
    other_idx_parts: List[np.ndarray] = []
    leftover_parts: List[np.ndarray] = []
    for cls in classes:
        if int(cls) == int(target_class):
            continue
        pool = rng.permutation(np.where(targets == cls)[0])
        if len(pool) < n_other_per_class:
            raise ValueError(
                f"类别 {int(cls)} 样本不足：需要 n_other_per_class="
                f"{n_other_per_class}，实际只有 {len(pool)}")
        other_idx_parts.append(pool[:n_other_per_class])
        leftover_parts.append(pool[n_other_per_class:])

    other_idx = np.concatenate(other_idx_parts) if other_idx_parts else np.array([], dtype=np.int64)
    leftover = np.concatenate(leftover_parts) if leftover_parts else np.array([], dtype=np.int64)
    leftover = rng.permutation(leftover)
    if len(leftover) < n_query:
        raise ValueError(
            f"非目标类剩余样本不足以构造 query：需要 {n_query}，"
            f"扣除 other 后只剩 {len(leftover)}")
    query_idx = leftover[:n_query]

    probe = ProbeSet(
        ref=Subset(test_dataset, ref_idx.tolist()),
        other=Subset(test_dataset, other_idx.tolist()),
        query=Subset(test_dataset, query_idx.tolist()),
        target=Subset(test_dataset, target_idx.tolist()),
        target_class=int(target_class),
        indices={"ref": ref_idx.tolist(), "other": other_idx.tolist(),
                 "query": query_idx.tolist(), "target": target_idx.tolist()},
        meta={"probe_seed": seed, "n_ref": n_ref,
              "n_other_per_class": n_other_per_class, "n_query": n_query,
              "n_target": n_target, "num_classes": int(len(classes))},
    )

    # 不相交是硬要求，构造完立刻自检，不给"静默虚高"留缝隙。
    assert_disjoint(probe.ref, probe.other, probe.query, probe.target)
    return probe
