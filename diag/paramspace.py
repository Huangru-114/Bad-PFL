"""参数空间的逐坐标度量 —— **纯 numpy**，不 import torch。

# 为什么单独成模块、为什么不 import torch

`PLAN_T0T4.md §7` 的 Stage 0 全部在 CPU 上做，而本机没有 torch/GPU。
把"算什么"与"从哪读"分开之后：

- 本模块只接受 ``Dict[str, np.ndarray]``，可以用手工构造的、**精确可手算**的
  小字典做单元测试（`diag/tests/test_paramspace.py`），不需要任何 checkpoint；
- 读 ``.pt`` 的那一层（`diag/exp_t0.py` 的 ``load_state``）才 import torch，
  且是**延迟导入**，因此在没有 torch 的机器上依然能跑本模块的全部测试。

# 张量分类（用结构判定，不用名字猜）

BN 的仿射参数 ``bn.weight`` / ``bn.bias`` 与卷积的 ``conv.weight`` 在键名上
只差父模块名，靠 ``"bn" in key`` 这类子串匹配去分是不可靠的（``layer1.0.bn1``
可以，但用户自定义命名就不行）。这里改用**结构判据**：

    key 的父模块下存在 ``running_mean``  ->  该 key 是 BN 仿射参数

``running_mean`` / ``running_var`` / ``num_batches_tracked`` 本身归为 BN buffer。

这条判据对本仓库的 ResNet 精确成立（`resnet.py` 的每个 ``nn.BatchNorm2d``
都会带 ``running_mean``），且不依赖任何命名约定。

# 为什么必须把 BN 单独拎出来（`PLAN_T0T4.md` 坑 6）

**FedBN 一直开着，BN 从不聚合**（见 `CLAUDE.md` / `README §4b`）。全局模型的
BN buffer 停在初始化值，把它算进"参数位移"里会得到一个与训练无关的常数偏置。
所以：**主分析用可训练参数**（weight / bias / bn_affine），BN buffer 的统计量
单独报，不混在一起。若 BN 真承载了大量后门能量，那本身就是一个结果。

# 手写 Pearson / Spearman 的理由

`analysis_exposure` 用了 ``scipy.stats.spearmanr``，但 scipy 不是本项目的
硬依赖（很多分析模块只用 numpy/pandas）。Stage 0 要能在最小环境里跑，
所以这里自带秩相关，并采用**平均秩**处理并列（与 scipy 的默认一致）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "KIND_WEIGHT", "KIND_BIAS", "KIND_BN_AFFINE", "KIND_BN_BUFFER",
    "TRAINABLE_KINDS",
    "parameter_kind", "layer_of", "block_of",
    "ParamIndex", "build_index", "flatten", "displacement",
    "l2", "cosine", "sign_agreement", "relative_displacement",
    "group_energy", "topk_energy", "average_rank", "rank_quantile",
    "pearson", "spearman", "binned_curve", "layer_table",
]

KIND_WEIGHT = "weight"
KIND_BIAS = "bias"
KIND_BN_AFFINE = "bn_affine"
KIND_BN_BUFFER = "bn_buffer"

#: 主分析纳入的种类。BN buffer 被**刻意排除**（见模块 docstring）。
TRAINABLE_KINDS: Tuple[str, ...] = (KIND_WEIGHT, KIND_BIAS, KIND_BN_AFFINE)

_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


# ---------------------------------------------------------------------------
# 键的分类与分组
# ---------------------------------------------------------------------------
def layer_of(key: str) -> str:
    """键所属的模块路径 = 去掉最后一段。

    ``layer1.0.bn1.weight -> layer1.0.bn1``；顶层的 ``linear.weight -> linear``；
    没有点号的键返回 ``""``（整个 state_dict 只有一个张量的退化情形）。
    """
    return key.rsplit(".", 1)[0] if "." in key else ""


def block_of(key: str) -> str:
    """粗粒度分组 = 键的第一段。

    ResNet 下得到 ``conv1 / bn1 / layer1 / layer2 / layer3 / layer4 / linear``。
    逐层表太长时用它画图；**逐层表本身不做聚合**，两者都出。
    """
    return key.split(".", 1)[0]


def parameter_kind(key: str, all_keys: Iterable[str]) -> str:
    """张量种类：``weight`` / ``bias`` / ``bn_affine`` / ``bn_buffer``。

    ``all_keys`` 是整份 state_dict 的键集合 —— BN 仿射参数靠"父模块下有
    ``running_mean``"这个**结构判据**识别（见模块 docstring），而不是名字里
    有没有 "bn"。
    """
    leaf = key.rsplit(".", 1)[-1]
    if leaf in _BN_BUFFER_SUFFIXES:
        return KIND_BN_BUFFER
    parent = layer_of(key)
    sibling = f"{parent}.running_mean" if parent else "running_mean"
    if sibling in set(all_keys):
        return KIND_BN_AFFINE
    if leaf == "bias":
        return KIND_BIAS
    return KIND_WEIGHT


class ParamIndex:
    """展平向量与原 state_dict 之间的双向映射。

    只收录 **浮点** 张量：``num_batches_tracked`` 是 int64 计数器，把它塞进
    L2 位移里没有任何含义。被跳过的键留在 ``excluded`` 里如实可查，**不静默丢弃**。
    """

    def __init__(self, keys: Sequence[str], sizes: Sequence[int],
                 kinds: Sequence[str], shapes: Sequence[Tuple[int, ...]],
                 excluded: Sequence[str]):
        self.keys: List[str] = list(keys)
        self.sizes: List[int] = [int(s) for s in sizes]
        self.kinds: List[str] = list(kinds)
        self.shapes: List[Tuple[int, ...]] = [tuple(s) for s in shapes]
        self.excluded: List[str] = list(excluded)
        self.offsets: List[int] = np.concatenate(
            [[0], np.cumsum(self.sizes)]).astype(int).tolist()
        self.layers: List[str] = [layer_of(k) for k in self.keys]
        self.blocks: List[str] = [block_of(k) for k in self.keys]

    # -- 基本属性 ---------------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.offsets[-1]) if self.offsets else 0

    def __len__(self) -> int:
        return len(self.keys)

    def slice_of(self, key: str) -> slice:
        i = self.keys.index(key)
        return slice(self.offsets[i], self.offsets[i + 1])

    # -- 掩码 -------------------------------------------------------------
    def _mask_from(self, labels: Sequence[str], wanted: Iterable[str]
                   ) -> np.ndarray:
        wanted = set(wanted)
        mask = np.zeros(self.n_params, dtype=bool)
        for i, label in enumerate(labels):
            if label in wanted:
                mask[self.offsets[i]:self.offsets[i + 1]] = True
        return mask

    def kind_mask(self, kinds: Iterable[str] = TRAINABLE_KINDS) -> np.ndarray:
        """逐坐标的布尔掩码。默认是"可训练参数"，即**排除 BN buffer**。"""
        return self._mask_from(self.kinds, kinds)

    def layer_mask(self, layers: Iterable[str]) -> np.ndarray:
        return self._mask_from(self.layers, layers)

    def group_labels(self, by: str = "layer") -> List[str]:
        """逐坐标的分组标签，长度 = ``n_params``。"""
        if by == "layer":
            source = self.layers
        elif by == "block":
            source = self.blocks
        elif by == "kind":
            source = self.kinds
        elif by == "key":
            source = self.keys
        else:
            raise ValueError(f"未知分组 '{by}'；可选 layer / block / kind / key")
        out: List[str] = []
        for i, label in enumerate(source):
            out.extend([label] * self.sizes[i])
        return out


def build_index(state: Dict[str, np.ndarray]) -> ParamIndex:
    """按**键名排序**建立索引 —— 排序保证跨 run / 跨机器的展平顺序一致。"""
    all_keys = list(state.keys())
    keys, sizes, kinds, shapes, excluded = [], [], [], [], []
    for key in sorted(all_keys):
        value = np.asarray(state[key])
        if not np.issubdtype(value.dtype, np.floating):
            excluded.append(key)
            continue
        keys.append(key)
        sizes.append(int(value.size))
        kinds.append(parameter_kind(key, all_keys))
        shapes.append(tuple(value.shape))
    return ParamIndex(keys, sizes, kinds, shapes, excluded)


def flatten(state: Dict[str, np.ndarray],
            index: Optional[ParamIndex] = None) -> np.ndarray:
    """按 ``index`` 的顺序展平成 float64 一维向量。

    float64 是刻意的：checkpoint 存的是 float32，而 T0 要在千万量级的坐标上
    求平方和，float32 累加的相对误差可以到 1e-3 量级，足以改变"位移小"的判断。
    """
    index = build_index(state) if index is None else index
    parts = []
    for key, shape in zip(index.keys, index.shapes):
        value = np.asarray(state[key], dtype=np.float64)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"键 '{key}' 的形状 {tuple(value.shape)} 与索引里的 {shape} "
                f"不一致 —— 两个 state_dict 不是同一个模型结构")
        parts.append(value.reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def displacement(state_from: Dict[str, np.ndarray],
                 state_to: Dict[str, np.ndarray],
                 index: Optional[ParamIndex] = None
                 ) -> Tuple[np.ndarray, np.ndarray, ParamIndex]:
    """返回 ``(theta_from, delta = theta_to - theta_from, index)``。

    索引以 ``state_from`` 为准；两边的键集合不一致时直接报错，**不取交集** ——
    静默取交集会把"两个 run 用了不同模型"这种事故变成一张看起来正常的表。
    """
    index = build_index(state_from) if index is None else index
    missing = sorted(set(index.keys) - set(state_to.keys()))
    extra = sorted(set(k for k in state_to
                       if np.issubdtype(np.asarray(state_to[k]).dtype,
                                        np.floating)) - set(index.keys))
    if missing or extra:
        raise ValueError(
            f"两个 state_dict 的浮点键集合不一致：\n"
            f"  只在 from 里：{missing[:5]}{' …' if len(missing) > 5 else ''}\n"
            f"  只在 to   里：{extra[:5]}{' …' if len(extra) > 5 else ''}")
    theta_from = flatten(state_from, index)
    theta_to = flatten(state_to, index)
    return theta_from, theta_to - theta_from, index


# ---------------------------------------------------------------------------
# 向量层面的度量
# ---------------------------------------------------------------------------
def l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    """余弦相似度。任一侧范数为 0 时返回 **nan**，不返回 0。

    0 会被下游当成"正交"读，而"零向量与谁都不成角度"是**未定义**，不是正交
    （`CLAUDE.md` 铁律 5：无定义的指标留空）。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} vs {y.shape}")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= 0.0 or ny <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def sign_agreement(x: np.ndarray, y: np.ndarray,
                   eps: float = 0.0) -> Dict[str, float]:
    """逐坐标符号一致率，**连同分母一起返回**。

    只在两侧都 ``|.| > eps`` 的坐标上比较。零坐标既不算一致也不算冲突 ——
    把它们记作"一致"会让稀疏向量的一致率虚高到 1。

    返回 ``n_compared`` / ``n_skipped`` 是必需的：``rate`` 单独一个数没法判断
    它是在 1e7 个坐标上算的还是在 3 个坐标上算的。有效坐标为 0 时 rate 为 nan。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} vs {y.shape}")
    valid = (np.abs(x) > eps) & (np.abs(y) > eps)
    n_compared = int(valid.sum())
    if n_compared == 0:
        return {"rate": float("nan"), "conflict_rate": float("nan"),
                "n_compared": 0, "n_skipped": int(x.size)}
    agree = int((np.sign(x[valid]) == np.sign(y[valid])).sum())
    rate = agree / n_compared
    return {"rate": float(rate), "conflict_rate": float(1.0 - rate),
            "n_compared": n_compared, "n_skipped": int(x.size - n_compared)}


def relative_displacement(delta: np.ndarray, base: np.ndarray) -> float:
    """‖Δθ‖ / ‖θ‖ —— 位移的无量纲刻度。``‖θ‖ = 0`` 时返回 nan。"""
    denominator = l2(base)
    if denominator <= 0.0:
        return float("nan")
    return l2(delta) / denominator


def group_energy(delta: np.ndarray, index: ParamIndex, by: str = "layer",
                 base: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """逐组的位移能量（平方和）与占比。

    ``energy_share`` 的分母是**传入的这条 delta 的总能量**。若调用方只传了
    可训练参数的子向量，占比就是"在可训练参数里的占比" —— 分母是什么由调用方
    决定，本函数不替它选。
    """
    delta = np.asarray(delta, dtype=np.float64)
    labels = np.asarray(index.group_labels(by))
    if labels.shape[0] != delta.shape[0]:
        raise ValueError(
            f"分组标签长度 {labels.shape[0]} 与向量长度 {delta.shape[0]} 不一致"
            f" —— 索引与向量不是同一个掩码下的产物")
    total = float(np.dot(delta, delta))
    rows: List[Dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        block = delta[mask]
        energy = float(np.dot(block, block))
        row: Dict[str, Any] = {
            "group": label, "n_params": int(mask.sum()),
            "l2_delta": float(np.sqrt(energy)),
            "energy": energy,
            "energy_share": (energy / total if total > 0 else float("nan")),
            "mean_abs_delta": float(np.abs(block).mean()) if block.size else
                              float("nan"),
        }
        if base is not None:
            base_block = np.asarray(base, dtype=np.float64)[mask]
            row["l2_base"] = l2(base_block)
            row["relative_displacement"] = relative_displacement(block,
                                                                 base_block)
        rows.append(row)
    return rows


def topk_energy(delta: np.ndarray,
                fractions: Sequence[float] = (0.0001, 0.001, 0.01, 0.1)
                ) -> List[Dict[str, float]]:
    """按 |Δ| 排序后，前 k 比例的坐标占了多少位移能量。

    这是"位移是不是集中在少数坐标上"的直接读数：均匀分布时
    ``energy_share ≈ k``，高度集中时 ``≫ k``。**所有 k 都要报**
    （`PLAN_T0T4.md` 坑 8：不挑一个好看的 k）。
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = int(delta.size)
    total = float(np.dot(delta, delta))
    order = np.argsort(-np.abs(delta), kind="stable")
    squared_sorted = delta[order] ** 2
    cumulative = np.cumsum(squared_sorted)
    rows: List[Dict[str, float]] = []
    for fraction in fractions:
        count = int(np.ceil(float(fraction) * n))
        count = max(0, min(n, count))
        captured = float(cumulative[count - 1]) if count > 0 else 0.0
        rows.append({
            "fraction": float(fraction), "n_top": count,
            "energy_share": (captured / total if total > 0 else float("nan")),
            "concentration": ((captured / total) / float(fraction)
                              if total > 0 and fraction > 0 else float("nan")),
        })
    return rows


def average_rank(x: np.ndarray) -> np.ndarray:
    """1..n 的平均秩（并列取平均），与 scipy 的默认一致。"""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_x = x[order]
    # 把每一段并列的秩换成该段的平均值
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return ranks


def rank_quantile(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """``mask`` 选中的坐标，在全体 ``values`` 里落在什么分位。

    T1 的核心读数（"P 落在 g_ben 的低分位还是高分位"）就是这个；T0 用它做
    "大位移坐标是不是集中在某一层"的稳健性检查。
    分位定义为 ``(rank - 0.5) / n``，因此均匀随机的子集期望是 0.5。
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"形状不一致：{values.shape} vs {mask.shape}")
    n = int(values.size)
    if n == 0 or not mask.any():
        return {"mean_quantile": float("nan"), "median_quantile": float("nan"),
                "n_selected": int(mask.sum()), "n_total": n}
    quantiles = (average_rank(values) - 0.5) / n
    picked = quantiles[mask]
    return {"mean_quantile": float(picked.mean()),
            "median_quantile": float(np.median(picked)),
            "n_selected": int(mask.sum()), "n_total": n}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关。任一侧方差为 0 时返回 nan（相关系数未定义）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} vs {y.shape}")
    if x.size < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(xc, yc) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关 = 平均秩上的 Pearson。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} vs {y.shape}")
    return pearson(average_rank(x), average_rank(y))


def binned_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 10,
                 mode: str = "quantile") -> List[Dict[str, float]]:
    """把 ``y`` 按 ``x`` 分箱 —— "大 θ 的坐标是不是位移也大"这类问题的读数。

    ``mode='quantile'`` 按 x 的分位切（每箱样本数相近，重尾数据下更可读），
    ``mode='uniform'`` 按 x 的值域等距切。空箱**如实保留**（count=0，统计量为
    nan），不合并 —— 合并会让 x 轴不再等距。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"形状不一致：{x.shape} vs {y.shape}")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins 至少为 1")
    if x.size == 0:
        return []

    if mode == "quantile":
        edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    elif mode == "uniform":
        edges = np.linspace(float(x.min()), float(x.max()), n_bins + 1)
    else:
        raise ValueError(f"未知 mode '{mode}'；可选 quantile / uniform")
    edges = np.asarray(edges, dtype=np.float64)

    rows: List[Dict[str, float]] = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (x >= low) & (x <= high)      # 最后一箱闭区间，收住最大值
        else:
            mask = (x >= low) & (x < high)
        count = int(mask.sum())
        rows.append({
            "bin": i, "x_low": float(low), "x_high": float(high),
            "count": count,
            "x_mean": float(x[mask].mean()) if count else float("nan"),
            "y_mean": float(y[mask].mean()) if count else float("nan"),
            "y_median": float(np.median(y[mask])) if count else float("nan"),
            "y_std": (float(y[mask].std(ddof=1)) if count >= 2
                      else float("nan")),
        })
    return rows


def layer_table(theta: np.ndarray, delta: np.ndarray, index: ParamIndex,
                by: str = "layer") -> List[Dict[str, Any]]:
    """逐层（或逐 block / 逐 kind）的位移剖面表。

    每行给出：坐标数、‖θ‖、‖Δθ‖、相对位移、能量占比、平均 |Δ|、
    以及 ``cos(θ, Δθ)`` —— 最后一个能区分"整体缩放"（cos ≈ ±1）与
    "方向改写"（cos ≈ 0）。
    """
    theta = np.asarray(theta, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if theta.shape != delta.shape:
        raise ValueError(f"形状不一致：{theta.shape} vs {delta.shape}")
    labels = np.asarray(index.group_labels(by))
    if labels.shape[0] != delta.shape[0]:
        raise ValueError(
            f"分组标签长度 {labels.shape[0]} 与向量长度 {delta.shape[0]} 不一致")

    rows = group_energy(delta, index, by=by, base=theta)
    for row in rows:
        mask = labels == row["group"]
        row["cos_theta_delta"] = cosine(theta[mask], delta[mask])
        row["kind"] = _dominant_kind(index, by, row["group"])
    return rows


def _dominant_kind(index: ParamIndex, by: str, group: str) -> str:
    """该组里出现的种类；混合时用 ``+`` 连起来，不挑一个代表。"""
    if by == "kind":
        return str(group)
    labels = index.layers if by == "layer" else (
        index.blocks if by == "block" else index.keys)
    kinds = sorted({kind for kind, label in zip(index.kinds, labels)
                    if label == group})
    return "+".join(kinds)
