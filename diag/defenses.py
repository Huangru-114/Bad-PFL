"""五种服务器端聚合规则的统一接口（实验 I / J）。

======================  ==========================  =========================
防御                     是否做客户端级二元决策        I^(k)
======================  ==========================  =========================
``fedavg``              否                          恒为 1
``median``              否                          取值落在"中间两位"的坐标比例
``invariant``           否                          trim 存活率 × 掩码保留率
``multi_krum``          **是**                      1 if 被选中 else 0
``flame``               **是**                      1 if 在良性簇内 else 0
======================  ==========================  =========================

``median`` 与 ``invariant`` **不判断"这个客户端是恶意的"**，因此
``DefenseOutcome.selected`` 为 ``None``，下游一律不得为它们计算 TPR/FPR
（实验 J 陷阱 1 / 实验 I 陷阱 9）。

# 三条共同的实现约束

1. **只作用于共享参数**：key 过滤走 ``diag.fedbn``，与 ``pfl.py:8/17`` 同一份
   逻辑。FedBN 私有的 key 与非浮点 key 一律走简单平均，并保留在返回的字典里，
   好让 ``fedbn_update`` 照常 pop —— FedBN 的行为一位不变。
2. **不写坏输入**：``agg_avg`` 的别名 bug（``server.py:5``）意味着任何原地写
   都会污染第一个客户端上传的 dict。本模块的产物一律是新张量。
3. **不展平**：``[N, D]`` 在 ResNet-10 上约 200MB。所有需要两两关系的量
   （Krum 距离、FLAME 余弦）都从**流式累加的 Gram 矩阵**导出::

       G[i,j] = <g_i, g_j>
       ||g_i − g_j||² = G[i,i] + G[j,j] − 2 G[i,j]
       cos(i,j)       = G[i,j] / sqrt(G[i,i] G[j,j])

   Gram 矩阵是 N×N，按 key 累加即可。

# 伪梯度方向

全部沿用 ``g_i = w_prev − w_i``（旧减新），与 ``diag.invariant_agg`` 一致；
聚合结果 ``ḡ`` 之后 ``w_t = w_prev − ḡ``。无防御时精确退化为 FedAvg，
有测试固定。
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .fedbn import split_keys
from .instrumentation import (compose_influence, median_influence,
                              rank_window_fraction, trim_survival)
# Gram 原语只在 instrumentation 里定义一份 —— 两处各写一份必然漂移，
# 而漂移是静默的（FLAME 的簇和 Multi-Krum 的分数会基于不同的距离）。
# 这里用别名保留旧的私有名，本文件其余部分不改。
from .instrumentation import gram_matrix
from .instrumentation import pairwise_cosine as _pairwise_cosine
from .instrumentation import pairwise_distance as _pairwise_distance
from .instrumentation import pseudo_grad_stack as _pseudo_grad_stack
from .invariant_agg import (and_mask, sign_consistency, trim_count,
                            trimmed_mean)

__all__ = ["DEFENSES", "ABLATION_VARIANTS", "DefenseOutcome", "Defense",
           "build_defense", "use_defense", "gram_matrix"]

DEFENSES: Tuple[str, ...] = ("fedavg", "median", "invariant", "multi_krum",
                             "flame", "oracle_exclude")

# 实验 I §3.1 的消融：三个变体都是**同一个聚合器的参数设置**，不是三份实现。
# 这样"单组件"与"组合"之间不可能出现实现差异带来的伪差别。
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    # 组合：掩码 + trimmed mean
    "invariant": {},
    # 仅 AND-mask：trim_alpha=0 -> 不裁剪，退化为掩码 ⊙ 简单平均
    "invariant_mask_only": {"trim_alpha": 0.0},
    # 仅 trimmed-mean：tau=-1 -> 一致性恒 > τ，掩码全放行
    "invariant_trim_only": {"tau": -1.0},
}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
@dataclass
class DefenseOutcome:
    """一轮聚合的埋点产物。字段名与 ``diag.instrumentation.RoundRecord`` 对齐。"""

    influence: np.ndarray
    selected: Optional[np.ndarray] = None      # None = 该防御不做客户端级决策
    trim_survival_rate: Optional[np.ndarray] = None
    mask_keep_ratio: float = float("nan")
    zero_sign_ratio: float = float("nan")
    effective_trim_n: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def makes_client_decision(self) -> bool:
        return self.selected is not None


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
def _plain_mean(state_dicts: Sequence[Dict[str, torch.Tensor]], key: str
                ) -> torch.Tensor:
    reference = state_dicts[0][key]
    if not torch.is_tensor(reference):
        return copy.deepcopy(reference)
    # 用 mean(dtype=float64) 而不是先把整个 stack 转成 float64：
    # ResNet-10 的最大一层上后者是约 190MB 的临时分配。
    stacked = torch.stack([state[key].detach() for state in state_dicts], dim=0)
    return stacked.mean(dim=0, dtype=torch.float64).to(reference.dtype)


def _apply(w_prev: Dict[str, torch.Tensor], key: str,
           aggregated: torch.Tensor) -> torch.Tensor:
    """``w_t = w_prev − ḡ``，保留原 dtype。"""
    reference = w_prev[key]
    return (reference.detach().to(torch.float32) - aggregated).to(reference.dtype)


# ---------------------------------------------------------------------------
# 防御
# ---------------------------------------------------------------------------
class Defense:
    """所有聚合规则的基类。子类只需实现 ``_aggregate_shared``。"""

    name = "base"

    def __call__(self, w_prev: Dict[str, torch.Tensor],
                 state_dicts: Sequence[Dict[str, torch.Tensor]]
                 ) -> Tuple[Dict[str, torch.Tensor], DefenseOutcome]:
        if not state_dicts:
            raise ValueError("没有收到任何客户端更新")
        shared_float, shared_nonfloat, private = split_keys(state_dicts[0])

        update: Dict[str, torch.Tensor] = {}
        # FedBN 私有 + 非浮点：简单平均，保留在字典里让 fedbn_update 照常 pop
        for key in shared_nonfloat + private:
            update[key] = _plain_mean(state_dicts, key)

        shared, outcome = self._aggregate_shared(w_prev, state_dicts,
                                                 shared_float)
        update.update(shared)
        outcome.extra.setdefault("defense", self.name)
        outcome.extra.setdefault("n_shared_float_keys", len(shared_float))
        outcome.extra.setdefault("nonfloat_fallback_keys",
                                 ";".join(shared_nonfloat))
        return update, outcome

    def _aggregate_shared(self, w_prev, state_dicts, keys
                          ) -> Tuple[Dict[str, torch.Tensor], DefenseOutcome]:
        raise NotImplementedError


class FedAvg(Defense):
    """无防御。与 ``server.agg_avg`` 数值等价，但不写坏输入。"""

    name = "fedavg"

    def _aggregate_shared(self, w_prev, state_dicts, keys):
        n_clients = len(state_dicts)
        update = {key: _plain_mean(state_dicts, key) for key in keys}
        return update, DefenseOutcome(
            influence=compose_influence("fedavg", n_clients=n_clients))


class Median(Defense):
    """逐坐标中位数。**不做客户端级决策** -> 没有 TPR/FPR。"""

    name = "median"

    def _aggregate_shared(self, w_prev, state_dicts, keys):
        n_clients = len(state_dicts)
        update: Dict[str, torch.Tensor] = {}
        hits = torch.zeros(n_clients, dtype=torch.float64)   # 累加器在 CPU
        total = 0
        for key in keys:
            stacked = _pseudo_grad_stack(w_prev, state_dicts, key)
            update[key] = _apply(w_prev, key, stacked.median(dim=0).values)
            numel = stacked[0].numel()
            hits += median_influence(stacked).cpu() * numel
            total += numel
        influence = (hits / max(total, 1)).numpy()
        return update, DefenseOutcome(
            influence=compose_influence("median", n_clients=n_clients,
                                        median_influence_values=influence))


class InvariantAggregator(Defense):
    """符号一致性 AND-mask ⊙ trimmed mean（Wang et al., AISTATS 2024）。

    实现细节与偏离记录见 ``diag/invariant_agg.py`` 的模块 docstring。
    """

    name = "invariant"

    def __init__(self, tau: float = 0.2, trim_alpha: float = 0.25, *,
                 mask_mode: str = "binary", sign_zero_policy: str = "count",
                 variant: str = "invariant"):
        self.tau = float(tau)
        self.trim_alpha = float(trim_alpha)
        self.mask_mode = mask_mode
        self.sign_zero_policy = sign_zero_policy
        self.name = variant

    def _aggregate_shared(self, w_prev, state_dicts, keys):
        n_clients = len(state_dicts)
        k = trim_count(n_clients, self.trim_alpha)
        update: Dict[str, torch.Tensor] = {}
        kept, total, zeros_total, consistency_sum = 0, 0, 0, 0.0
        survival = torch.zeros(n_clients, dtype=torch.float64)   # 累加器在 CPU
        effective = n_clients

        for key in keys:
            stacked = _pseudo_grad_stack(w_prev, state_dicts, key)
            consistency, zeros = sign_consistency(stacked, self.sign_zero_policy)
            mask = and_mask(consistency, self.tau, mode=self.mask_mode,
                            n_clients=n_clients)
            centre, effective = trimmed_mean(stacked, self.trim_alpha)
            update[key] = _apply(w_prev, key, mask * centre)

            numel = consistency.numel()
            total += numel
            kept += int((mask > 0).sum())
            zeros_total += int(zeros.sum())
            consistency_sum += float(consistency.sum())
            survival += trim_survival(stacked, k).cpu() * numel

        mask_keep_ratio = kept / total if total else float("nan")
        survival_rate = (survival / max(total, 1)).numpy()
        return update, DefenseOutcome(
            influence=compose_influence(
                "invariant", n_clients=n_clients,
                trim_survival_rate=survival_rate,
                mask_keep_ratio=mask_keep_ratio),
            trim_survival_rate=survival_rate,
            mask_keep_ratio=mask_keep_ratio,
            zero_sign_ratio=(zeros_total / (total * n_clients) if total
                             else float("nan")),
            effective_trim_n=effective,
            extra={"tau": self.tau, "trim_alpha": self.trim_alpha,
                   "trim_k": k, "mask_mode": self.mask_mode,
                   "sign_zero_policy": self.sign_zero_policy,
                   "mean_sign_consistency": (consistency_sum / total if total
                                             else float("nan"))})


class MultiKrum(Defense):
    """Multi-Krum：按 Krum 分数选 ``c`` 个更新求平均。**做客户端级决策。**

    Krum 分数 = 到最近的 ``n − f − 2`` 个其他更新的平方距离之和，越小越"合群"。
    ``f`` 是假定的拜占庭数；N=10、f=2 时取最近的 6 个。
    """

    name = "multi_krum"

    def __init__(self, n_byzantine: int = 2, n_selected: Optional[int] = None):
        self.n_byzantine = int(n_byzantine)
        self.n_selected = n_selected

    def _aggregate_shared(self, w_prev, state_dicts, keys):
        n_clients = len(state_dicts)
        f = min(self.n_byzantine, max(0, (n_clients - 3) // 2))
        neighbours = max(1, n_clients - f - 2)
        c = int(self.n_selected) if self.n_selected else max(1, n_clients - f)
        c = min(c, n_clients)

        gram = gram_matrix(w_prev, state_dicts, keys)
        squared = _pairwise_distance(gram) ** 2
        squared.fill_diagonal_(float("inf"))
        scores = torch.sort(squared, dim=1).values[:, :neighbours].sum(dim=1)
        chosen = torch.argsort(scores)[:c].tolist()

        selected = np.zeros(n_clients, dtype=bool)
        selected[chosen] = True
        picked = [state_dicts[i] for i in chosen]
        update = {key: _apply(w_prev, key,
                              _pseudo_grad_stack(w_prev, picked, key).mean(dim=0))
                  for key in keys}
        return update, DefenseOutcome(
            influence=compose_influence("multi_krum", n_clients=n_clients,
                                        selected=selected),
            selected=selected,
            extra={"krum_scores": scores.tolist(), "n_byzantine": f,
                   "n_neighbours": neighbours, "n_selected": c})


class Flame(Defense):
    """FLAME：中位距离裁剪 -> HDBSCAN 余弦聚类 -> 聚合 + 加噪。

    # N=10 时 HDBSCAN 会退化（实验 J §4.1）

    FLAME 原设定 ``min_cluster_size ≈ N/2 + 1``；N=10 时是 6。10 个点上
    HDBSCAN 很可能把**全部**点判为噪声（label = −1）。此时回退为"全部接受"，
    并把 ``fallback_triggered`` 记进埋点 —— 回退触发的轮次比例必须报告，
    否则会把"FLAME 没拦住"与"FLAME 根本没生效"混为一谈。

    三个环节分别埋点（实验 J §4.2）：裁剪（``clipped`` / 裁剪前后范数）、
    聚类（``selected`` / 簇数 / 噪声点数 / 最大簇大小）、加噪（``noise_sigma``）。
    """

    name = "flame"

    def __init__(self, noise_lambda: float = 0.001,
                 min_cluster_size: Optional[int] = None,
                 seed: Optional[int] = 0):
        self.noise_lambda = float(noise_lambda)
        self.min_cluster_size = min_cluster_size
        # 噪声**始终**走独立的 Generator，绝不碰全局 torch RNG。
        # 否则每轮加噪都会推进全局随机流，改变后续所有的客户端采样、
        # PGD 随机起点与生成器初始化 —— 而这种污染没有任何症状，
        # 只会让 FLAME 组与其他组不再同源。seed=None 时用一个固定基准，
        # 配合轮次计数器让噪声逐轮变化。
        self.seed = 0 if seed is None else int(seed)
        self._round = 0

    def _cluster(self, cosine: np.ndarray, n_clients: int
                 ) -> Tuple[np.ndarray, Dict[str, Any]]:
        from sklearn.cluster import HDBSCAN

        min_size = int(self.min_cluster_size or (n_clients // 2 + 1))
        min_size = max(2, min(min_size, n_clients))
        distance = np.clip(1.0 - cosine, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        labels = HDBSCAN(min_cluster_size=min_size, min_samples=1,
                         metric="precomputed",
                         allow_single_cluster=True).fit_predict(distance)

        valid = labels[labels >= 0]
        info = {
            "flame_n_clusters": int(len(np.unique(valid))) if valid.size else 0,
            "flame_n_noise": int((labels < 0).sum()),
            "flame_min_cluster_size": min_size,
        }
        if valid.size == 0:
            # 全部被判为噪声 -> 回退为全部接受，并如实记录
            info.update({"flame_max_cluster_size": 0,
                         "flame_fallback_triggered": True})
            return np.ones(n_clients, dtype=bool), info

        counts = np.bincount(valid)
        largest = int(np.argmax(counts))
        info.update({"flame_max_cluster_size": int(counts[largest]),
                     "flame_fallback_triggered": False})
        return labels == largest, info

    def _aggregate_shared(self, w_prev, state_dicts, keys):
        n_clients = len(state_dicts)
        gram = gram_matrix(w_prev, state_dicts, keys)
        norms = gram.diagonal().clamp(min=0.0).sqrt()
        cosine = _pairwise_cosine(gram).numpy()

        # --- 环节 1：按更新范数的中位数裁剪 ---
        clip_bound = float(norms.median())
        scale = torch.clamp(clip_bound / norms.clamp(min=1e-12), max=1.0)
        clipped = (scale < 1.0).numpy()

        # --- 环节 2：HDBSCAN 聚类 ---
        selected, cluster_info = self._cluster(cosine, n_clients)

        # --- 环节 3：聚合被选中者的裁剪后更新，再加噪 ---
        chosen = np.flatnonzero(selected).tolist()
        weights = scale[chosen]
        sigma = self.noise_lambda * clip_bound
        generator = torch.Generator().manual_seed(self.seed + self._round * 7919)
        self._round += 1

        update: Dict[str, torch.Tensor] = {}
        for key in keys:
            stacked = _pseudo_grad_stack(w_prev, state_dicts, key)[chosen]
            # weights 来自 CPU 上的 gram，stacked 可能在 GPU 上
            local_weights = weights.to(stacked.device, stacked.dtype)
            weighted = stacked * local_weights.view(
                -1, *([1] * (stacked.dim() - 1)))
            aggregated = weighted.mean(dim=0)
            if sigma > 0:
                # 噪声在 CPU 上按独立 Generator 生成后再搬过去：
                # CPU 的 Generator 不能直接用于 CUDA 张量的 normal_()。
                noise = torch.empty(aggregated.shape, dtype=torch.float32).normal_(
                    mean=0.0, std=sigma, generator=generator)
                aggregated = aggregated + noise.to(aggregated.device,
                                                   aggregated.dtype)
            update[key] = _apply(w_prev, key, aggregated)

        extra = {"clip_bound": clip_bound, "noise_sigma": sigma,
                 "clipped": clipped.tolist(),
                 "norm_before_clip": norms.tolist(),
                 "norm_after_clip": (norms * scale).tolist(),
                 "cos_to_others_mean": (
                     (cosine.sum(axis=1) - 1.0) / max(n_clients - 1, 1)).tolist()}
        extra.update(cluster_info)
        return update, DefenseOutcome(
            influence=compose_influence("flame", n_clients=n_clients,
                                        selected=selected),
            selected=selected, extra=extra)


class OracleExclude(Defense):
    """"静默"恶意客户端：按**真实标签**完美剔除恶意更新。

    恶意客户端一切照常 —— 本地在投毒数据上训练、每次被选中都对着刚下发的
    全局模型重训生成器 30 步 —— **只是它的更新永远不进入聚合**。

    # 这一格测什么

    它同时是三件东西：

    1. **任何排除型防御的上界。** TPR=1、FPR=0，没有防御能做得更好。
       若后门在这个设定下仍然出现在**良性**客户端的个性化模型里，那就说明
       "识别并排除恶意客户端"这条路原则上走不通。
    2. **整条流水线的负对照。** 恶意 -> 良性的唯一通道就是聚合。把它掐断后
       良性 ASR 若仍高于基线，说明有**意料之外的泄漏**，是 bug 而不是发现。
    3. **实验 E3 的连续对齐版本，也是最强的白盒 δ。** E3 是对着**最终**的
       干净模型训 30 步（ASR 0.1462）；这里生成器在约 50 次参与 × 30 步
       ≈ 1500 步里**持续追踪正在训练的那个干净模型**。若 δ 仍然无效，
       "δ 是目标类自然特征"在其最强形式下也被否掉。

    位置上它介于 ``bad=1``（模型被投毒 47 轮）与 E2/E3（模型从未被投毒，
    但生成器只对齐了别的轨迹或只训了 30 步）之间。

    # 正对照不可省

    恶意客户端**自己**的个性化模型是被投毒的，其 ASR 应当很高。
    没有这个正对照，"良性 ASR 在基线"既可能是"排除成功"也可能是
    "投毒根本没生效"，两者读不出来 —— 见 ``--eval-include-malicious``。
    """

    name = "oracle_exclude"

    def __init__(self, inner: Optional[Defense] = None,
                 malicious_getter: Optional[Callable[[], np.ndarray]] = None):
        # 内层默认 FedAvg：既然攻击者已被完美剔除，剩下的就该是无防御的平均，
        # 否则把"完美排除"与"某个聚合规则"两件事混在一起了。
        self.inner = inner or FedAvg()
        self.malicious_getter = malicious_getter

    def __call__(self, w_prev, state_dicts):
        if self.malicious_getter is None:
            raise ValueError(
                "OracleExclude 需要 malicious_getter —— 它要按真实标签剔除，"
                "所以必须能拿到当轮选中客户端的恶意掩码")
        malicious = np.asarray(self.malicious_getter(), dtype=bool)
        if malicious.shape[0] != len(state_dicts):
            raise ValueError(
                f"恶意掩码长度 {malicious.shape[0]} 与本轮上传数 "
                f"{len(state_dicts)} 不一致 —— 顺序或时机对不上，必须中止")
        keep = np.flatnonzero(~malicious).tolist()
        if not keep:
            # 全部选中的客户端都是恶意的：没有合法的聚合结果。
            # 静默地把攻击者聚合进去是最坏的结局，所以直接中止。
            raise RuntimeError(
                "本轮选中的客户端全是恶意的，完美排除后无更新可聚合。"
                "请降低恶意比例或增大每轮选中数。")

        update, outcome = self.inner(w_prev, [state_dicts[i] for i in keep])

        # 影响力与决策按**全体**选中客户端回填：被剔除的记 0
        influence = np.zeros(len(state_dicts), dtype=float)
        influence[keep] = np.asarray(outcome.influence, dtype=float)
        selected = ~malicious

        survival = None
        if outcome.trim_survival_rate is not None:
            survival = np.zeros(len(state_dicts), dtype=float)
            survival[keep] = np.asarray(outcome.trim_survival_rate, dtype=float)

        extra = dict(outcome.extra)
        extra.update({"defense": self.name, "inner_defense": self.inner.name,
                      "n_excluded_oracle": int(malicious.sum())})
        return update, DefenseOutcome(
            influence=influence, selected=selected,
            trim_survival_rate=survival,
            mask_keep_ratio=outcome.mask_keep_ratio,
            zero_sign_ratio=outcome.zero_sign_ratio,
            effective_trim_n=outcome.effective_trim_n, extra=extra)


# ---------------------------------------------------------------------------
# 构造与接入
# ---------------------------------------------------------------------------
def build_defense(name: str, **kwargs: Any) -> Defense:
    """按名字构造防御。消融变体（``invariant_mask_only`` 等）在这里展开成参数。"""
    name = str(name).lower()
    if name in ABLATION_VARIANTS:
        merged = dict(kwargs)
        merged.update(ABLATION_VARIANTS[name])
        return InvariantAggregator(variant=name,
                                   **{k: v for k, v in merged.items()
                                      if k in ("tau", "trim_alpha", "mask_mode",
                                               "sign_zero_policy")})
    if name in ("fedavg", "avg", "none"):
        return FedAvg()
    if name == "median":
        return Median()
    if name == "multi_krum":
        return MultiKrum(**{k: v for k, v in kwargs.items()
                            if k in ("n_byzantine", "n_selected")})
    if name == "flame":
        return Flame(**{k: v for k, v in kwargs.items()
                        if k in ("noise_lambda", "min_cluster_size", "seed")})
    if name == "oracle_exclude":
        inner_name = str(kwargs.get("inner", "fedavg"))
        if inner_name == "oracle_exclude":
            raise ValueError("oracle_exclude 不能嵌套自身")
        inner = build_defense(inner_name, **{k: v for k, v in kwargs.items()
                                            if k != "inner"})
        return OracleExclude(inner=inner,
                             malicious_getter=kwargs.get("malicious_getter"))
    raise ValueError(
        f"未知的防御 '{name}'；可选 {list(DEFENSES) + list(ABLATION_VARIANTS)}")


def use_defense(server: Any, defense: Defense,
                on_round: Optional[Callable[[DefenseOutcome, List[Dict[str, torch.Tensor]],
                                             Dict[str, torch.Tensor]], None]] = None
                ) -> Callable[[], None]:
    """把 ``server.agg_and_update`` 换成给定的防御，返回还原函数。

    ``w_prev`` 直接读 ``server.global_model``：``fl_process.py:36`` 调用
    ``agg_and_update`` 时全局模型尚未被本轮更新（``load_state_dict`` 在该函数
    最后一行）。绑定的是**实例方法**，``BasicServer`` 类不受影响，
    原仓库文件仍为零 diff。

    ``on_round(outcome, state_dicts, w_prev)`` 在 ``load_state_dict`` **之前**
    调用，便于埋点用同一份 ``w_prev`` 计算逐轮指标。
    """
    original = server.agg_and_update

    @torch.no_grad()
    def agg_and_update(self, state_dicts) -> None:
        w_prev = {key: value.detach().clone()
                  for key, value in self.global_model.state_dict().items()}
        self.update, outcome = defense(w_prev, state_dicts)
        if on_round is not None:
            on_round(outcome, list(state_dicts), w_prev)
        # 与 server.py:33-34 一致：先跑钩子（FedBN 在这里 pop 私有 key）再 load
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

    server.agg_and_update = types.MethodType(agg_and_update, server)

    def restore() -> None:
        server.agg_and_update = original

    return restore
