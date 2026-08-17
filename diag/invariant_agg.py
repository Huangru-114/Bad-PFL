"""Invariant Aggregator（Wang et al., AISTATS 2024）在本仓库上的重新实现。

论文无官方代码，本模块按 Algorithm 1 重写。

# 算法

    1. m  = 1[ |Σ_i sign(g_i)| / N > τ ]      # 逐维度的二值 AND-mask
    2. g̃ = TrMean({g_1..g_N}; α)              # 逐维度，两端各去 ⌈αN⌉ 个后求均值
    3. ḡ = m ⊙ g̃                              # 掩掉"只让少数客户端受益"的方向
    然后 w_t = w_{t-1} − ḡ

其中伪梯度 **g_i = w_{t-1} − w_i**（旧模型减新模型，陷阱 1：方向搞反会让防御
变成加速攻击）。

# 与本仓库对接时的五件事

1. **仓库聚合的是参数不是伪梯度。** ``server.agg_avg``（server.py:4-10）直接对
   ``state_dict`` 求平均。本模块先把参数转成伪梯度，聚合后再转回参数。
2. **``agg_avg`` 的别名问题。** 首行 ``average_dict = state_dicts[0]`` 是引用。
   本模块**从不写入传入的 state_dict**，产物一律是新张量（有回归测试）。
3. **只作用于共享参数。** key 过滤走 ``diag.fedbn.is_fedbn_private_key``，
   与 ``pfl.py:8/17`` 是同一份逻辑（陷阱 3）。FedBN 私有的 key 仍按简单平均
   放进 ``server.update`` 并保留在字典里，好让注册在 ``before_update_global``
   上的 ``fedbn_update`` 像原来一样把它们 pop 掉 —— **FedBN 的行为一位不变**。
4. **非浮点张量退回简单平均**（陷阱 4），并把 key 列进统计里。
5. **强制 fp32。** 仓库训练用 ``autocast``，符号在 fp16 下不稳定。

# 相对论文的偏离（必须记录，§9）

- **掩码实现为二值 {0,1}**。论文 Definition 7 与 Algorithm 1 第 1 行的掩码带有
  ``1/N`` 因子；照字面实现会把聚合结果整体缩放 ``1/N``，等价于把学习率除以 N。
  判为笔误。``mask_mode="one_over_n"`` 保留了字面版本，供后续需要时对照，
  **不是默认值**。
- **``sign(0) = 0``**：PyTorch 中零梯度的符号是 0，会拉低一致性。默认按论文
  字面把 0 计入分母（``sign_zero_policy="count"``）；``"abstain"`` 变体把分母
  改为非零元素数。两种都会报告 ``zero_sign_ratio``，由数据决定用哪个。
- **非浮点 key 的 dtype**：``agg_avg`` 对整数 buffer 做 ``/ len`` 会得到浮点
  张量；本模块保留原 dtype。在 FedBN 下这些 key 一律被 pop，差异不可观测。

# 有效样本数

每轮只选 10 个客户端。α=0.25 → 每端去 ⌈2.5⌉=3 个，**只剩 4 个**参与平均；
α=0.1 → 每端去 1 个，剩 8 个。``effective_trim_n`` 逐轮记录（陷阱 7），
剩余为 0 时直接报错而不是静默退化。
"""

from __future__ import annotations

import copy
import math
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .fedbn import is_fedbn_private_key, split_keys

__all__ = ["MASK_MODES", "SIGN_ZERO_POLICIES", "RoundStats", "pseudo_gradients",
           "sign_consistency", "and_mask", "trimmed_mean", "trim_count",
           "invariant_aggregate", "use_invariant_aggregator",
           "InvariantAggregatorHandle"]

MASK_MODES: Tuple[str, ...] = ("binary", "one_over_n")
SIGN_ZERO_POLICIES: Tuple[str, ...] = ("count", "abstain")


# ---------------------------------------------------------------------------
# 逐轮统计
# ---------------------------------------------------------------------------
@dataclass
class RoundStats:
    """一轮聚合的可复核统计量。全部会落进 CSV。"""

    round_index: int = -1
    n_clients: int = 0
    n_shared_float_keys: int = 0
    n_shared_float_dims: int = 0
    n_nonfloat_fallback_keys: int = 0
    nonfloat_fallback_keys: List[str] = field(default_factory=list)
    n_fedbn_private_keys: int = 0
    tau: float = 0.0
    trim_alpha: float = 0.0
    trim_k: int = 0
    effective_trim_n: int = 0
    mask_keep_ratio: float = float("nan")
    zero_sign_ratio: float = float("nan")
    mean_sign_consistency: float = float("nan")
    mask_mode: str = "binary"
    sign_zero_policy: str = "count"

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["nonfloat_fallback_keys"] = ";".join(self.nonfloat_fallback_keys)
        return row


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------
def pseudo_gradients(w_prev: Dict[str, torch.Tensor],
                     state_dicts: Sequence[Dict[str, torch.Tensor]],
                     keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """``g_i = w_prev − w_i``，按 key 堆叠成 ``[N, *shape]``。

    方向是 **旧减新**（陷阱 1）。配合 ``w_t = w_prev − ḡ``，在无掩码、无裁剪时
    精确退化为 FedAvg —— ``test_reduces_to_fedavg_without_mask_or_trim``
    就是用这一点同时钉死伪梯度方向与更新符号的。

    全程 fp32：仓库训练用 ``autocast``，fp16 下符号不稳定。
    """
    stacked: Dict[str, torch.Tensor] = {}
    for key in keys:
        reference = w_prev[key].detach().to(torch.float32)
        rows = []
        for state in state_dicts:
            if key not in state:
                raise KeyError(
                    f"客户端上传的 state_dict 缺少 key '{key}'，无法构造伪梯度。")
            rows.append(reference - state[key].detach().to(
                reference.device, torch.float32))
        stacked[key] = torch.stack(rows, dim=0)
    return stacked


def sign_consistency(stacked: torch.Tensor, policy: str = "count"
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐维度的符号一致性 ``|Σ_i sign(g_i)| / 分母``。

    Returns
    -------
    ``(consistency, zero_mask_count)``：一致性张量（形状 = 单个客户端的参数形状），
    以及"符号恰为 0 的条目数"张量（同形状），供统计 ``zero_sign_ratio``。

    ``policy``：

    - ``"count"``（默认，论文字面）：分母 = N，符号为 0 的客户端拉低一致性。
    - ``"abstain"``：分母 = 非零符号的个数，把 0 视为弃权。全零维度返回 0。
    """
    if policy not in SIGN_ZERO_POLICIES:
        raise ValueError(f"未知的 sign_zero_policy '{policy}'；"
                         f"可选 {list(SIGN_ZERO_POLICIES)}")
    signs = torch.sign(stacked)
    n_clients = signs.shape[0]
    zeros = (signs == 0).sum(dim=0)
    total = signs.sum(dim=0).abs()

    if policy == "count":
        denominator = torch.full_like(total, float(n_clients))
    else:
        denominator = (n_clients - zeros).to(total.dtype)
    # 分母为 0（该维度所有客户端的梯度都恰好是 0）时一致性定义为 0：
    # "没有任何客户端表达意见"不该被当成"完全一致"。
    consistency = torch.where(denominator > 0, total / denominator.clamp(min=1),
                              torch.zeros_like(total))
    return consistency, zeros


def chance_mask_keep_ratio(n_clients: int, tau: float) -> float:
    """符号全为**独立公平硬币**时，AND-mask 的期望保留率。

    这是解读 ``mask_keep_ratio`` 必须有的参照。掩码的前提是"符号一致性指示
    该方向是否对多数客户端有益"；若实测保留率贴着这个基线，说明客户端之间
    **根本没有可利用的符号一致性**，掩码退化成一个随机的稀疏化 ——
    那是一种与"后门维度混过了掩码"完全不同的失效方式。

    另一个必须知道的后果：``Σ_i sign(g_i)`` 与 N 同奇偶，所以阈值只有
    ``⌊N/2⌋+1`` 个**互不相同**的取值。N=10 时 τ=0.2 与 τ=0.3 给出的是
    **同一个掩码**（都等价于 ``|Σ| ≥ 4``），扫描时不要把它们当两个点。
    """
    from math import comb

    n = int(n_clients)
    keep = sum(comb(n, j) for j in range(n + 1)
               if abs(2 * j - n) / n > float(tau))
    return keep / (2 ** n)


def and_mask(consistency: torch.Tensor, tau: float,
             mode: str = "binary", n_clients: int = 1) -> torch.Tensor:
    """``m = 1[consistency > τ]``，**严格大于**。

    ``mode="one_over_n"`` 是论文字面版本（掩码带 ``1/N`` 因子），会把整个聚合
    结果缩放 ``1/N``。判为笔误，仅保留供对照，见模块 docstring。
    """
    if mode not in MASK_MODES:
        raise ValueError(f"未知的 mask_mode '{mode}'；可选 {list(MASK_MODES)}")
    mask = (consistency > float(tau)).to(consistency.dtype)
    if mode == "one_over_n":
        mask = mask / float(n_clients)
    return mask


def trim_count(n_clients: int, alpha: float) -> int:
    """每端要去掉的个数 ``⌈αN⌉``。"""
    return int(math.ceil(float(alpha) * int(n_clients)))


def trimmed_mean(stacked: torch.Tensor, alpha: float
                 ) -> Tuple[torch.Tensor, int]:
    """逐维度的 trimmed mean：排序后两端各去 ``⌈αN⌉`` 个，剩下的求均值。

    Returns ``(mean, n_effective)``。``n_effective`` 是实际参与平均的客户端数，
    必须逐轮报告（陷阱 7）：N=10、α=0.25 时它只有 4。

    剩余为 0 时**直接报错**，不静默退化成别的东西。
    """
    n_clients = stacked.shape[0]
    k = trim_count(n_clients, alpha)
    n_effective = n_clients - 2 * k
    if n_effective <= 0:
        raise ValueError(
            f"trim 把样本全部去光了：N={n_clients}, alpha={alpha} -> 每端去 {k} 个，"
            f"剩 {n_effective} 个。请降低 alpha 或增大每轮选中的客户端数。")
    if k == 0:
        return stacked.mean(dim=0), n_clients
    ordered, _ = torch.sort(stacked, dim=0)
    return ordered[k:n_clients - k].mean(dim=0), n_effective


def _plain_mean(state_dicts: Sequence[Dict[str, torch.Tensor]], key: str
                ) -> torch.Tensor:
    """简单平均，保留原 dtype，且**不写入**任何输入张量。"""
    reference = state_dicts[0][key]
    if not torch.is_tensor(reference):
        return copy.deepcopy(reference)
    accumulator = torch.stack(
        [state[key].detach().to(torch.float64) for state in state_dicts], dim=0)
    return accumulator.mean(dim=0).to(reference.dtype)


# ---------------------------------------------------------------------------
# 一轮聚合
# ---------------------------------------------------------------------------
def invariant_aggregate(w_prev: Dict[str, torch.Tensor],
                        state_dicts: Sequence[Dict[str, torch.Tensor]],
                        tau: float, trim_alpha: float, *,
                        mask_mode: str = "binary",
                        sign_zero_policy: str = "count",
                        round_index: int = -1
                        ) -> Tuple[Dict[str, torch.Tensor], RoundStats]:
    """跑一轮 Invariant Aggregator，返回 ``(新的参数字典, 统计)``。

    返回的字典**包含全部 key**（共享的 + FedBN 私有的），与 ``agg_avg`` 的输出
    结构一致，好让 ``fedbn_update`` 照常 pop —— FedBN 的行为完全不变。
    """
    if not state_dicts:
        raise ValueError("没有收到任何客户端更新")

    shared_float, shared_nonfloat, private = split_keys(state_dicts[0])
    n_clients = len(state_dicts)
    stats = RoundStats(
        round_index=int(round_index), n_clients=n_clients,
        n_shared_float_keys=len(shared_float),
        n_nonfloat_fallback_keys=len(shared_nonfloat),
        nonfloat_fallback_keys=list(shared_nonfloat),
        n_fedbn_private_keys=len(private),
        tau=float(tau), trim_alpha=float(trim_alpha),
        trim_k=trim_count(n_clients, trim_alpha),
        mask_mode=str(mask_mode), sign_zero_policy=str(sign_zero_policy))

    update: Dict[str, torch.Tensor] = {}

    # --- FedBN 私有 + 非浮点：简单平均 ------------------------------------
    for key in shared_nonfloat + private:
        update[key] = _plain_mean(state_dicts, key)

    if not shared_float:
        return update, stats

    # --- 共享浮点参数：伪梯度 -> 掩码 + trimmed mean -> 回到参数 ----------
    stacked = pseudo_gradients(w_prev, state_dicts, shared_float)

    kept, total_dims, zero_entries, consistency_sum, effective = 0, 0, 0, 0.0, 0
    for key, tensor in stacked.items():
        consistency, zeros = sign_consistency(tensor, sign_zero_policy)
        mask = and_mask(consistency, tau, mode=mask_mode, n_clients=n_clients)
        centre, effective = trimmed_mean(tensor, trim_alpha)
        aggregated = mask * centre

        reference = w_prev[key]
        update[key] = (reference.detach().to(torch.float32) - aggregated
                       ).to(reference.dtype)

        numel = consistency.numel()
        total_dims += numel
        kept += int((mask > 0).sum())
        zero_entries += int(zeros.sum())
        consistency_sum += float(consistency.sum())

    stats.n_shared_float_dims = total_dims
    stats.effective_trim_n = effective
    stats.mask_keep_ratio = kept / total_dims if total_dims else float("nan")
    stats.zero_sign_ratio = (zero_entries / (total_dims * n_clients)
                             if total_dims else float("nan"))
    stats.mean_sign_consistency = (consistency_sum / total_dims if total_dims
                                   else float("nan"))
    return update, stats


# ---------------------------------------------------------------------------
# 接入 server
# ---------------------------------------------------------------------------
class InvariantAggregatorHandle:
    """``use_invariant_aggregator`` 的返回值：统计历史 + 还原入口。"""

    def __init__(self, server: Any, original: Any):
        self.server = server
        self._original = original
        self.history: List[RoundStats] = []
        self.round_index = 0

    def detach(self) -> None:
        """还原成仓库原本的 ``agg_and_update``（供测试与对照实验用）。"""
        if self._original is None:
            return
        self.server.agg_and_update = self._original
        self._original = None

    def rows(self) -> List[Dict[str, Any]]:
        return [stats.to_row() for stats in self.history]


def use_invariant_aggregator(server: Any, tau: float, trim_alpha: float, *,
                             mask_mode: str = "binary",
                             sign_zero_policy: str = "count",
                             verify_w_prev: bool = True
                             ) -> InvariantAggregatorHandle:
    """把 ``server.agg_and_update`` 换成 Invariant Aggregator。

    **不改动客户端侧任何代码，攻击保持原样**（§2.3）。绑定的是**实例方法**，
    ``BasicServer`` 类本身不受影响，原仓库文件仍为零 diff。

    # w_{t-1} 从哪里来

    直接读 ``server.global_model.state_dict()``。这是安全的，而且比在
    ``on_round_begin`` 存快照更不容易出错：``fl_process.py:36`` 调用
    ``agg_and_update`` 时，全局模型**还没有**被本轮更新过 ——
    ``load_state_dict`` 在该函数的最后一行才执行。

    ``verify_w_prev=True`` 时额外在 ``on_round_begin`` 存一份深拷贝，聚合时比对；
    不一致就报错。这条断言的作用是：**如果将来有别的代码在轮次中途改动了全局
    模型，上面那个前提就不再成立，而它失效时不会有任何症状。**
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"未知的 mask_mode '{mask_mode}'")
    if sign_zero_policy not in SIGN_ZERO_POLICIES:
        raise ValueError(f"未知的 sign_zero_policy '{sign_zero_policy}'")

    handle = InvariantAggregatorHandle(server, server.agg_and_update)
    snapshot: Dict[str, Optional[Dict[str, torch.Tensor]]] = {"w_prev": None}

    if verify_w_prev:
        from event_emitter import fl_event_emitter

        def _snapshot(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            snapshot["w_prev"] = {
                key: value.detach().cpu().clone()
                for key, value in server.global_model.state_dict().items()}
            handle.round_index = int(kwargs.get("cur_round", 0))

        fl_event_emitter.on("on_round_begin", _snapshot)
        original_detach = handle.detach

        def _detach() -> None:
            fl_event_emitter.off("on_round_begin", _snapshot)
            original_detach()

        handle.detach = _detach  # type: ignore[method-assign]

    @torch.no_grad()
    def agg_and_update(self, state_dicts) -> None:
        w_prev = self.global_model.state_dict()
        if verify_w_prev and snapshot["w_prev"] is not None:
            for key, expected in snapshot["w_prev"].items():
                if not torch.equal(expected, w_prev[key].detach().cpu()):
                    raise RuntimeError(
                        f"聚合时 server.global_model 已不是本轮下发的 w_(t-1)"
                        f"（'{key}' 不一致）。伪梯度 g = w_(t-1) − w_i 的前提"
                        f"失效，必须中止 —— 这个错误不会有任何其他症状。")

        self.update, stats = invariant_aggregate(
            w_prev, state_dicts, tau, trim_alpha, mask_mode=mask_mode,
            sign_zero_policy=sign_zero_policy, round_index=handle.round_index)
        handle.history.append(stats)

        # 与 server.py:33-34 完全一致：先跑注册的钩子（FedBN 在这里 pop 掉
        # 私有 key），再 load。FedBN 的行为一位不变。
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

    server.agg_and_update = types.MethodType(agg_and_update, server)
    return handle
