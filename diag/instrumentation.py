"""实验 I 与实验 J 共用的逐轮埋点层。

# 为什么是同一个模块

实验 I §5 要求"本节的所有定义必须与实验 J §1、§2、§5 逐字一致，否则两个实验的
结果无法并表"。各写一份必然漂移，而**漂移是静默的** —— 两张表看起来都对，
并到一起才是错的。所以定义只有这一份，两边都从这里 import。

# 三条必须记住的口径

1. **符号约定不影响任何指标。** 实验 J §2.1 用 ``Δ_i = w_i − w_{t-1}``
   （新减旧），实验 I §2.2 的伪梯度用 ``g_i = w_{t-1} − w_i``（旧减新）。
   两者差一个整体负号，而 ``update_norm`` / ``l2_to_median`` / ``cos_to_median``
   对整体取负**完全不变**（中位数同时翻号，余弦对两个向量同时取负不变）。
   本模块内部统一用 ``g = w_prev − w``，与聚合器一致；有测试固定这一等价性。

2. **只看共享参数。** key 过滤走 ``diag.fedbn``，与 ``pfl.py:8/17`` 同一份逻辑。
   **这一条不是形式要求。** FedBN 从不聚合 BN，而全局模型的 BN 停在初始化值，
   所以把 BN 算进距离会让"距离"主要在度量客户端本地数据分布带来的 BN 漂移
   —— 那是服务器端聚合规则根本不作用的坐标块。参见 ``diag/README.md`` §4b。

3. **不存更新向量。** 10 客户端 × 1000 轮 × ResNet-10 ≈ 200GB。每轮只落盘标量。
   为此所有量都按 key **流式累加**（L2 可按 key 累加平方和，余弦可累加内积与
   模长，影响力可累加坐标计数），全程不构造 ``[N, D]`` 的展平矩阵。

# 设备

模型在 GPU 上时，逐 key 的中间张量也在 GPU 上。**累加器的设备必须显式处理**，
否则会在 ``accumulator += gpu_tensor`` 处抛
``Expected all tensors to be on the same device``。本模块的约定是：

- ``rank_window_fraction`` 内部的计数器跟随输入设备（它要和 ``bincount``
  的输出相加）；
- ``round_signals`` 的累加器留在 **CPU**，每个 key 只把长度为 N 的归约结果
  ``.cpu()`` 搬回来。大张量始终留在 GPU，又不必在 GPU 上分配 float64 副本。

精度上用 ``sum(dtype=torch.float64)`` 在 float32 数据上做 float64 累加，
避免把 ``[N, D]`` 整个转成 float64（ResNet-10 上约 400MB）。

# 统一的影响力 I^(k)（实验 J §1.1 / 实验 I §5.2）

======================  ==============================================
防御                     I^(k)
======================  ==============================================
FedAvg                  恒为 1
Median                  该客户端取值落在"中间两位"的坐标比例
Invariant Aggregator    未被 trim 掉的坐标比例 × 掩码保留率
FLAME                   1 if 在良性簇内 else 0
Multi-Krum              1 if 被选中 else 0
======================  ==============================================

Median 与 Invariant Aggregator 的 trim 存活率共用同一个原语
``rank_window_fraction``：两者都是"排序后落在某个名次窗口内的坐标比例"，
只是窗口不同。写成两份代码必然会在并列时对不上。

# 不做客户端级二元决策的防御没有 TPR/FPR

Median 与 Invariant Aggregator **不判断"这个客户端是恶意的"**，
所以这三列恒为 ``None``（写进 CSV 就是空），**不得用 0 或 "N/A" 填充后
当数值参与统计**（实验 J 陷阱 1、实验 I 陷阱 9）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .fedbn import split_keys

__all__ = ["DEFENSES_WITH_CLIENT_DECISION", "RANK_SIGNALS", "RoundRecord",
           "rank_window_fraction", "median_influence", "trim_survival",
           "compose_influence", "round_signals", "RoundRecorder",
           "load_round_records", "rank_distribution", "pooled_auc",
           "influence_summary"]

# 只有这两类防御会输出"这个客户端是恶意的"这种判断
DEFENSES_WITH_CLIENT_DECISION: Tuple[str, ...] = ("flame", "multi_krum")

# 名次分布要报的三个信号（实验 J §3.3：另外两个作对照）
RANK_SIGNALS: Tuple[str, ...] = ("l2_to_median", "update_norm", "cos_to_median")


# ---------------------------------------------------------------------------
# 逐轮记录
# ---------------------------------------------------------------------------
@dataclass
class RoundRecord:
    """一轮的全部标量埋点。字段名与实验 J §2.2 / 实验 I §5.3 逐字对应。"""

    round_index: int
    client_ids: np.ndarray
    is_malicious: np.ndarray
    update_norm: np.ndarray
    l2_to_median: np.ndarray
    cos_to_median: np.ndarray
    influence: np.ndarray
    trim_survival_rate: Optional[np.ndarray] = None
    mask_keep_ratio: float = float("nan")
    zero_sign_ratio: float = float("nan")
    effective_trim_n: int = -1
    n_malicious_in_round: int = 0
    n_selected: int = 0
    defense: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_npz(self) -> Dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "extra"}
        payload["extra_json"] = json.dumps(self.extra, ensure_ascii=False,
                                           default=str)
        return {k: v for k, v in payload.items() if v is not None}


# ---------------------------------------------------------------------------
# 名次窗口：Median 的"中间两位"与 trimmed-mean 的"存活"是同一个原语
# ---------------------------------------------------------------------------
def rank_window_fraction(stacked: torch.Tensor, low: int, high: int
                         ) -> torch.Tensor:
    """每个客户端有多大比例的坐标，其排序名次落在 ``[low, high)`` 内。

    ``stacked`` 形状 ``[N, ...]``，返回形状 ``[N]``、取值 ∈ [0, 1]。
    名次是 0-indexed 的升序名次。

    并列时按 ``argsort`` 的稳定顺序分配名次 —— 不做"并列均分"，因为
    Median / trimmed-mean 的实现本身也是按排序位置取值的，指标必须与
    聚合器实际做的事一致，而不是与某个理想化的定义一致。
    """
    n_clients = stacked.shape[0]
    if not 0 <= low <= high <= n_clients:
        raise ValueError(f"名次窗口 [{low}, {high}) 超出 N={n_clients}")
    flat = stacked.reshape(n_clients, -1)
    order = torch.argsort(flat, dim=0)          # order[r, d] = 名次 r 的客户端
    # 累加器必须与数据同设备（见模块 docstring 的"设备"一节）
    counts = torch.zeros(n_clients, dtype=torch.float64, device=flat.device)
    for rank in range(low, high):
        counts += torch.bincount(order[rank], minlength=n_clients).to(counts.dtype)
    total = flat.shape[1]
    if not total:
        return torch.zeros(n_clients, dtype=torch.float64, device=flat.device)
    return counts / total


def median_influence(stacked: torch.Tensor) -> torch.Tensor:
    """Median 的 I^(k)：取值落在"中间两位"的坐标比例（实验 J §1.1）。

    N 为偶数时中间两位是 0-indexed 名次 ``N/2 − 1`` 与 ``N/2``；
    N 为奇数时只有一位（``N//2``）—— 此时"中间两位"无定义，取那一位。
    """
    n_clients = stacked.shape[0]
    if n_clients % 2 == 0:
        low, high = n_clients // 2 - 1, n_clients // 2 + 1
    else:
        low, high = n_clients // 2, n_clients // 2 + 1
    return rank_window_fraction(stacked, low, high)


def trim_survival(stacked: torch.Tensor, trim_k: int) -> torch.Tensor:
    """trimmed-mean 的存活率：未被两端各去 ``trim_k`` 个裁掉的坐标比例。"""
    n_clients = stacked.shape[0]
    return rank_window_fraction(stacked, trim_k, n_clients - trim_k)


def compose_influence(defense: str, *, n_clients: int,
                      median_influence_values: Optional[np.ndarray] = None,
                      trim_survival_rate: Optional[np.ndarray] = None,
                      mask_keep_ratio: Optional[float] = None,
                      selected: Optional[np.ndarray] = None) -> np.ndarray:
    """按防御类型组装 I^(k)。**五种防御的映射只在这里定义一次。**

    散在各处写必然在并表时对不上（实验 J §1.1 表格就是这一份代码的规格）。
    """
    defense = str(defense).lower()
    if defense in ("fedavg", "avg", "none"):
        return np.ones(n_clients, dtype=float)
    if defense == "median":
        if median_influence_values is None:
            raise ValueError("median 的 I^(k) 需要 median_influence_values")
        return np.asarray(median_influence_values, dtype=float)
    if defense in ("invariant", "invariant_aggregator", "ia"):
        if trim_survival_rate is None or mask_keep_ratio is None:
            raise ValueError(
                "Invariant Aggregator 的 I^(k) 需要 trim_survival_rate 与 "
                "mask_keep_ratio（实验 I §5.2：两者相乘）")
        return np.asarray(trim_survival_rate, dtype=float) * float(mask_keep_ratio)
    if defense in DEFENSES_WITH_CLIENT_DECISION:
        if selected is None:
            raise ValueError(f"{defense} 的 I^(k) 需要 selected 布尔数组")
        return np.asarray(selected, dtype=bool).astype(float)
    raise ValueError(
        f"未知的防御 '{defense}'；已定义 I^(k) 的有 fedavg / median / "
        f"invariant / {' / '.join(DEFENSES_WITH_CLIENT_DECISION)}")


# ---------------------------------------------------------------------------
# 逐轮标量指标（按 key 流式计算，不展平）
# ---------------------------------------------------------------------------
def round_signals(w_prev: Dict[str, torch.Tensor],
                  state_dicts: Sequence[Dict[str, torch.Tensor]],
                  *, trim_k: Optional[int] = None
                  ) -> Dict[str, np.ndarray]:
    """算一轮的 ``update_norm`` / ``l2_to_median`` / ``cos_to_median``，
    以及 Median 与 trimmed-mean 两种影响力分量。

    只用共享浮点参数（见模块 docstring 第 2 条）。全程 fp32，按 key 累加，
    **不构造 ``[N, D]`` 的展平矩阵**（ResNet-10 上那是约 200MB，再加 argsort
    的 int64 索引接近 600MB）。

    Returns
    -------
    ``{"update_norm", "l2_to_median", "cos_to_median",
       "median_influence", "trim_survival_rate"(可选)}``，每项长度 N。
    """
    if not state_dicts:
        raise ValueError("没有收到任何客户端更新")
    shared_float, _, _ = split_keys(state_dicts[0])
    n_clients = len(state_dicts)

    # 累加器留在 CPU，每个 key 只把**长度为 N 的归约结果**搬回来（几十个 float）。
    # 这样大张量始终留在 GPU 上，又不必在 GPU 上分配 float64 副本。
    sq_norm = torch.zeros(n_clients, dtype=torch.float64)
    sq_to_median = torch.zeros(n_clients, dtype=torch.float64)
    dot_median = torch.zeros(n_clients, dtype=torch.float64)
    sq_median = torch.zeros((), dtype=torch.float64)
    median_hits = torch.zeros(n_clients, dtype=torch.float64)
    trim_hits = torch.zeros(n_clients, dtype=torch.float64)
    total_coords = 0

    for key in shared_float:
        reference = w_prev[key].detach().to(torch.float32)
        stacked = torch.stack(
            [reference - state[key].detach().to(reference.device, torch.float32)
             for state in state_dicts], dim=0)
        flat = stacked.reshape(n_clients, -1)
        median = flat.median(dim=0).values

        # 用 sum(dtype=float64) 在 float32 数据上做 float64 累加：既保精度，
        # 又不用把 [N, D] 整个转成 float64（ResNet-10 上那是约 400MB）。
        sq_norm += (flat ** 2).sum(dim=1, dtype=torch.float64).cpu()
        sq_to_median += ((flat - median) ** 2).sum(dim=1, dtype=torch.float64).cpu()
        dot_median += (flat * median).sum(dim=1, dtype=torch.float64).cpu()
        sq_median += (median ** 2).sum(dtype=torch.float64).cpu()

        numel = flat.shape[1]
        median_hits += median_influence(stacked).cpu() * numel
        if trim_k is not None:
            trim_hits += trim_survival(stacked, trim_k).cpu() * numel
        total_coords += numel

    median_norm = torch.sqrt(sq_median).clamp(min=1e-12)
    norms = torch.sqrt(sq_norm)
    result = {
        "update_norm": norms.numpy(),
        "l2_to_median": torch.sqrt(sq_to_median).numpy(),
        "cos_to_median": (dot_median / (norms.clamp(min=1e-12) * median_norm)
                          ).numpy(),
        "median_influence": (median_hits / max(total_coords, 1)).numpy(),
    }
    if trim_k is not None:
        result["trim_survival_rate"] = (trim_hits / max(total_coords, 1)).numpy()
    return result


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
class RoundRecorder:
    """把每轮的 ``RoundRecord`` 写成 ``round_{t:04d}.npz``。"""

    def __init__(self, root, run_id: str):
        self.dir = Path(root) / str(run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.records: List[RoundRecord] = []

    def write(self, record: RoundRecord) -> Path:
        path = self.dir / f"round_{int(record.round_index):04d}.npz"
        np.savez_compressed(path, **record.to_npz())
        self.records.append(record)
        return path


def load_round_records(root, run_id: str = "") -> List[Dict[str, Any]]:
    """读回一个 run 的全部逐轮 npz，按轮次排序。"""
    directory = Path(root) / str(run_id) if run_id else Path(root)
    records = []
    for path in sorted(directory.glob("round_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            record = {key: payload[key] for key in payload.files}
        if "extra_json" in record:
            record["extra"] = json.loads(str(record.pop("extra_json")))
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# 名次分布与池化 AUC（实验 J §3.2 / 实验 I §5.4）
# ---------------------------------------------------------------------------
def rank_distribution(records: Sequence[Dict[str, Any]], signal: str
                     ) -> Dict[str, Any]:
    """恶意客户端按 ``signal`` **降序**的名次分布。

    为什么不报"逐轮 AUC 的平均值"：单轮通常只有 1 个恶意客户端，AUC 的粒度
    只有 1/9，取平均没有意义（实验 J §3.2、陷阱 2）。改报名次直方图，
    并另算一个**池化**的 Mann-Whitney AUC。

    **无恶意客户端的轮次被排除并单独计数**（约占 33%，陷阱 3/11）——
    它们不是"AUC 为 0.5"，而是无定义。
    """
    ranks: List[int] = []
    n_rounds_used, n_rounds_no_malicious, n_rounds_all_malicious = 0, 0, 0
    concordant, total_pairs = 0.0, 0

    for record in records:
        if signal not in record:
            continue
        scores = np.asarray(record[signal], dtype=float)
        malicious = np.asarray(record["is_malicious"], dtype=bool)
        if malicious.sum() == 0:
            n_rounds_no_malicious += 1
            continue
        if malicious.all():
            n_rounds_all_malicious += 1
            continue
        n_rounds_used += 1

        # 降序名次，1-indexed：名次 1 = 该信号最大者
        order = np.argsort(-scores, kind="stable")
        position = np.empty(len(scores), dtype=int)
        position[order] = np.arange(1, len(scores) + 1)
        ranks.extend(position[malicious].tolist())

        # 池化的 Mann-Whitney：并列各记 0.5
        pos, neg = scores[malicious], scores[~malicious]
        comparison = pos[:, None] - neg[None, :]
        concordant += float((comparison > 0).sum() + 0.5 * (comparison == 0).sum())
        total_pairs += pos.size * neg.size

    n_clients = int(len(records[0]["is_malicious"])) if records else 0
    histogram = np.bincount(np.asarray(ranks, dtype=int),
                            minlength=n_clients + 1)[1:] if ranks else np.zeros(0)
    return {
        "signal": signal,
        "ranks": np.asarray(ranks, dtype=int),
        "histogram": histogram,
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "pooled_auc": concordant / total_pairs if total_pairs else float("nan"),
        "n_malicious_observations": len(ranks),
        "n_rounds_used": n_rounds_used,
        "n_rounds_no_malicious": n_rounds_no_malicious,
        "n_rounds_all_malicious": n_rounds_all_malicious,
    }


def pooled_auc(records: Sequence[Dict[str, Any]], signal: str) -> float:
    """``rank_distribution`` 的 AUC 分量，单独取用时的快捷方式。"""
    return float(rank_distribution(records, signal)["pooled_auc"])


def influence_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """恶意 vs 良性的平均影响力 —— "防御是否真的在起作用"的度量。

    比值 ≈ 1 说明恶意客户端对聚合结果的影响与良性客户端没有区别，
    也就是防御在这一环没有起到任何作用。

    均值池化的是 **(轮次, 客户端) 观测**，不是"逐轮均值再平均"：单轮恶意数
    服从超几何分布（0 个约 33%、1 个约 41%、≥2 个约 26%），按轮平均会让只有
    1 个恶意客户端的轮次与有 3 个的轮次权重相同。分母以
    ``n_*_observations`` 一并给出，便于复核。
    """
    malicious_values: List[float] = []
    benign_values: List[float] = []
    survival_malicious: List[float] = []
    survival_benign: List[float] = []
    n_rounds_no_malicious = 0

    for record in records:
        malicious = np.asarray(record["is_malicious"], dtype=bool)
        if malicious.sum() == 0:
            n_rounds_no_malicious += 1
        influence = np.asarray(record["influence"], dtype=float)
        malicious_values.extend(influence[malicious].tolist())
        benign_values.extend(influence[~malicious].tolist())
        if "trim_survival_rate" in record:
            survival = np.asarray(record["trim_survival_rate"], dtype=float)
            survival_malicious.extend(survival[malicious].tolist())
            survival_benign.extend(survival[~malicious].tolist())

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    malicious_mean, benign_mean = _mean(malicious_values), _mean(benign_values)
    return {
        "influence_malicious_mean": malicious_mean,
        "influence_benign_mean": benign_mean,
        "influence_ratio": (malicious_mean / benign_mean
                            if benign_mean not in (0.0,) and
                            np.isfinite(benign_mean) else float("nan")),
        "trim_survival_malicious_mean": _mean(survival_malicious),
        "trim_survival_benign_mean": _mean(survival_benign),
        "n_malicious_observations": len(malicious_values),
        "n_benign_observations": len(benign_values),
        # 这些轮次仍然计入良性侧的统计，但没有恶意侧的样本，单独报出来
        "n_rounds_no_malicious": n_rounds_no_malicious,
    }
