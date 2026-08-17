"""实验 I 与实验 J 的汇总：逐轮检出指标、名次分布、影响力、ΔT 衰减。

从 ``run_fl --defense ... --instrument-dir ...`` 产出的
``round_*.npz`` 与 ``exp_ij_implantation_*.csv`` 读入，生成两份可并表的 CSV：

- ``exp_i_detection_*.csv`` / ``exp_j_detection_*.csv``：每行一轮
- ``exp_*_ranks_*.csv``：名次分布（实验 J §3.2 / 实验 I §5.4）

# 两条必须守住的纪律

1. **``tpr`` / ``fpr`` / ``precision`` 对 ``median`` 与 ``invariant`` 一律留空。**
   它们不做客户端级二元决策（实验 J §1 表格）。留空是**空字符串**，
   不是 0 也不是 "N/A" —— 后者会被下游当成数值参与统计。
2. **不报逐轮 AUC 的平均值。** 单轮通常只有 1 个恶意客户端，AUC 粒度只有
   1/9。改报名次分布 + 池化 Mann-Whitney AUC。0 恶意的轮次排除并计数。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .instrumentation import (DEFENSES_WITH_CLIENT_DECISION, RANK_SIGNALS,
                              influence_summary, load_round_records,
                              rank_distribution)
from .invariant_agg import chance_mask_keep_ratio

__all__ = ["DETECTION_COLUMNS", "detection_table", "ranks_table",
           "decay_table", "ablation_verdict", "mask_diagnosis", "main"]

DETECTION_COLUMNS = [
    "round", "defense", "tau", "alpha", "alpha_dirichlet", "seed",
    "n_malicious_in_round", "n_selected",
    "tpr", "fpr", "youden_j", "n_excluded",
    "influence_malicious_mean", "influence_benign_mean",
    "trim_survival_malicious_mean", "trim_survival_benign_mean",
    "mask_keep_ratio", "zero_sign_ratio", "effective_trim_n",
    "malicious_rank_l2_mean", "malicious_rank_norm_mean",
    "malicious_rank_cos_mean",
    "flame_n_clusters", "flame_n_noise", "flame_max_cluster_size",
    "flame_fallback_triggered", "clip_rate_malicious", "clip_rate_benign",
    "cos_to_others_malicious_mean", "cos_to_others_benign_mean",
]


def _group_mean(values: np.ndarray, mask: np.ndarray) -> float:
    picked = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    picked = picked[np.isfinite(picked)]
    return float(picked.mean()) if picked.size else float("nan")


def _descending_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    position = np.empty(len(scores), dtype=int)
    position[order] = np.arange(1, len(scores) + 1)
    return position


def detection_table(records: Sequence[Dict[str, Any]], *,
                    tau: float = float("nan"),
                    trim_alpha: float = float("nan"),
                    alpha_dirichlet: float = float("nan"),
                    seed: int = 0) -> pd.DataFrame:
    """每行一轮的检出指标。无定义的列填 ``None``（写进 CSV 就是空）。"""
    rows: List[Dict[str, Any]] = []
    for record in records:
        malicious = np.asarray(record["is_malicious"], dtype=bool)
        benign = ~malicious
        defense = str(record.get("defense", ""))
        extra = record.get("extra", {}) or {}

        row: Dict[str, Any] = {
            "round": int(record["round_index"]), "defense": defense,
            "tau": float(extra.get("tau", tau)),
            "alpha": float(extra.get("trim_alpha", trim_alpha)),
            "alpha_dirichlet": float(alpha_dirichlet), "seed": int(seed),
            "n_malicious_in_round": int(malicious.sum()),
            "n_selected": int(len(malicious)),
            "influence_malicious_mean": _group_mean(record["influence"], malicious),
            "influence_benign_mean": _group_mean(record["influence"], benign),
            "mask_keep_ratio": float(record.get("mask_keep_ratio", np.nan)),
            "zero_sign_ratio": float(record.get("zero_sign_ratio", np.nan)),
            "effective_trim_n": int(record.get("effective_trim_n", -1)),
            # 无客户端级决策的防御：这几列留空，绝不填 0
            "tpr": None, "fpr": None, "precision": None,
            "youden_j": None, "n_excluded": None,
            "trim_survival_malicious_mean": None,
            "trim_survival_benign_mean": None,
            "flame_n_clusters": None, "flame_n_noise": None,
            "flame_max_cluster_size": None, "flame_fallback_triggered": None,
            "clip_rate_malicious": None, "clip_rate_benign": None,
        }

        if "trim_survival_rate" in record:
            survival = np.asarray(record["trim_survival_rate"], dtype=float)
            row["trim_survival_malicious_mean"] = _group_mean(survival, malicious)
            row["trim_survival_benign_mean"] = _group_mean(survival, benign)

        for signal, column in (("l2_to_median", "malicious_rank_l2_mean"),
                               ("update_norm", "malicious_rank_norm_mean"),
                               ("cos_to_median", "malicious_rank_cos_mean")):
            if signal in record and malicious.any():
                ranks = _descending_ranks(record[signal])
                row[column] = float(ranks[malicious].mean())
            else:
                row[column] = None

        # 只有做二元决策的防御才有 TPR/FPR（实验 J §1）
        selected = extra.get("selected", None)
        if defense in DEFENSES_WITH_CLIENT_DECISION and selected is not None:
            excluded = ~np.asarray(selected, dtype=bool)   # 被排除 = 判为恶意
            true_positive = int((excluded & malicious).sum())
            false_positive = int((excluded & benign).sum())
            tpr = true_positive / malicious.sum() if malicious.any() else None
            fpr = false_positive / benign.sum() if benign.any() else None
            row["tpr"] = tpr
            row["fpr"] = fpr
            row["n_excluded"] = int(excluded.sum())
            row["precision"] = (true_positive / excluded.sum()
                                if excluded.any() else None)
            # Youden's J = TPR − FPR。**单看 TPR 没有意义**：排除集的大小由
            # FLAME 的 min_cluster_size / Multi-Krum 的 c 决定而不是由数据决定，
            # 多排除几个就能把 TPR 抬上去。排除集与"是否恶意"无关时
            # E[TPR] = E[FPR] = 排除集大小/N，于是 J 的零假设恰好是 0。
            row["youden_j"] = (tpr - fpr if tpr is not None and fpr is not None
                               else None)

        if "clipped" in extra:
            clipped = np.asarray(extra["clipped"], dtype=bool)
            row["clip_rate_malicious"] = _group_mean(clipped.astype(float), malicious)
            row["clip_rate_benign"] = _group_mean(clipped.astype(float), benign)
        if "cos_to_others_mean" in extra:
            # FLAME 聚类环节的直接证据（实验 J §4.2）：恶意与良性的余弦分布
            # 若重叠，聚类那一环就没有信号可用。
            cosine = np.asarray(extra["cos_to_others_mean"], dtype=float)
            row["cos_to_others_malicious_mean"] = _group_mean(cosine, malicious)
            row["cos_to_others_benign_mean"] = _group_mean(cosine, benign)
        for key in ("flame_n_clusters", "flame_n_noise", "flame_max_cluster_size",
                    "flame_fallback_triggered"):
            if key in extra:
                row[key] = extra[key]
        rows.append(row)

    frame = pd.DataFrame(rows)
    rest = [c for c in frame.columns if c not in DETECTION_COLUMNS]
    return frame[[c for c in DETECTION_COLUMNS if c in frame.columns] + rest]


def ranks_table(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """三个信号的名次分布 + 池化 AUC（实验 J §3.2）。"""
    rows = []
    for signal in RANK_SIGNALS:
        result = rank_distribution(records, signal)
        row = {k: v for k, v in result.items() if k not in ("ranks", "histogram")}
        for index, count in enumerate(result["histogram"], start=1):
            row[f"rank_{index}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def decay_table(implantation: pd.DataFrame) -> pd.DataFrame:
    """ASR 相对 ``rounds_since_last_malicious`` 的衰减（实验 J §5.2）。

    衰减速率是防御有效性的**独立**度量 —— 好的防御应加快衰减，
    而不只是压低峰值。
    """
    if implantation.empty:
        return pd.DataFrame()
    block = implantation[implantation["rounds_since_last_malicious"] >= 0]
    rows = []
    for (defense, gap), group in block.groupby(
            ["defense", "rounds_since_last_malicious"]):
        row: Dict[str, Any] = {"defense": defense, "delta_t": int(gap),
                               "n": int(len(group))}
        for column in ("asr_global_targeted", "asr_personalized_targeted",
                       "acc_global", "acc_personalized"):
            if column in group.columns:
                values = group[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[f"{column}_mean"] = (float(values.mean()) if values.size
                                         else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["defense", "delta_t"]).reset_index(
        drop=True)


def mask_diagnosis(detection: pd.DataFrame) -> Dict[str, Any]:
    """AND-mask 到底在做什么：与"随机符号"基线比，以及是否对攻击者有反应。

    两个必须一起看的量：

    1. ``mask_keep_ratio`` **相对随机基线的超出量**。掩码的前提是符号一致性
       能指示"该方向对多数客户端有益"；若实测保留率贴着随机基线，
       说明客户端之间根本没有可利用的符号一致性，掩码退化成随机稀疏化。
       这与"后门维度混过了掩码"是**两种不同的失效方式**，结论也不同。
    2. **有/无恶意客户端的轮次之间，保留率是否有差别**。若掩码在抓攻击者，
       攻击者在场时被掩掉的维度应当更多；若两者一样，掩码对攻击**没有反应**。
    """
    if detection.empty or "mask_keep_ratio" not in detection.columns:
        return {"verdict": "未能确定：检出表里没有 mask_keep_ratio"}

    observed = detection["mask_keep_ratio"].astype(float)
    if not np.isfinite(observed).any():
        return {"verdict": "未能确定：mask_keep_ratio 全为 nan（该防御没有掩码）"}

    n_selected = int(detection["n_selected"].mode().iloc[0])
    tau = float(detection["tau"].mode().iloc[0])
    chance = chance_mask_keep_ratio(n_selected, tau)
    mean_observed = float(observed.mean())

    with_malicious = detection[detection["n_malicious_in_round"] > 0]
    without = detection[detection["n_malicious_in_round"] == 0]
    keep_with = (float(with_malicious["mask_keep_ratio"].mean())
                 if len(with_malicious) else np.nan)
    keep_without = (float(without["mask_keep_ratio"].mean())
                    if len(without) else np.nan)

    excess = mean_observed - chance
    responds = (abs(keep_with - keep_without)
                if np.isfinite(keep_with) and np.isfinite(keep_without) else np.nan)

    notes = []
    if abs(excess) < 0.05 * chance:
        notes.append(
            f"保留率 {mean_observed:.4f} 贴着随机符号基线 {chance:.4f}"
            f"（超出 {excess:+.4f}）—— 客户端之间几乎没有可利用的符号一致性，"
            f"掩码退化成随机保留约 {mean_observed:.0%} 的坐标")
    if np.isfinite(responds) and responds < observed.std():
        notes.append(
            f"有恶意（{keep_with:.4f}）与无恶意（{keep_without:.4f}）的轮次"
            f"保留率相差 {responds:.4f}，小于轮间标准差 {observed.std():.4f}"
            f" —— 掩码对攻击者是否在场**没有反应**")
    return {
        "n_selected": n_selected, "tau": tau,
        "mask_keep_ratio_mean": mean_observed,
        "mask_keep_ratio_chance": chance,
        "excess_over_chance": excess,
        "keep_ratio_with_malicious": keep_with,
        "keep_ratio_without_malicious": keep_without,
        "notes": notes,
        # |Σ sign| 与 N 同奇偶，所以只能取 {N%2, N%2+2, ...}。
        # N=10 时 τ=0.2 与 τ=0.3 都等价于 |Σ| ≥ 4 —— 是**同一个掩码**。
        # （sign(0)=0 会破坏奇偶性，但实测 zero_sign_ratio ≈ 3e-06，可忽略。）
        "equivalent_threshold": min(
            (v for v in range(n_selected % 2, n_selected + 1, 2)
             if v / n_selected > tau), default=n_selected),
    }


def decision_diagnosis(detection: pd.DataFrame) -> Dict[str, Any]:
    """FLAME / Multi-Krum 的检出表现，**以"排除集与恶意无关"为零假设**。

    # 为什么不能只看 TPR

    这两个防御每轮排除的客户端数是由超参决定的，不是由数据决定的：
    FLAME 排除"最大簇之外"的全部客户端（簇的大小被 ``min_cluster_size`` 钉住），
    Multi-Krum 排除 ``N − c`` 个。**多排除几个就能把 TPR 抬上去**，
    极限是排除全部 -> TPR=1，而那显然不是检出能力。

    若排除集的选取与"是否恶意"完全无关，则

        E[TPR] = E[FPR] = 排除集大小 / N

    所以有信息量的统计量是 **Youden's J = TPR − FPR**，零假设恰好为 0，
    且与排除集大小无关。

    # FLAME 在 N=10 上的一个已知退化

    ``min_cluster_size = N//2 + 1 = 6``。实测（合成数据）HDBSCAN 在 10 个点上
    **不会**全判噪声，而是几乎总能切出一个恰好 6 个点的簇、其余 4 个标为噪声，
    **即使这 10 个向量是纯随机近正交的**。也就是说排除集的大小由超参强制，
    与数据结构无关 —— 这正是上面那个零假设要对付的情形。
    """
    if detection.empty:
        return {"verdict": "未能确定：检出表为空"}
    defense = str(detection["defense"].mode().iloc[0])
    if defense not in DEFENSES_WITH_CLIENT_DECISION:
        return {"verdict": f"'{defense}' 不做客户端级二元决策，没有 TPR/FPR "
                           f"可言（实验 J §1）。请看影响力 I^(k) 而不是检出率。"}

    usable = detection[detection["n_malicious_in_round"] > 0].copy()
    n_no_malicious = int((detection["n_malicious_in_round"] == 0).sum())
    if usable.empty:
        return {"verdict": "未能确定：没有任何一轮存在恶意客户端"}

    def _mean(column: str) -> float:
        values = usable[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    tpr, fpr, j = _mean("tpr"), _mean("fpr"), _mean("youden_j")
    excluded = _mean("n_excluded")
    n_selected = float(detection["n_selected"].mode().iloc[0])
    chance = excluded / n_selected if n_selected else float("nan")

    notes = []
    if np.isfinite(j) and abs(j) < 0.05:
        notes.append(
            f"Youden's J = {j:+.4f} ≈ 0 —— 排除集与「是否恶意」基本无关。"
            f"TPR {tpr:.4f} 只是「每轮排掉 {excluded:.1f}/{n_selected:.0f} 个」"
            f"的副产物（该情形下的期望值 {chance:.4f}）")
    if "flame_fallback_triggered" in usable.columns:
        fallback = usable["flame_fallback_triggered"].astype(float)
        fallback = fallback[np.isfinite(fallback)]
        if fallback.size:
            rate = float(fallback.mean())
            notes.append(f"HDBSCAN 回退（全部判为噪声 -> 全部接受）触发率 "
                         f"{rate:.1%}"
                         + ("；这些轮次 FLAME 实际上没有生效" if rate > 0 else ""))
    return {
        "defense": defense, "tpr": tpr, "fpr": fpr, "youden_j": j,
        "n_excluded_mean": excluded, "n_selected": n_selected,
        "chance_tpr_fpr": chance,
        "n_rounds_used": int(len(usable)),
        "n_rounds_no_malicious": n_no_malicious,
        "notes": notes,
    }


def ablation_verdict(implantation: pd.DataFrame, *, last_n: int = 5
                     ) -> Dict[str, Any]:
    """实验 I §3.1 的硬检查点：单组件必须**显著**劣于组合。

    取每个变体最后 ``last_n`` 个评估点的平均 ASR（个性化模型）。

    # 为什么只看"单组件 > 组合"是不够的

    §3.1 的原话是"单组件必须**显著**劣于组合"，论文 Table 4 的落差是
    .001 vs .655/.512 —— 组合把 ASR 压到了地板上。如果三个变体都停在 0.9 以上，
    "单组件 0.99 > 组合 0.91"这个排序**不构成实现正确的证据**：
    它同样可以由"防御对这个攻击完全无效"产生，而那正是本检查点要排除的另一种
    可能。只报排序会把一次**无区分力**的检查说成"通过"。

    因此判据分三档，并且**以无防御的 FedAvg 为参照**（而不是某个拍脑袋的常数）：

    - 组合明显低于 FedAvg **且**两个单组件都高于组合 → 通过
    - 排序满足但组合并未把 ASR 压下来 → **未能确定**（检查无区分力），
      此时既不能说实现正确，也不能说实现有误
    - 排序被违反 → 未通过，实现很可能有误

    ⚠️ 这里用的是 Bad-PFL 本身，**不是论文 Table 4 的边缘案例攻击**，
    因此拿不到 .001 / .655 / .512 三个绝对数字做对照。这一点必须随结论一起报告。
    """
    needed = {"invariant", "invariant_mask_only", "invariant_trim_only"}
    present = set(implantation["defense"].unique())
    missing = sorted(needed - present)
    caveat = ("消融用的是 Bad-PFL 而非论文 Table 4 的边缘案例攻击，"
              "只能检验相对关系，无法与 .001/.655/.512 直接对照")
    if missing:
        return {"verdict": f"未能确定：缺少变体 {missing}",
                "missing": missing, "available": sorted(present),
                "caveat": caveat}

    def _tail_mean(name: str) -> float:
        block = implantation[implantation["defense"] == name].sort_values("round")
        values = block["asr_personalized_targeted"].to_numpy(dtype=float)
        values = values[np.isfinite(values)][-int(last_n):]
        return float(values.mean()) if values.size else float("nan")

    asr = {name: _tail_mean(name) for name in sorted(needed)}
    baseline = _tail_mean("fedavg") if "fedavg" in present else float("nan")
    asr["fedavg (no defence)"] = baseline

    combined = asr["invariant"]
    singles = {k: v for k, v in asr.items()
               if k.startswith("invariant_")}
    worse = {k: v > combined for k, v in singles.items()}
    ordering_ok = all(worse.values())
    gaps = {k: v - combined for k, v in singles.items()}

    result: Dict[str, Any] = {
        "asr": asr, "single_component_worse_than_combined": worse,
        "gap_vs_combined": gaps, "fedavg_baseline": baseline,
        "caveat": caveat,
    }

    if not ordering_ok:
        result.update({
            "passed": False,
            "verdict": ("**未通过**：存在单组件的 ASR 不高于组合，实现很可能有误"
                        "（最可能是掩码方向、伪梯度符号、或 trim 作用的维度）")})
        return result

    if not np.isfinite(baseline):
        result.update({
            "passed": None,
            "verdict": ("未能确定：排序满足（两个单组件都高于组合），但**缺少无防御的 "
                        "fedavg 对照**，无法判断组合是否真的把 ASR 压下来了。"
                        "全部变体都停在高位时，排序本身可以由'防御完全无效'产生，"
                        "不构成实现正确的证据。请补跑 `--defense fedavg` 同配置。")})
        return result

    # 组合相对无防御的下降幅度，与单组件之间的落差作比较
    suppression = baseline - combined
    max_gap = max(gaps.values()) if gaps else 0.0
    if suppression <= max_gap:
        result.update({
            "passed": None,
            "verdict": (f"未能确定：排序满足，但组合相对 fedavg 只降了 "
                        f"{suppression:+.4f}，不大于单组件之间的最大落差 "
                        f"{max_gap:.4f}。这次检查**没有区分力** —— 它既符合"
                        f"'实现正确但防御对 Bad-PFL 无效'，也符合'实现有误'。")})
        return result

    result.update({
        "passed": True,
        "verdict": (f"通过：组合把 ASR 从 fedavg 的 {baseline:.4f} 压到 "
                    f"{combined:.4f}（降 {suppression:.4f}），且两个单组件都更高")})
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 I / J 的汇总")
    parser.add_argument("--instrument-dir", required=True,
                        help="包含 round_*.npz 的目录（单个 run）")
    parser.add_argument("--implantation-glob", default=None,
                        help="exp_ij_implantation_*.csv 的 glob")
    parser.add_argument("--out-prefix", default="results/exp_ij")
    parser.add_argument("--alpha-dirichlet", type=float, default=float("nan"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = load_round_records(args.instrument_dir)
    if not records:
        print(f"[exp_ij] {args.instrument_dir} 下没有 round_*.npz")
        return 1

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    detection = detection_table(records, alpha_dirichlet=args.alpha_dirichlet,
                                seed=args.seed)
    detection.to_csv(f"{prefix}_detection.csv", index=False)
    print(f"[exp_ij] 检出指标 {len(detection)} 行 -> {prefix}_detection.csv")

    ranks = ranks_table(records)
    ranks.to_csv(f"{prefix}_ranks.csv", index=False)
    print(f"[exp_ij] 名次分布 -> {prefix}_ranks.csv")

    summary = influence_summary(records)
    print("\n=== 影响力（恶意 vs 良性）===")
    for key in ("influence_malicious_mean", "influence_benign_mean",
                "influence_ratio", "n_malicious_observations",
                "n_benign_observations", "n_rounds_no_malicious"):
        print(f"  {key:<32} {summary[key]}")
    print("  比值 ≈ 1 表示防御在这一环没有起作用")

    decision = decision_diagnosis(detection)
    if "verdict" not in decision:
        print("\n=== 检出表现（零假设：排除集与是否恶意无关）===")
        print(f"  TPR              {decision['tpr']:.4f}")
        print(f"  FPR              {decision['fpr']:.4f}")
        print(f"  Youden's J       {decision['youden_j']:+.4f}   ← 零假设 = 0")
        print(f"  每轮排除          {decision['n_excluded_mean']:.2f}"
              f" / {decision['n_selected']:.0f}"
              f"（无关时 TPR=FPR={decision['chance_tpr_fpr']:.4f}）")
        print(f"  用了 {decision['n_rounds_used']} 轮，"
              f"排除 0 恶意 {decision['n_rounds_no_malicious']} 轮")
        for note in decision["notes"]:
            print(f"  ⚠️ {note}")

    mask = mask_diagnosis(detection)
    if "verdict" not in mask:
        print("\n=== AND-mask 在做什么 ===")
        print(f"  实测保留率     {mask['mask_keep_ratio_mean']:.4f}")
        print(f"  随机符号基线   {mask['mask_keep_ratio_chance']:.4f}"
              f"  (N={mask['n_selected']}, τ={mask['tau']}"
              f" 等价于 |Σ sign| ≥ {mask['equivalent_threshold']})")
        print(f"  超出随机       {mask['excess_over_chance']:+.4f}")
        print(f"  有恶意 / 无恶意 {mask['keep_ratio_with_malicious']:.4f}"
              f" / {mask['keep_ratio_without_malicious']:.4f}")
        for note in mask["notes"]:
            print(f"  ⚠️ {note}")

    print("\n=== 名次分布（不报逐轮 AUC 的平均值）===")
    for _, row in ranks.iterrows():
        print(f"  {row['signal']:<14} 池化AUC={row['pooled_auc']:.4f} "
              f"平均名次={row['mean_rank']:.2f} "
              f"用了 {int(row['n_rounds_used'])} 轮，"
              f"排除 0 恶意 {int(row['n_rounds_no_malicious'])} 轮")

    if args.implantation_glob:
        import glob as globlib

        paths = sorted(globlib.glob(args.implantation_glob))
        if paths:
            implantation = pd.concat([pd.read_csv(p) for p in paths],
                                     ignore_index=True)
            decay = decay_table(implantation)
            decay.to_csv(f"{prefix}_decay.csv", index=False)
            print(f"\n[exp_ij] ΔT 衰减 {len(decay)} 行 -> {prefix}_decay.csv")
            verdict = ablation_verdict(implantation)
            print("\n=== 实验 I §3.1 消融判据 ===")
            print(f"  {verdict['verdict']}")
            if "asr" in verdict:
                for name, value in verdict["asr"].items():
                    print(f"    {name:<24} ASR={value:.4f}")
                print(f"  ⚠️ {verdict['caveat']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
