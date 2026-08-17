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

__all__ = ["DETECTION_COLUMNS", "detection_table", "ranks_table",
           "decay_table", "ablation_verdict", "main"]

DETECTION_COLUMNS = [
    "round", "defense", "tau", "alpha", "alpha_dirichlet", "seed",
    "n_malicious_in_round", "n_selected",
    "tpr", "fpr", "precision",
    "influence_malicious_mean", "influence_benign_mean",
    "trim_survival_malicious_mean", "trim_survival_benign_mean",
    "mask_keep_ratio", "zero_sign_ratio", "effective_trim_n",
    "malicious_rank_l2_mean", "malicious_rank_norm_mean",
    "malicious_rank_cos_mean",
    "flame_n_clusters", "flame_n_noise", "flame_max_cluster_size",
    "flame_fallback_triggered", "clip_rate_malicious", "clip_rate_benign",
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
            # 无客户端级决策的防御：三列留空，绝不填 0
            "tpr": None, "fpr": None, "precision": None,
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
            row["tpr"] = (true_positive / malicious.sum()
                          if malicious.any() else None)
            row["fpr"] = (false_positive / benign.sum() if benign.any() else None)
            row["precision"] = (true_positive / excluded.sum()
                                if excluded.any() else None)

        if "clipped" in extra:
            clipped = np.asarray(extra["clipped"], dtype=bool)
            row["clip_rate_malicious"] = _group_mean(clipped.astype(float), malicious)
            row["clip_rate_benign"] = _group_mean(clipped.astype(float), benign)
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
