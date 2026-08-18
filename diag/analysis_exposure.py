"""步骤 0：ASR vs **累积吸收的恶意幅度** E —— 坍缩图。

# 为什么不是"漏进去几次"

按次数计数不自洽（真实数据）：

    fedavg  bad=1   漏 47 次（全额未裁剪）  -> ASR > 0.80
    FLAME   bad=5   漏 107 次              -> ASR ≈ 0.30

漏得更多反而 ASR 更低。差别在于 FLAME 把 95.1% 的恶意更新裁到中位范数再加噪,
**漏过去的那些幅度被削掉了**。所以自变量应该是幅度而不是次数：

    E = Σ_t Σ_{k∈mal} w_k(t) · ‖g_k(t)‖ ,    w_k(t) = I_k(t) · s_k(t) / D(t)

- ``I_k`` 影响力（``instrumentation.compose_influence``，五种防御各自的定义）
- ``s_k`` 裁剪系数（FLAME 的 ``norm_after_clip / norm_before_clip``，其余为 1）
- ``D``  该防御**实际的有效贡献者数**，即聚合式的分母

``D`` 逐防御给出，不用一个通用的近似糊过去：

======================  ==================  ==========================
防御                     D                   Σ_k w_k
======================  ==================  ==========================
fedavg                  N                   1
median                  2（偶数 N 的中间两位）  1
multi_krum / flame /    Σ I（被选中的个数）    flame 为 mean(s) ≤ 1（裁剪即收缩）
oracle_exclude
invariant               effective_trim_n     mask_keep_ratio ≤ 1（掩码即收缩）
======================  ==================  ==========================

于是 FLAME 的裁剪与 Invariant 的掩码这两种"不排除、只削弱"的机制
**会如实体现在 E 里**，而按次数计数会把它们完全忽略掉。

# 两条必须随图一起报的限制

1. **E 是上界，不是实际注入量。** 埋点只存了 ‖g_k‖（标量范数），没有存向量,
   所以无法计算 ‖Σ_{k∈mal} w_k g_k‖ —— 恶意更新之间、以及与良性更新之间的
   方向抵消都没有被扣掉。要精确到向量得改埋点重跑。
2. **median / invariant 的 E 是近似。** 它们逐坐标地取舍，用一个标量影响力
   乘以整条更新的范数，等于假设吸收在各坐标上是均匀的。fedavg / flame /
   multi_krum / oracle_exclude 的权重则是精确的。CSV 里有
   ``exposure_is_exact`` 列，图上用空心标记区分，**不要混在一起做回归**。

用法::

    python -m diag.analysis_exposure \\
        --instrument-root instrumentation \\
        --implantation-glob "results/raw/exp_ij_implantation_*.csv" \\
        --out results/figs/exp_exposure.png \\
        --summary-out results/exp_exposure.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import assert_no_cjk_in_figure
from .analysis_density import load_implantation, tail_summary
from .analysis_ij import _style
from .instrumentation import load_round_records

__all__ = ["clip_scale", "effective_denominator", "round_exposure",
           "exposure_by_run", "join_with_asr", "collapse_diagnosis",
           "plot_exposure", "main"]

# 权重分母精确已知的防御；其余的 E 是近似（逐坐标取舍被标量化了）
_EXACT_DEFENSES = ("fedavg", "avg", "none", "multi_krum", "flame",
                   "oracle_exclude")


def clip_scale(record: Dict[str, Any]) -> np.ndarray:
    """每个客户端的裁剪系数 s_k ∈ (0, 1]。没有裁剪环节的防御一律返回 1。"""
    n_clients = int(np.asarray(record["influence"]).shape[0])
    extra = record.get("extra", {}) or {}
    before = extra.get("norm_before_clip")
    after = extra.get("norm_after_clip")
    if before is None or after is None:
        return np.ones(n_clients, dtype=float)
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape[0] != n_clients or after.shape[0] != n_clients:
        raise ValueError(
            f"裁剪范数长度 {before.shape[0]}/{after.shape[0]} 与客户端数 "
            f"{n_clients} 不一致（round {record.get('round_index')}）")
    # before=0 只可能出现在更新完全为零的退化轮次，此时 s 取 1 不影响 E（‖g‖=0）
    return np.where(before > 0, after / np.maximum(before, 1e-12), 1.0)


def effective_denominator(record: Dict[str, Any]) -> float:
    """聚合式里的分母 D —— 该防御实际的有效贡献者数。"""
    defense = str(record.get("defense", "")).lower()
    influence = np.asarray(record["influence"], dtype=float)
    n_clients = int(influence.shape[0])

    if defense in ("fedavg", "avg", "none"):
        return float(n_clients)
    if defense == "median":
        # 偶数 N 取中间两位的均值，奇数 N 只有一位
        return 2.0 if n_clients % 2 == 0 else 1.0
    if defense in ("invariant", "invariant_aggregator", "ia",
                   "invariant_mask_only", "invariant_trim_only"):
        effective = float(record.get("effective_trim_n", -1))
        if not np.isfinite(effective) or effective <= 0:
            raise ValueError(
                f"invariant 的 E 需要 effective_trim_n，但记录里是 {effective}"
                f"（round {record.get('round_index')}）")
        return effective
    # 二元决策类：分母就是被选中的个数
    total = float(influence.sum())
    if total <= 0:
        # 一个都没选中：这一轮没有任何更新被吸收，E 贡献为 0
        return float("nan")
    return total


def round_exposure(record: Dict[str, Any]) -> Dict[str, float]:
    """一轮的恶意 / 良性吸收幅度。"""
    influence = np.asarray(record["influence"], dtype=float)
    norms = np.asarray(record["update_norm"], dtype=float)
    malicious = np.asarray(record["is_malicious"], dtype=bool)
    if not (influence.shape == norms.shape == malicious.shape):
        raise ValueError(
            f"round {record.get('round_index')} 的三个数组长度不一致："
            f"influence={influence.shape} update_norm={norms.shape} "
            f"is_malicious={malicious.shape}")

    scale = clip_scale(record)
    denominator = effective_denominator(record)
    if not np.isfinite(denominator):
        weights = np.zeros_like(influence)
    else:
        weights = influence * scale / denominator

    contribution = weights * norms
    return {
        "round_index": int(record.get("round_index", -1)),
        "exposure_malicious": float(contribution[malicious].sum()),
        "exposure_benign": float(contribution[~malicious].sum()),
        "n_malicious": int(malicious.sum()),
        "weight_sum": float(weights.sum()),
        "mean_clip_scale_malicious": (float(scale[malicious].mean())
                                      if malicious.any() else float("nan")),
    }


def exposure_by_run(instrument_root, run_id: str = "") -> Dict[str, Any]:
    """把一个 run 的逐轮 npz 累加成一个 E。"""
    records = load_round_records(instrument_root, run_id)
    if not records:
        raise FileNotFoundError(
            f"{Path(instrument_root) / run_id} 下没有 round_*.npz")

    per_round = [round_exposure(record) for record in records]
    frame = pd.DataFrame(per_round)
    defense = str(records[0].get("defense", "")).lower()
    return {
        "run_id": run_id or Path(instrument_root).name,
        "defense": defense,
        "exposure_malicious": float(frame["exposure_malicious"].sum()),
        "exposure_benign": float(frame["exposure_benign"].sum()),
        "n_rounds": int(len(frame)),
        "n_malicious_observations": int(frame["n_malicious"].sum()),
        "n_rounds_no_malicious": int((frame["n_malicious"] == 0).sum()),
        "mean_weight_sum": float(frame["weight_sum"].mean()),
        "mean_clip_scale_malicious": float(
            frame["mean_clip_scale_malicious"].mean(skipna=True)),
        "exposure_is_exact": defense in _EXACT_DEFENSES,
    }


def _run_dirs(instrument_root) -> List[Path]:
    root = Path(instrument_root)
    if not root.is_dir():
        raise FileNotFoundError(f"埋点根目录不存在：{root}")
    if list(root.glob("round_*.npz")):
        return [root]                       # 直接指到了单个 run
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and list(d.glob("round_*.npz")))
    if not dirs:
        raise FileNotFoundError(f"{root} 下没有任何含 round_*.npz 的子目录")
    return dirs


def join_with_asr(instrument_root, implantation_glob: str, tail: int = 5,
                  asr_column: str = "asr_personalized_targeted",
                  acc_column: str = "acc_personalized",
                  assume_bad_num: Optional[int] = None) -> pd.DataFrame:
    """按 ``run_id`` 把 E 与尾部 ASR 并起来。

    ``run_fl`` 的埋点目录名与植入 CSV 名同源
    （``run_id`` / ``exp_ij_implantation_{run_id}.csv``），所以用它做连接键 ——
    比分别解析 defense / bad / seed 再拼回去可靠得多。
    """
    exposures = []
    for directory in _run_dirs(instrument_root):
        exposures.append(exposure_by_run(directory.parent, directory.name)
                         if directory != Path(instrument_root)
                         else exposure_by_run(directory))
    exposure = pd.DataFrame(exposures)

    frame = load_implantation(implantation_glob, assume_bad_num)
    frame["run_id"] = (frame["source_file"]
                       .str.replace(r"^exp_ij_implantation_", "", regex=True)
                       .str.replace(r"\.csv$", "", regex=True))
    # 一行一个 run：E 是逐 run 累加的，第二级不能再跨 seed 并，否则两边对不齐
    summary = tail_summary(frame, tail=tail, asr_column=asr_column,
                           acc_column=acc_column,
                           group_by=("run_id", "defense", "bad_client_num"))

    merged = summary.merge(exposure.drop(columns=["defense"]), on="run_id",
                           how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(
            "E 与 ASR 没有任何一个 run_id 对得上。埋点目录名应当形如 "
            f"'flame_attack_a0.5_s0_bad5'，植入 CSV 形如 "
            f"'exp_ij_implantation_flame_attack_a0.5_s0_bad5.csv'。\n"
            f"  埋点侧：{sorted(exposure['run_id'])}\n"
            f"  植入侧：{sorted(summary['run_id'].dropna())}")

    missing = sorted(set(exposure["run_id"]) - set(merged["run_id"]))
    if missing:
        print(f"[analysis_exposure] ⚠️ 这些 run 有 E 但没配上 ASR，已排除："
              f"{missing}")
    merged["exposure_per_round"] = (merged["exposure_malicious"]
                                    / merged["n_rounds"].clip(lower=1))
    merged["exposure_ratio_mal_over_benign"] = (
        merged["exposure_malicious"]
        / merged["exposure_benign"].replace(0.0, np.nan))
    return merged.sort_values("exposure_malicious").reset_index(drop=True)


def collapse_diagnosis(merged: pd.DataFrame) -> Dict[str, Any]:
    """ASR 是不是 E 的单调函数 —— 以及是不是**跨防御的同一条**单调函数。

    判据刻意保守：只报相关性与跨防御的一致性，**不拟合曲线**。样本量是
    十几个格子，任何参数化拟合都只是在描述噪声。
    """
    from scipy.stats import spearmanr

    exact = merged[merged["exposure_is_exact"]]
    result: Dict[str, Any] = {
        "n_cells": int(len(merged)),
        "n_cells_exact": int(len(exact)),
        "n_defenses": int(merged["defense"].nunique()),
        "n_densities": int(merged["bad_client_num"].nunique()),
    }
    if len(merged) < 4:
        result["verdict"] = (f"未能确定：只有 {len(merged)} 个格子，"
                             f"相关性没有意义。至少要 4 个 (防御, 密度) 组合。")
        return result

    rho, pvalue = spearmanr(merged["exposure_malicious"], merged["asr_mean"])
    result["spearman_rho"] = float(rho)
    result["spearman_p"] = float(pvalue)
    if len(exact) >= 4:
        rho_exact, p_exact = spearmanr(exact["exposure_malicious"],
                                       exact["asr_mean"])
        result["spearman_rho_exact_only"] = float(rho_exact)
        result["spearman_p_exact_only"] = float(p_exact)

    # 跨防御一致性：每个防御内部单独的 ρ。若各自都单调、且合并后仍单调,
    # 才谈得上"坍缩到同一条曲线"。
    per_defense = {}
    for defense, group in merged.groupby("defense"):
        if len(group) >= 3:
            rho_d, _ = spearmanr(group["exposure_malicious"], group["asr_mean"])
            per_defense[str(defense)] = float(rho_d)
    result["spearman_by_defense"] = per_defense

    consistent = (per_defense and
                  all(value > 0.5 for value in per_defense.values()))
    if np.isfinite(rho) and rho > 0.8 and consistent:
        result["verdict"] = (
            f"坍缩：ρ={rho:.3f}（p={pvalue:.3g}），且每个防御内部各自单调 —— "
            f"E 可以当作单一自变量。后续每个防御都可以化简成"
            f"「每单位 ACC 代价能降多少 E」。")
    elif np.isfinite(rho) and rho > 0.8:
        result["verdict"] = (
            f"部分坍缩：合并 ρ={rho:.3f}，但并非每个防御内部都单调 "
            f"({per_defense})。可能还有第二个自变量（候选：轮次的时间分布、"
            f"FLAME 的加噪 σ）。如实报告，不要硬拟合。")
    else:
        result["verdict"] = (
            f"不坍缩：ρ={rho:.3f}（p={pvalue:.3g}）。E 单独解释不了 ASR，"
            f"说明还有第二个自变量。**不要**据此宣称存在阈值。")
    return result


def estimate_threshold(merged: pd.DataFrame, level: float = 0.5
                       ) -> Optional[float]:
    """ASR 跨过 ``level`` 时的 E —— 仅在按 E 排序后确实穿越一次时才给值。

    多次穿越说明不单调，此时返回 ``None`` 而不是取第一次 —— 取第一次会把
    一条噪声曲线报成一个干净的阈值。
    """
    block = merged.sort_values("exposure_malicious")
    asr = block["asr_mean"].to_numpy(dtype=float)
    exposure = block["exposure_malicious"].to_numpy(dtype=float)
    crossings = [i for i in range(len(asr) - 1)
                 if (asr[i] < level) != (asr[i + 1] < level)]
    if len(crossings) != 1:
        return None
    i = crossings[0]
    span = asr[i + 1] - asr[i]
    if abs(span) < 1e-12:
        return float(exposure[i])
    fraction = (level - asr[i]) / span
    return float(exposure[i] + fraction * (exposure[i + 1] - exposure[i]))


def plot_exposure(merged: pd.DataFrame, out_path,
                  threshold: Optional[float] = None) -> Path:
    """ASR vs 累积吸收的恶意幅度。一个标记 = 一个 (防御, 密度) 格子。"""
    fig, axis = plt.subplots(figsize=(7.4, 5.2))

    for defense, group in merged.groupby("defense"):
        label, color = _style(str(defense))
        exact = bool(group["exposure_is_exact"].iloc[0])
        axis.scatter(group["exposure_malicious"], group["asr_mean"],
                     s=70, color=color if exact else "none",
                     edgecolors=color, linewidths=1.8,
                     marker="o" if exact else "s",
                     label=label + ("" if exact else " (approx. E)"),
                     zorder=3)
        for _, row in group.iterrows():
            axis.annotate(f"{int(row['bad_client_num'])}",
                          (row["exposure_malicious"], row["asr_mean"]),
                          textcoords="offset points", xytext=(6, 5),
                          fontsize=7, color=color)

    if threshold is not None and np.isfinite(threshold) and threshold > 0:
        axis.axvline(threshold, color="0.4", linestyle=":", linewidth=1.4,
                     zorder=1)
        axis.annotate(f"ASR = 0.5 at E ≈ {threshold:.3g}",
                      (threshold, 0.52), rotation=90, fontsize=8,
                      color="0.3", ha="right", va="bottom")

    positive = merged["exposure_malicious"] > 0
    if positive.sum() >= 2 and (merged.loc[positive, "exposure_malicious"].max()
                                / merged.loc[positive,
                                             "exposure_malicious"].min() > 20):
        axis.set_xscale("log")              # 跨数量级时线性轴会把低端全挤在 0 上
    axis.set_xlabel("Cumulative absorbed malicious magnitude  E "
                    "(upper bound; see caption)")
    axis.set_ylabel("Backdoor ASR (personalized, tail mean)")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0)
    axis.set_title("Backdoor implantation is governed by absorbed magnitude,\n"
                   "not by the number of leaked malicious updates")

    lines = ["Point labels = number of malicious clients.  E sums "
             "w_k(t)*||g_k(t)|| over malicious client-rounds.",
             "E is an UPPER BOUND: only update norms were recorded, so "
             "directional cancellation is not subtracted."]
    # 图上没有近似格子时不要提"空心方块" —— 说明一个不存在的图例会让人去找它
    if not merged["exposure_is_exact"].all():
        lines.append("Open squares use a scalar influence times a full-update "
                     "norm, so their E is approximate.")
    # 脚注放在画布**外面**（y<0）再靠 bbox_inches="tight" 把画布撑开 ——
    # 这是本仓库 analysis_ij._finish 用的同一套做法。试过先 tight_layout 再
    # subplots_adjust 把脚注塞进画布内：xlabel 会跟着坐标区一起移动，
    # 固定位置的脚注必然压在它上面。
    fig.tight_layout()
    fig.text(0.5, -0.01, "\n".join(lines), ha="center", va="top", fontsize=7.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_cjk_in_figure(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="步骤 0：ASR vs 累积吸收的恶意幅度 E")
    parser.add_argument("--instrument-root", default="instrumentation")
    parser.add_argument("--implantation-glob",
                        default="results/raw/exp_ij_implantation_*.csv")
    parser.add_argument("--out", default="results/figs/exp_exposure.png")
    parser.add_argument("--summary-out", default="results/exp_exposure.csv")
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--asr-column", default="asr_personalized_targeted")
    parser.add_argument("--acc-column", default="acc_personalized")
    parser.add_argument("--assume-bad-num", type=int, default=None)
    args = parser.parse_args(argv)

    merged = join_with_asr(args.instrument_root, args.implantation_glob,
                           tail=args.tail, asr_column=args.asr_column,
                           acc_column=args.acc_column,
                           assume_bad_num=args.assume_bad_num)

    out_summary = Path(args.summary_out)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_summary, index=False)
    print(f"[analysis_exposure] {len(merged)} 个格子 -> {out_summary}")

    print(f"\n=== 累积吸收的恶意幅度（按 E 升序）===")
    print(f"  {'run_id':<38}{'bad':>4}{'E':>12}{'E/轮':>10}"
          f"{'ASR':>8}{'ACC':>8}{'精确':>6}")
    for _, row in merged.iterrows():
        print(f"  {row['run_id']:<38}{int(row['bad_client_num']):>4}"
              f"{row['exposure_malicious']:>12.4g}"
              f"{row['exposure_per_round']:>10.4g}"
              f"{row['asr_mean']:>8.4f}{row['acc_mean']:>8.4f}"
              f"{'是' if row['exposure_is_exact'] else '近似':>6}")

    diagnosis = collapse_diagnosis(merged)
    print("\n=== 坍缩判据 ===")
    for key in ("n_cells", "n_cells_exact", "n_defenses", "n_densities",
                "spearman_rho", "spearman_p", "spearman_rho_exact_only"):
        if key in diagnosis:
            value = diagnosis[key]
            print(f"  {key:<26} {value:.4g}" if isinstance(value, float)
                  else f"  {key:<26} {value}")
    if diagnosis.get("spearman_by_defense"):
        print(f"  各防御内部 ρ                {diagnosis['spearman_by_defense']}")
    print(f"  {diagnosis['verdict']}")

    threshold = estimate_threshold(merged)
    if threshold is not None:
        print(f"\n  ASR 跨过 0.5 时 E ≈ {threshold:.4g}")
    else:
        print("\n  ASR 在 0.5 处**不是只穿越一次**（或根本没穿越），"
              "不报阈值 —— 取第一次穿越会把噪声曲线报成干净的阈值。")

    figure = plot_exposure(merged, args.out, threshold=threshold)
    print(f"[analysis_exposure] 图 -> {figure}")

    approx = merged[~merged["exposure_is_exact"]]
    if not approx.empty:
        print(f"\n  ⚠️ {len(approx)}/{len(merged)} 个格子的 E 是近似"
              f"（{sorted(set(approx['defense']))}）：逐坐标取舍被标量化了。"
              f"图上是空心方块，**不要和实心圆混在一起做回归**。")
    print("  ⚠️ E 是上界：埋点只有 ‖g_k‖，方向抵消没有扣掉。要精确到向量"
          "得改埋点重跑。")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
