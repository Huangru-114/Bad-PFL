"""实验 F 的汇总与绘图：冻结生成器的时间衰减。

三张图（§3.2）：

- **F-1** 热力图，横轴 t、纵轴 s，色值 ``asr_targeted``。对角线是"新鲜生成器"，
  向右延伸是衰减方向。**下三角（t < s）留空**，不填 0 也不插值（陷阱 7/8）。
- **F-2（判据图）** 折线，横轴 ``gap = t − s``，每条线对应一个 s。
- **F-3** 对角线曲线，横轴 s，纵轴 ``asr_targeted(s, s)`` ——
  "新鲜生成器效力随攻击成熟度的变化"，作对照基准。

每张图都出**绝对值**与**相对基线**两个版本。§3.3 的判据以相对版本为准：
模型准确率随轮次变化，基线 ASR 也随之漂移，绝对值里混了这一项。

⚠️ **图中一律使用英文**（出版要求）。``diag.analysis._finish`` 会在保存前扫描
整张图的文字，发现 CJK 直接抛异常，所以违规不会静默通过。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import _finish, load_raw

__all__ = ["METRIC_ABS", "METRIC_REL", "summarize_f", "matrix_pivot",
           "decay_table", "classify_decay", "plot_f1", "plot_f2", "plot_f3",
           "main"]

METRIC_ABS = "asr_targeted"
METRIC_REL = "asr_targeted_rel"

# §3.3 的判据阈值。写死在这里，报告同时打印规则与实测值，读者可自行复核。
DECAY_GAP = 200          # "gap=200 时降至对角线值的 50% 以下" 里的 gap
DECAY_RATIO = 0.50
STABLE_GAP = 500         # "gap=500 时仍保持对角线值的 80% 以上" 里的 gap
STABLE_RATIO = 0.80


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize_f(raw_csv_glob, eval_target: str = "personalized",
                perturb_mode: str = "delta_only") -> pd.DataFrame:
    """按 ``(alpha, seed, gen_round_s, model_round_t)`` 聚合跨客户端的均值/标准差。

    ``eval_target='personalized'`` 时每格有多个客户端，聚合给出均值与
    跨客户端 std；``global_bn_borrowed`` 每格只有一行，std 为 nan。

    ``staleness`` 一并聚合（取最大值）—— 它是快照机制引入的协变量，
    不能因为不方便就从表里消失。
    """
    raw = load_raw(raw_csv_glob)
    required = {"gen_round_s", "model_round_t", "perturb_mode", "eval_target",
                METRIC_ABS}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"exp_f CSV 缺少列: {sorted(missing)}")

    block = raw[(raw["eval_target"] == eval_target)
                & (raw["perturb_mode"] == perturb_mode)].copy()
    block = block[np.isfinite(block["gen_round_s"].to_numpy(dtype=float))]
    if block.empty:
        return pd.DataFrame()

    metrics = [c for c in (METRIC_ABS, METRIC_REL, "asr_unfiltered",
                           "error_rate", "clean_acc", "baseline_asr")
               if c in block.columns]
    rows = []
    for keys, group in block.groupby(["alpha", "seed", "gen_round_s",
                                      "model_round_t"]):
        alpha, seed, s, t = keys
        row = {"alpha": float(alpha), "seed": int(seed),
               "gen_round_s": int(s), "model_round_t": int(t),
               "gap": int(t) - int(s), "n_clients": int(len(group)),
               "eval_target": eval_target, "perturb_mode": perturb_mode}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (float(finite.std(ddof=1)) if len(finite) > 1
                                    else np.nan)
        if "staleness" in group.columns:
            row["staleness_max"] = int(group["staleness"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["alpha", "seed", "gen_round_s", "model_round_t"]).reset_index(drop=True)


def matrix_pivot(summary: pd.DataFrame, metric: str = METRIC_ABS) -> pd.DataFrame:
    """把汇总表转成 s × t 的矩阵。下三角是 ``NaN``（从未计算过）。"""
    column = f"{metric}_mean"
    if column not in summary.columns:
        raise KeyError(f"汇总表里没有 '{column}'")
    return summary.pivot_table(index="gen_round_s", columns="model_round_t",
                               values=column, aggfunc="mean").sort_index()


def decay_table(summary: pd.DataFrame, metric: str = METRIC_REL) -> pd.DataFrame:
    """每个 s 一行：对角线值、若干 gap 处的值与保留比例。

    ``retention(gap) = value(s, s+gap) / value(s, s)``。对角线值 ≈ 0 时比例
    没有意义，返回 nan 而不是一个巨大的数 —— 那会在图上制造假信号。
    """
    column = f"{metric}_mean"
    rows = []
    for s, group in summary.groupby("gen_round_s"):
        diagonal = group[group["gap"] == 0][column]
        base = float(diagonal.iloc[0]) if len(diagonal) else np.nan
        row: Dict[str, Any] = {"gen_round_s": int(s), "diagonal": base,
                               "max_gap": int(group["gap"].max())}
        for gap in (DECAY_GAP, STABLE_GAP):
            at_gap = group[group["gap"] == gap][column]
            value = float(at_gap.iloc[0]) if len(at_gap) else np.nan
            row[f"value_gap{gap}"] = value
            row[f"retention_gap{gap}"] = (
                value / base if np.isfinite(base) and abs(base) > 1e-6 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gen_round_s").reset_index(drop=True)


def classify_decay(table: pd.DataFrame, mature_s: int = 500) -> str:
    """§3.3 的判据。规则写死，不满足时返回"未能确定"，不向邻近结论靠拢。

    只看成熟的生成器（``s >= mature_s``），避免早期未收敛的干扰。
    """
    mature = table[table["gen_round_s"] >= int(mature_s)]
    if mature.empty:
        return (f"未能确定：没有 s >= {mature_s} 的成熟生成器"
                f"（可用 s: {sorted(table['gen_round_s'].tolist())}）")

    decay = mature[f"retention_gap{DECAY_GAP}"].to_numpy(dtype=float)
    stable = mature[f"retention_gap{STABLE_GAP}"].to_numpy(dtype=float)
    decay = decay[np.isfinite(decay)]
    stable = stable[np.isfinite(stable)]

    early = table[table["gen_round_s"] < int(mature_s)]
    early_decay = early[f"retention_gap{DECAY_GAP}"].to_numpy(dtype=float)
    early_decay = early_decay[np.isfinite(early_decay)]

    verdicts = []
    if len(decay) and float(decay.mean()) < DECAY_RATIO:
        verdicts.append(
            f"H_coadapt 成立：s>={mature_s} 时 gap={DECAY_GAP} 的平均保留率 "
            f"{decay.mean():.3f} < {DECAY_RATIO}，论文的持久性解释被推翻")
    if len(stable) and float(stable.mean()) > STABLE_RATIO:
        verdicts.append(
            f"H_stable 成立：s>={mature_s} 时 gap={STABLE_GAP} 的平均保留率 "
            f"{stable.mean():.3f} > {STABLE_RATIO}，δ 稳定存在于共享参数，"
            f"防御落点需相应调整")
    # 第三种情况：早期衰减快、晚期平坦 —— 对论文有利，但同样是有价值的定量结果，
    # 不因为"不符合预期"就淡化（§3.3 的明确要求）。
    if (len(early_decay) and len(decay)
            and float(early_decay.mean()) < DECAY_RATIO
            and float(decay.mean()) >= DECAY_RATIO):
        verdicts.append(
            f"早期 s 衰减快（gap={DECAY_GAP} 保留率 {early_decay.mean():.3f}）、"
            f"晚期 s 平坦（{decay.mean():.3f}）：互强化确实在让 δ 收敛，"
            f"**这支持论文**，且给出了论文从未报告过的收敛时间尺度")

    if verdicts:
        return "；同时 ".join(verdicts)

    # 区分两种"未能确定"：**没有数据** 与 **有数据但落在判据的空隙里**。
    # 混为一谈会让读者误以为已经测过了。
    #
    # 前一种在 P2 的稀疏网格上是常态：网格步长 200 时 gap 只能取 200 的倍数，
    # gap=500 根本不存在。判据要求的是**实测值**，此处绝不插值（陷阱 8）。
    if not len(decay) and not len(stable):
        available = sorted({int(g) for g in table["max_gap"]})
        return (f"未能确定：矩阵里没有 gap={DECAY_GAP} 或 gap={STABLE_GAP} 的格子"
                f"（各 s 的最大 gap 为 {available}）。判据只认这两个 gap 上的"
                f"实测值，**不插值**；要给出结论需要一个能取到这些 gap 的网格"
                f"（例如步长 50 的完整网格）。")
    parts = []
    parts.append(f"gap={DECAY_GAP} 保留率 "
                 + (f"{decay.mean():.3f}（未低于 {DECAY_RATIO}）"
                    if len(decay) else "无数据"))
    parts.append(f"gap={STABLE_GAP} 保留率 "
                 + (f"{stable.mean():.3f}（未超过 {STABLE_RATIO}）"
                    if len(stable) else "无数据"))
    return (f"未能确定：s>={mature_s} 时 " + "，".join(parts)
            + f"；§3.3 的判据只覆盖这两端，中间区间无定义，不向邻近结论靠拢。")


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
_METRIC_LABEL = {
    METRIC_ABS: "ASR_targeted",
    METRIC_REL: "ASR_targeted - baseline(theta_t)",
}


def plot_f1(summary: pd.DataFrame, out_path, metric: str = METRIC_ABS) -> Path:
    """图 F-1：s × t 热力图。下三角留空（陷阱 7），缺失的格子同样留空（陷阱 8）。"""
    matrix = matrix_pivot(summary, metric)
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # 必须 .copy()：直接对注册表里的 colormap 调 set_bad 会全局改掉它，
    # 污染同一进程里后续的所有图。
    diverging = metric == METRIC_REL
    cmap = plt.get_cmap("coolwarm" if diverging else "viridis").copy()
    # 留空的格子画成**斜纹**而不是某个灰度：coolwarm 的零值本身就是浅灰，
    # 用灰色填充会让"没算"和"效应为 0"看起来一样 —— 而这正是陷阱 7 要防的。
    cmap.set_bad(alpha=0.0)
    limit = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))) or 1.0
    vmin, vmax = (-limit, limit) if diverging else (0.0, 1.0)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                      origin="lower")

    from matplotlib.patches import Rectangle

    blank = matrix.to_numpy(dtype=float)
    for row in range(blank.shape[0]):
        for column in range(blank.shape[1]):
            if np.isfinite(blank[row, column]):
                continue
            ax.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1,
                                   facecolor="white", edgecolor="0.75",
                                   hatch="///", linewidth=0.4, zorder=1))

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(i) for i in matrix.index], fontsize=8)
    ax.set_xlabel("Model round t (evaluated snapshot)")
    ax.set_ylabel("Generator round s (frozen snapshot)")
    ax.set_title(f"F-1: frozen-generator decay matrix ({_METRIC_LABEL[metric]})")
    fig.colorbar(image, ax=ax, label=_METRIC_LABEL[metric])

    # 逐格标注数值：矩阵最多 20x20，标注仍然可读，且免去读者对色标反推数值。
    # 文字颜色按底色明暗选：viridis 低值是深紫、高值是亮黄；coolwarm 两端深、
    # 中间白。选反了在极值格上会读不出数字。
    array = matrix.to_numpy(dtype=float)
    if array.size <= 400:
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                if not np.isfinite(value):
                    continue
                dark = (abs(value) > 0.6 * limit) if diverging else (value < 0.5)
                ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if dark else "black")

    # 脚注分两行：单行放不下会被画到画布之外（tight_layout 不裁 supxlabel）
    fig.supxlabel("hatched = not computed. t < s is meaningless: a future "
                  "generator against a past model.\nMissing snapshots stay "
                  "hatched as well — never interpolated, never filled with 0.",
                  fontsize=7, style="italic", color="dimgray")
    return _finish(fig, out_path)


def plot_f2(summary: pd.DataFrame, out_path, metric: str = METRIC_REL) -> Path:
    """图 F-2（判据图）：横轴 gap，每条线一个 s。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    column = f"{metric}_mean"
    std_column = f"{metric}_std"
    rounds = sorted(summary["gen_round_s"].unique())
    palette = plt.get_cmap("viridis")

    for index, s in enumerate(rounds):
        group = summary[summary["gen_round_s"] == s].sort_values("gap")
        color = palette(index / max(1, len(rounds) - 1))
        errors = (group[std_column].to_numpy(dtype=float)
                  if std_column in group.columns else None)
        if errors is not None and not np.isfinite(errors).any():
            errors = None
        ax.errorbar(group["gap"], group[column], yerr=errors, marker="o",
                    markersize=4, color=color, capsize=2, label=f"s = {int(s)}")

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("gap = t - s  (rounds elapsed since the generator was frozen)")
    ax.set_ylabel(_METRIC_LABEL[metric])
    ax.set_title("F-2: decay of a frozen generator as the model drifts")
    ax.legend(fontsize=7, title="Generator snapshot", title_fontsize=8,
              ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.supxlabel("H_coadapt predicts a falling curve; H_stable predicts a flat "
                  "one. Curves starting at different heights reflect generator "
                  "maturity, which figure F-3 isolates.",
                  fontsize=7, style="italic", color="dimgray")
    return _finish(fig, out_path)


def plot_f3(summary: pd.DataFrame, out_path, metric: str = METRIC_ABS) -> Path:
    """图 F-3：对角线曲线 ``asr(s, s)`` —— 新鲜生成器的效力随攻击成熟度变化。"""
    diagonal = summary[summary["gap"] == 0].sort_values("gen_round_s")
    column, std_column = f"{metric}_mean", f"{metric}_std"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    errors = (diagonal[std_column].to_numpy(dtype=float)
              if std_column in diagonal.columns else None)
    if errors is not None and not np.isfinite(errors).any():
        errors = None
    ax.errorbar(diagonal["gen_round_s"], diagonal[column], yerr=errors,
                marker="o", color="tab:red", capsize=3,
                label="fresh generator: ASR(s, s)")

    if "baseline_asr_mean" in summary.columns:
        ax.plot(diagonal["gen_round_s"], diagonal["baseline_asr_mean"],
                marker="s", markersize=4, linestyle="--", color="tab:gray",
                label="baseline: no perturbation on theta_t")

    ax.set_xlabel("Round s (attack maturity)")
    ax.set_ylabel(_METRIC_LABEL[metric])
    ax.set_title("F-3: effectiveness of a freshly aligned generator over training")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.supxlabel("This is the control baseline for figure F-2: it separates "
                  "'the generator was still immature' from 'the generator went "
                  "stale'.", fontsize=7, style="italic", color="dimgray")
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 F 的汇总与绘图")
    parser.add_argument("--raw-glob", default="results/raw/exp_f_matrix_*.csv")
    parser.add_argument("--out-dir", default="results/figs")
    parser.add_argument("--summary-out", default="results/exp_f_summary.csv")
    parser.add_argument("--decay-out", default="results/exp_f_decay.csv")
    parser.add_argument("--eval-target", default="personalized",
                        choices=["personalized", "global_bn_borrowed"])
    parser.add_argument("--mode", default="delta_only")
    parser.add_argument("--mature-s", type=int, default=500)
    args = parser.parse_args(argv)

    summary = summarize_f(args.raw_glob, eval_target=args.eval_target,
                          perturb_mode=args.mode)
    if summary.empty:
        print(f"[analysis_f] 没有匹配的行"
              f"（eval_target={args.eval_target}, mode={args.mode}）")
        return 1

    table = decay_table(summary)
    suffix = f"_{args.eval_target}_{args.mode}"
    for path, frame in ((args.summary_out, summary), (args.decay_out, table)):
        path = Path(path).with_stem(Path(path).stem + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[analysis_f] {len(frame)} 行 -> {path}")

    out_dir = Path(args.out_dir)
    produced = [
        plot_f1(summary, out_dir / f"exp_f_F1{suffix}.png", METRIC_ABS),
        plot_f1(summary, out_dir / f"exp_f_F1{suffix}_rel.png", METRIC_REL),
        plot_f2(summary, out_dir / f"exp_f_F2{suffix}.png", METRIC_REL),
        plot_f2(summary, out_dir / f"exp_f_F2{suffix}_abs.png", METRIC_ABS),
        plot_f3(summary, out_dir / f"exp_f_F3{suffix}.png", METRIC_ABS),
    ]
    for path in produced:
        print(f"[analysis_f] 图 -> {path}")

    print(f"\n=== §3.3 判据（eval_target={args.eval_target}, "
          f"mode={args.mode}, 相对基线口径）===")
    print(f"规则: gap={DECAY_GAP} 保留率 < {DECAY_RATIO} -> H_coadapt；"
          f"gap={STABLE_GAP} 保留率 > {STABLE_RATIO} -> H_stable")
    print(f"结论: {classify_decay(table, mature_s=args.mature_s)}")
    if "staleness_max" in summary.columns and int(summary["staleness_max"].max()) > 0:
        print(f"提示: 快照最大 staleness = {int(summary['staleness_max'].max())} 轮"
              f"（FedBN 下客户端只在参与的轮次更新本地模型）。"
              f"它与 gap 的效应无法完全解耦，见 summary CSV 的 staleness_max 列。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
