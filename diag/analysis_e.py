"""实验 E 的汇总与绘图：对抗地板的分解。

三张图（§4）：

- **E-1（主图）** 柱状图，横轴 7 种扰动模式，纵轴 ``ASR_targeted``，
  分组为 E1 attacked / E2 clean-transfer / E3 clean-oracle。
- **E-2** 横轴 alpha，三条线 ``ASR_attacked`` / ``ASR_clean_transfer`` / ``Delta``，
  展示分解如何随异构度变化。
- **E-3** ``ASR_targeted`` vs ``ErrorRate`` 散点，按扰动模式着色 ——
  直观展示"定向 vs 非定向"的分离。

⚠️ **图中一律使用英文**（出版要求）。``diag.analysis._finish`` 会在保存前扫描
整张图的文字，发现 CJK 直接抛异常，所以违规不会静默通过。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import _alpha_order, _finish, load_raw

__all__ = ["EXP_E_MODE_ORDER", "MODE_LABEL", "CELL_STYLE", "summarize_e",
           "decomposition_e", "classify_floor", "plot_e1", "plot_e2",
           "plot_e3", "main"]

# 排列顺序：基线 -> 正对照 -> 两个成分 -> 完整触发器 -> 两个负对照
EXP_E_MODE_ORDER: Tuple[str, ...] = ("none", "real_target", "delta_only",
                                     "xi_only", "full", "random_eps4",
                                     "random_eps8")

MODE_LABEL: Dict[str, str] = {
    "none": "none\n(clean)",
    "real_target": "real_target\n(pos. control)",
    "delta_only": "delta_only\n(4/255)",
    "xi_only": "xi_only\n(4/255)",
    "full": "full = delta+xi\n(8/255)",
    "random_eps4": "random_eps4\n(neg. control)",
    "random_eps8": "random_eps8\n(neg. control)",
}

CELL_STYLE: Dict[str, Tuple[str, str]] = {
    "E1": ("E1 attacked (paper baseline)", "tab:red"),
    "E2": ("E2 clean-transfer (adversarial floor)", "tab:blue"),
    "E3": ("E3 clean-oracle (white-box ceiling)", "tab:green"),
    "E4": ("E4 clean-blackbox (no test-time gradients)", "tab:purple"),
}

# §5 的判据（只对 E2 / full / 共享探针集 / 排除目标类后的 ASR_targeted 生效）
FLOOR_BANDS: Tuple[Tuple[float, float, str, str], ...] = (
    (0.00, 0.15, "true backdoor", "tab:green"),
    (0.20, 0.45, "mixture (expected)", "tab:orange"),
    (0.70, 1.00, "adversarial attack", "tab:red"),
)
CHANCE_LEVEL = 0.10   # CIFAR-10 十分类的随机水平


def classify_floor(asr_clean_transfer: float) -> str:
    """按 §5 主判据给出结论标签。

    spec 只定义了三段，**留下了两个空隙**（0.15–0.20 与 0.45–0.70）。
    落在空隙里时如实返回"未定义"，不向邻近区间靠拢。
    """
    if not np.isfinite(asr_clean_transfer):
        return "未能确定（无有效数值）"
    for low, high, label, _ in FLOOR_BANDS:
        if low <= asr_clean_transfer <= high:
            return {"true backdoor": "真后门，论文叙事成立",
                    "mixture (expected)": "混合体（预期落点）",
                    "adversarial attack": "本质是对抗攻击"}[label]
    return (f"落在 spec 未定义的区间（{asr_clean_transfer:.3f}）——"
            f"判据只覆盖 ≤0.15 / 0.20–0.45 / >0.70")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize_e(raw_csv_glob) -> pd.DataFrame:
    """按 ``(alpha, seed, cell, perturb_mode)`` 聚合跨客户端的均值/标准差。

    nan 行为：``real_target`` 的 ``asr_targeted`` 全为 nan（分母为空），
    聚合后仍是 nan —— 该组应看 ``asr_targeted_unfiltered``。
    """
    raw = load_raw(raw_csv_glob)
    required = {"alpha", "seed", "cell", "perturb_mode", "asr_targeted"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"exp_e CSV 缺少列: {sorted(missing)}")

    # ⚠️ `linf_realized` / `clipped_*` 是**忠实性读数**，不是可有可无的附加列：
    # 原实现（fba.py:55）加完 δ 之后不 clamp，本工具链 clamp 了，重建的触发器
    # 因此系统性偏弱。2026-09 审计发现它们此前被这张表整列丢掉，于是"偏离有多大"
    # 从来没人看得见。加回来。
    metrics = [c for c in ("asr_targeted", "asr_targeted_unfiltered",
                           "error_rate", "clean_acc",
                           "linf_realized", "linf_budget",
                           "clipped_fraction", "clipped_l1_ratio")
               if c in raw.columns]
    rows = []
    for keys, block in raw.groupby(["alpha", "seed", "cell", "perturb_mode"]):
        alpha, seed, cell, mode = keys
        row = {"alpha": float(alpha), "seed": int(seed), "cell": str(cell),
               "perturb_mode": str(mode), "n_clients": int(len(block))}
        for metric in metrics:
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = (float(finite.std(ddof=1)) if len(finite) > 1
                                    else np.nan)
            row[f"{metric}_n"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["alpha", "seed", "cell", "perturb_mode"],
        ascending=[False, True, True, True]).reset_index(drop=True)


def decomposition_e(summary: pd.DataFrame, mode: str = "full") -> pd.DataFrame:
    """核心分解 ``ASR_attacked = ASR_clean_transfer + Delta``，逐 (alpha, seed)。"""
    block = summary[summary["perturb_mode"] == mode]
    rows = []
    for (alpha, seed), sub in block.groupby(["alpha", "seed"]):
        picked = {cell: sub[sub["cell"] == cell]["asr_targeted_mean"]
                  for cell in ("E1", "E2", "E3", "E4")}
        values = {cell: (float(series.iloc[0]) if len(series) else np.nan)
                  for cell, series in picked.items()}
        rows.append({
            "alpha": float(alpha), "seed": int(seed), "perturb_mode": mode,
            "ASR_attacked": values["E1"],
            "ASR_clean_transfer": values["E2"],
            "ASR_clean_oracle": values["E3"],
            "ASR_clean_blackbox": values["E4"],
            "Delta": values["E1"] - values["E2"],
            "floor_fraction": (values["E2"] / values["E1"]
                               if np.isfinite(values["E1"]) and values["E1"] > 0
                               else np.nan),
            "verdict": classify_floor(values["E2"]),
        })
    return pd.DataFrame(rows).sort_values(["alpha", "seed"],
                                          ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _modes_present(summary: pd.DataFrame) -> List[str]:
    present = set(summary["perturb_mode"].unique())
    ordered = [m for m in EXP_E_MODE_ORDER if m in present]
    return ordered + sorted(present - set(ordered))


def plot_e1(summary: pd.DataFrame, out_path, alpha: Optional[float] = None,
            seed: Optional[int] = None) -> Path:
    """图 E-1（主图）：7 种扰动模式 × 三个 cell 的 ASR_targeted 柱状图。

    ``real_target`` 的 ``ASR_targeted`` 按定义是 nan（该组全部是目标类样本，
    排除后分母为空），因此**不画柱**，改用空心标记显示
    ``asr_targeted_unfiltered``，即该模型在目标类上的准确率。
    """
    block = summary.copy()
    if alpha is not None:
        block = block[block["alpha"] == alpha]
    if seed is not None:
        block = block[block["seed"] == seed]

    modes = _modes_present(block)
    cells = [c for c in ("E1", "E2", "E3", "E4")
             if c in set(block["cell"].unique())]
    positions = np.arange(len(modes))
    width = 0.8 / max(1, len(cells))

    fig, ax = plt.subplots(figsize=(11, 5))
    for index, cell in enumerate(cells):
        label, color = CELL_STYLE[cell]
        means, errors = [], []
        for mode in modes:
            row = block[(block["cell"] == cell) & (block["perturb_mode"] == mode)]
            means.append(float(row["asr_targeted_mean"].mean()) if len(row) else np.nan)
            errors.append(float(row["asr_targeted_std"].mean()) if len(row) else 0.0)
        offset = (index - (len(cells) - 1) / 2) * width
        ax.bar(positions + offset, means, width * 0.92, yerr=errors, capsize=3,
               label=label, color=color, alpha=0.85)

        # real_target 的补充标记：ASR_targeted 无定义，改画不排除口径
        if "real_target" in modes:
            row = block[(block["cell"] == cell)
                        & (block["perturb_mode"] == "real_target")]
            if len(row):
                value = float(row["asr_targeted_unfiltered_mean"].mean())
                ax.scatter([positions[modes.index("real_target")] + offset], [value],
                           marker="o", s=70, facecolors="none", edgecolors=color,
                           linewidths=1.8, zorder=5)

    ax.axhline(CHANCE_LEVEL, color="black", linestyle="--", linewidth=1.0,
               label=f"Chance level ({CHANCE_LEVEL:.0%})")
    ax.set_xticks(positions)
    ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], fontsize=8)
    ax.set_ylabel("ASR_targeted  (target-class samples excluded)")
    title = "E-1: attack success by perturbation mode"
    if alpha is not None:
        title += f"  (alpha={alpha})"
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    # 注释放到坐标区之外：压在柱子上会挡数据
    fig.supxlabel("hollow markers: real_target is shown with the unfiltered rate "
                  "(its filtered denominator is empty by construction)",
                  fontsize=7, style="italic", color="dimgray")
    return _finish(fig, out_path)


def plot_e2(summary: pd.DataFrame, out_path, mode: str = "full") -> Path:
    """图 E-2：分解随异构度的变化。三条线 = attacked / clean-transfer / Delta。"""
    table = decomposition_e(summary, mode=mode)
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas = _alpha_order(table)
    positions = np.arange(len(alphas))

    series = [("ASR_attacked", "ASR_attacked (E1)", "tab:red", "o", "-"),
              ("ASR_clean_transfer", "ASR_clean_transfer (E2) = floor",
               "tab:blue", "s", "-"),
              ("Delta", "Delta = implanted by training", "tab:gray", "^", "--")]
    for column, label, color, marker, style in series:
        means, errors = [], []
        for alpha in alphas:
            values = table[table["alpha"] == alpha][column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.errorbar(positions, means, yerr=errors, label=label, color=color,
                    marker=marker, linestyle=style, capsize=3)

    # 先定好横轴范围，分带标签才不会被画到坐标区之外
    ax.set_xlim(-0.5, len(alphas) - 0.5)

    # §5 的判据分带（只对 E2 的 full 模式有意义）
    for low, high, label, color in FLOOR_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.07)
        ax.text(len(alphas) - 0.58, (low + high) / 2, label, fontsize=7,
                va="center", ha="right", color=color, style="italic")

    ax.axhline(CHANCE_LEVEL, color="black", linestyle=":", linewidth=1.0,
               label=f"Chance level ({CHANCE_LEVEL:.0%})")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha (heterogeneity increases to the right)")
    ax.set_ylabel(f"ASR_targeted  (mode = {mode})")
    ax.set_title("E-2: adversarial floor vs implanted component across heterogeneity")
    # Delta 可能为负（地板高于 attacked）——那是有意义的结果，不能被裁掉
    values = table[["ASR_attacked", "ASR_clean_transfer", "Delta"]].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lower = min(0.0, float(finite.min()) - 0.05) if finite.size else 0.0
    ax.set_ylim(lower, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_e3(raw_csv_glob, out_path, cells: Optional[Sequence[str]] = None) -> Path:
    """图 E-3：ASR_targeted vs ErrorRate 散点，按扰动模式着色。

    用途：直观区分"定向"与"非定向"。``xi_only`` 预期落在右下角
    （高 ErrorRate、低 ASR），``full`` 预期偏向右上。

    两套编码分开：**颜色 = 扰动模式，标记形状 = cell**，图例也拆成两个 ——
    否则 7 模式 × 4 cell 会产生 28 条图例，把图淹掉。
    每个 (模式, cell) 的质心用带黑边的大标记叠在淡色散点之上，
    因为在 100 个客户端的密度下，看得清的是聚集位置而不是单个点。

    ``real_target`` 的 ``asr_targeted`` 恒为 nan（分母为空），自动不参与本图。
    """
    raw = load_raw(raw_csv_glob)
    if cells is not None:
        raw = raw[raw["cell"].isin(list(cells))]

    # 两个轴都无有效值的模式（如 real_target）直接跳过，避免出现空图例
    usable = raw[np.isfinite(raw["asr_targeted"].to_numpy(dtype=float))
                 & np.isfinite(raw["error_rate"].to_numpy(dtype=float))]
    modes = [m for m in EXP_E_MODE_ORDER
             if m in set(usable["perturb_mode"].unique())]
    present_cells = [c for c in ("E1", "E2", "E3", "E4")
                     if c in set(usable["cell"].unique())]
    palette = plt.get_cmap("tab10")
    markers = {"E1": "o", "E2": "s", "E3": "^", "E4": "D"}
    mode_color = {mode: palette(i % 10) for i, mode in enumerate(modes)}

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for mode in modes:
        for cell in present_cells:
            sub = usable[(usable["perturb_mode"] == mode)
                         & (usable["cell"] == cell)]
            if sub.empty:
                continue
            ax.scatter(sub["error_rate"], sub["asr_targeted"], s=10, alpha=0.22,
                       color=mode_color[mode], marker=markers[cell],
                       linewidths=0)
            ax.scatter([sub["error_rate"].mean()], [sub["asr_targeted"].mean()],
                       s=130, color=mode_color[mode], marker=markers[cell],
                       edgecolors="black", linewidths=1.1, zorder=5)

    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("ErrorRate  P(pred != y_true | y_true != y_t)")
    ax.set_ylabel("ASR_targeted  P(pred == y_t | y_true != y_t)")
    ax.set_title("E-3: targeted vs untargeted effect, by perturbation mode")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # 图例拆成两个：颜色 -> 模式，标记 -> cell
    from matplotlib.lines import Line2D

    color_handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                            color=mode_color[mode], label=mode) for mode in modes]
    color_handles.append(Line2D([], [], linestyle=":", color="black",
                                label="ASR = ErrorRate"))
    shape_handles = [Line2D([], [], marker=markers[cell], linestyle="",
                            markersize=8, color="dimgray",
                            markeredgecolor="black", label=CELL_STYLE[cell][0])
                     for cell in present_cells]

    first = ax.legend(handles=color_handles, title="Perturbation mode (color)",
                      fontsize=7, title_fontsize=8, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=shape_handles, title="Cell (marker)", fontsize=7,
              title_fontsize=8, loc="lower right")

    fig.supxlabel("large outlined markers = per-(mode, cell) centroid;  "
                  "bottom-right = disrupts the label but not toward y_t;  "
                  "top-right = disrupts and redirects to y_t",
                  fontsize=7, style="italic", color="dimgray")
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 E 的汇总与绘图")
    parser.add_argument("--raw-glob", default="results/raw/exp_e_*.csv")
    parser.add_argument("--out-dir", default="results/figs")
    parser.add_argument("--summary-out", default="results/exp_e_summary.csv")
    parser.add_argument("--decomposition-out",
                        default="results/exp_e_decomposition.csv")
    parser.add_argument("--e1-alpha", type=float, default=None,
                        help="E-1 只画这个 alpha；不给则对所有 alpha 取平均")
    args = parser.parse_args(argv)

    summary = summarize_e(args.raw_glob)
    decomposition = decomposition_e(summary)

    for path, frame in ((args.summary_out, summary),
                        (args.decomposition_out, decomposition)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[analysis_e] {len(frame)} 行 -> {path}")

    out_dir = Path(args.out_dir)
    produced = [plot_e1(summary, out_dir / "exp_e_E1.png", alpha=args.e1_alpha),
                plot_e2(summary, out_dir / "exp_e_E2.png"),
                plot_e3(args.raw_glob, out_dir / "exp_e_E3.png")]
    for path in produced:
        print(f"[analysis_e] 图 -> {path}")

    print("\n=== 主判据（E2 / full / 共享探针集 / 排除目标类）===")
    for _, row in decomposition.iterrows():
        print(f"  alpha={row['alpha']:<6} seed={int(row['seed'])}  "
              f"ASR_attacked={row['ASR_attacked']:.4f}  "
              f"floor={row['ASR_clean_transfer']:.4f}  "
              f"Delta={row['Delta']:.4f}  ->  {row['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
