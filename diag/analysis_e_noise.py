"""E1 上"随机噪声响应 × 数据异构度"的分组折线图。

# 要验证什么

`random_eps4` / `random_eps8` 在实验 E 里本来只是**负对照** —— 同预算、无方向的
扰动理应无效。但 α=0.5 下 100 个客户端合并后的数字是：

    none         0.0270 ± 0.0090
    random_eps4  0.0335 ± 0.0115
    random_eps8  0.0434 ± 0.0165

随机噪声确实把 ASR 抬高了。问题是：**这个抬升在恶意客户端和良性客户端之间是否
不同，且是否随异构度 α 变化？**

若恶意客户端的抬升明显更大、并随 α 减小而扩大，说明被投毒的模型在目标类方向上
留下了一个**无方向扰动也能触发的偏置** —— 那是后门在决策几何上的痕迹，
与 δ 这把"钥匙"是两回事，对防御的含义也不同。

# 为什么需要单独一个脚本

`results/exp_e_summary.csv` 是按 ``(alpha, seed, cell, perturb_mode)`` 聚合的，
**没有 ``is_malicious`` 这一维**，恰好是这里要拆的那一维。只有逐客户端的
``results/raw/exp_e_E1_*.csv`` 才有。

# 两个 panel 分别回答什么

- **上（绝对值）**：各组在各扰动下的 ASR 本身，附各组的 ``none`` 基线虚线。
- **下（扣基线的 lift）**：``ASR(random) − ASR(none)``。这一步是**同一客户端逐个
  相减后再聚合**，不是两个组均值相减 —— 这样误差棒反映的是配对差异的离散度，
  而不是两个独立量的方差之和。"抬升了多少"要看这个 panel。

⚠️ **图中一律使用英文**（出版要求）。``diag.analysis._finish`` 会在保存前扫描
整张图的文字，发现 CJK 直接抛异常。

用法::

    python -m diag.analysis_e_noise \\
        --raw-glob "results/raw/exp_e_E1_*.csv" \\
        --out-dir results/figs \\
        --summary-out results/exp_e_noise_response.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (_alpha_order, assert_no_cjk_in_figure, load_raw,
                       mannwhitney)

__all__ = ["NOISE_MODES", "BASELINE_MODE", "GROUP_STYLE", "summarize_noise",
           "plot_noise_response", "main"]

NOISE_MODES: Tuple[str, ...] = ("random_eps4", "random_eps8")
BASELINE_MODE = "none"

# 颜色 = 客户端分组，线型/标记 = 扰动预算。两个编码分开，四条线才读得出来。
GROUP_STYLE: Dict[bool, Tuple[str, str]] = {
    False: ("benign", "tab:blue"),
    True: ("malicious", "tab:red"),
}
MODE_STYLE: Dict[str, Tuple[str, str]] = {
    "random_eps4": ("o", "-"),
    "random_eps8": ("^", "--"),
}
MODE_LABEL: Dict[str, str] = {
    "random_eps4": "random noise, eps=4/255",
    "random_eps8": "random noise, eps=8/255",
}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize_noise(raw_csv_glob, metric: str = "asr_targeted",
                    cell: str = "E1") -> pd.DataFrame:
    """按 ``(alpha, is_malicious, perturb_mode)`` 聚合，客户端是抽样单位。

    lift 的算法很关键：先在**每个客户端内部**做
    ``metric(random) − metric(none)``，再对客户端聚合。直接拿两个组均值相减会
    得到同样的均值、但错误的误差棒（配对信息被丢掉了）。

    多 seed 时把客户端合并入池，并单独记录 ``n_seeds`` —— 不做 seed 级的二次
    聚合，因为不同 seed 下的"同一个 client_id"并不是同一个客户端。
    """
    raw = load_raw(raw_csv_glob)

    required = {"alpha", "cell", "perturb_mode", "is_malicious", "client_id",
                "seed", metric}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"CSV 缺少列: {sorted(missing)}；实际列: {sorted(raw.columns)}")

    # 不静默过滤：混进别的 cell 会画出一张"看起来很正常"的错图
    cells = sorted(str(c) for c in raw["cell"].unique())
    if cells != [cell]:
        raise ValueError(
            f"期望只有 cell={cell} 的行，但实际含 {cells}。"
            f"请把 --raw-glob 收窄到 exp_e_{cell}_*.csv —— 拿别的 cell 的行"
            f"当 {cell} 画会得到一张看不出错的错图。")

    present = set(raw["perturb_mode"].unique())
    absent = [m for m in (BASELINE_MODE,) + NOISE_MODES if m not in present]
    if absent:
        raise ValueError(
            f"CSV 里缺少模式 {absent}（实际有 {sorted(present)}）。"
            f"'{BASELINE_MODE}' 是扣基线所必需的，random_* 是本图的主体。")

    raw["is_malicious"] = raw["is_malicious"].astype(bool)
    key = ["alpha", "seed", "client_id", "perturb_mode"]
    duplicated = raw.duplicated(subset=key).sum()
    if duplicated:
        # glob 匹配到重叠的文件时会静默把样本量翻倍，均值不变但 std 变小，
        # 看起来完全正常。这里直接拒绝。
        example = raw[raw.duplicated(subset=key, keep=False)].iloc[0]
        raise ValueError(
            f"有 {duplicated} 行在 {key} 上重复（例如 alpha={example['alpha']}, "
            f"seed={example['seed']}, client_id={example['client_id']}, "
            f"mode={example['perturb_mode']}）。多半是 --raw-glob 匹配到了"
            f"重叠的文件；重复会让样本量虚增、误差棒变窄而均值不变，"
            f"是一种看不出来的错误。")

    keyed = raw.set_index(key)[metric]
    baseline = keyed.xs(BASELINE_MODE, level="perturb_mode")

    rows: List[Dict[str, object]] = []
    for (alpha, malicious, mode), group in raw[
            raw["perturb_mode"].isin(NOISE_MODES)].groupby(
            ["alpha", "is_malicious", "perturb_mode"]):
        values = group[metric].to_numpy(dtype=float)
        # 逐客户端配对相减
        index = pd.MultiIndex.from_arrays(
            [group["alpha"], group["seed"], group["client_id"]])
        base = baseline.reindex(index).to_numpy(dtype=float)
        lift = values - base

        finite, lift_finite = values[np.isfinite(values)], lift[np.isfinite(lift)]
        rows.append({
            "alpha": float(alpha),
            "is_malicious": bool(malicious),
            "group": GROUP_STYLE[bool(malicious)][0],
            "perturb_mode": str(mode),
            "n_clients": int(len(finite)),
            "n_seeds": int(group["seed"].nunique()),
            "n_missing_baseline": int((~np.isfinite(base)).sum()),
            "asr_mean": float(finite.mean()) if len(finite) else np.nan,
            "asr_std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            "baseline_mean": (float(base[np.isfinite(base)].mean())
                              if np.isfinite(base).any() else np.nan),
            "lift_mean": (float(lift_finite.mean()) if len(lift_finite)
                          else np.nan),
            "lift_std": (float(lift_finite.std(ddof=1)) if len(lift_finite) > 1
                         else np.nan),
        })

    summary = pd.DataFrame(rows)

    # 良性 vs 恶意的组间检验。恶意只有 10 个客户端，功效很低 —— 列名里带上 n，
    # 不允许只报 p 不报 n。
    for column in ("mw_p_asr_benign_vs_malicious", "mw_p_lift_benign_vs_malicious"):
        summary[column] = np.nan
    for (alpha, mode), block in raw[
            raw["perturb_mode"].isin(NOISE_MODES)].groupby(
            ["alpha", "perturb_mode"]):
        index = pd.MultiIndex.from_arrays(
            [block["alpha"], block["seed"], block["client_id"]])
        block = block.assign(
            _lift=block[metric].to_numpy(dtype=float)
            - baseline.reindex(index).to_numpy(dtype=float))
        benign = block[~block["is_malicious"]]
        malicious = block[block["is_malicious"]]
        _, p_asr = mannwhitney(benign[metric], malicious[metric])
        _, p_lift = mannwhitney(benign["_lift"], malicious["_lift"])
        mask = (summary["alpha"] == float(alpha)) & (summary["perturb_mode"] == mode)
        summary.loc[mask, "mw_p_asr_benign_vs_malicious"] = p_asr
        summary.loc[mask, "mw_p_lift_benign_vs_malicious"] = p_lift

    summary["metric"] = metric
    summary["cell"] = cell
    return summary.sort_values(
        ["alpha", "perturb_mode", "is_malicious"],
        ascending=[False, True, True]).reset_index(drop=True)


def baseline_by_group(raw_csv_glob, metric: str = "asr_targeted") -> pd.DataFrame:
    """各 (alpha, is_malicious) 的 ``none`` 基线，供图上画参考虚线。"""
    raw = load_raw(raw_csv_glob)
    raw["is_malicious"] = raw["is_malicious"].astype(bool)
    block = raw[raw["perturb_mode"] == BASELINE_MODE]
    rows = []
    for (alpha, malicious), group in block.groupby(["alpha", "is_malicious"]):
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append({"alpha": float(alpha), "is_malicious": bool(malicious),
                     "baseline_mean": float(values.mean()) if len(values) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def plot_noise_response(summary: pd.DataFrame, baselines: pd.DataFrame,
                        out_path, metric: str = "asr_targeted") -> Path:
    """上下两个 panel 共享横轴：绝对 ASR / 扣基线的 lift。"""
    alphas = _alpha_order(summary)
    positions = np.arange(len(alphas))
    single_point = len(alphas) < 2

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})

    handles: List[object] = []
    for malicious in (False, True):
        group_label, color = GROUP_STYLE[malicious]
        for mode in NOISE_MODES:
            marker, linestyle = MODE_STYLE[mode]
            block = summary[(summary["is_malicious"] == malicious)
                            & (summary["perturb_mode"] == mode)]
            if block.empty:
                continue
            ordered = block.set_index("alpha").reindex(alphas)
            n_clients = int(np.nanmax(ordered["n_clients"].to_numpy(dtype=float)))
            label = f"{group_label} (n={n_clients}), {MODE_LABEL[mode]}"
            # 单点时不画线：一条连接不存在的中间值的线会被读成趋势
            style = "none" if single_point else linestyle
            container = top.errorbar(
                positions, ordered["asr_mean"], yerr=ordered["asr_std"],
                color=color, marker=marker, linestyle=style, capsize=3,
                markersize=6, label=label)
            handles.append(container)
            bottom.errorbar(positions, ordered["lift_mean"],
                            yerr=ordered["lift_std"], color=color, marker=marker,
                            linestyle=style, capsize=3, markersize=6, label=label)

    # 各组的 none 基线：绝对值 panel 才有意义（lift panel 的基线恒为 0）
    for malicious in (False, True):
        group_label, color = GROUP_STYLE[malicious]
        block = baselines[baselines["is_malicious"] == malicious]
        if block.empty:
            continue
        ordered = block.set_index("alpha").reindex(alphas)
        line, = top.plot(positions, ordered["baseline_mean"], color=color,
                         linestyle=":", linewidth=1.4, alpha=0.9,
                         label=f"{group_label}, no perturbation (baseline)")
        handles.append(line)

    bottom.axhline(0.0, color="black", linewidth=0.8, linestyle=":")

    top.set_ylabel(f"{metric}")
    top.set_title("Response to same-budget random noise on attacked models (E1)")
    top.grid(alpha=0.3)
    # 图例放在坐标区**之外**：四条线都向右上走，任何 loc="best" 都会压住数据
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
               fontsize=8, frameon=True)

    bottom.set_ylabel(f"lift = {metric}(random) - {metric}(none)")
    bottom.set_xlabel("Dirichlet alpha (heterogeneity increases to the right)")
    bottom.set_xticks(positions)
    bottom.set_xticklabels([str(a) for a in alphas])
    bottom.set_xlim(-0.4, len(alphas) - 0.6)
    bottom.grid(alpha=0.3)

    note = ("lift is computed per client, then aggregated - not as a difference "
            "of group means, so the error bars show paired spread.\n"
            "Each random_* value is already the mean of "
            "exp_e.random_repeats sampling draws, so single-draw variance is "
            "not what the error bars represent.")
    if single_point:
        note = ("Only one alpha available: markers only, no connecting line - "
                "no trend can be read from a single point.\n") + note
    # 用 fig.text 放在坐标区下方而不是 supxlabel：supxlabel 的 y 是固定的，
    # 注释多一行就会压住 bottom 的 xlabel。负的 y 配合 bbox_inches="tight"
    # 会让画布向下扩展把它包进来。``assert_no_cjk_in_figure`` 同样扫描
    # ``fig.texts``，所以 CJK 校验不会因此漏掉这段文字。
    fig.text(0.5, -0.02, note, fontsize=7, style="italic", color="dimgray",
             ha="center", va="top")

    # 不走 analysis._finish：它无条件调 tight_layout，而 tight_layout 不认
    # 坐标区外的 fig.legend（会告警并把图例裁掉）。CJK 校验照旧执行 ——
    # 那才是 _finish 里真正不能省的部分。
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_cjk_in_figure(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E1 上随机噪声响应 × 异构度的良性/恶意分组折线图")
    parser.add_argument("--raw-glob", default="results/raw/exp_e_E1_*.csv",
                        help="逐客户端的 E1 原始 CSV")
    parser.add_argument("--out-dir", default="results/figs")
    parser.add_argument("--summary-out",
                        default="results/exp_e_noise_response.csv")
    parser.add_argument("--metric", default="asr_targeted",
                        choices=["asr_targeted", "asr_targeted_unfiltered",
                                 "error_rate"],
                        help="主指标。asr_targeted 排除真实标签为目标类的样本，"
                             "与实验 E 的既有结论同口径")
    parser.add_argument("--cell", default="E1")
    args = parser.parse_args(argv)

    summary = summarize_noise(args.raw_glob, metric=args.metric, cell=args.cell)
    baselines = baseline_by_group(args.raw_glob, metric=args.metric)

    out_summary = Path(args.summary_out)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"[analysis_e_noise] {len(summary)} 行 -> {out_summary}")

    figure = plot_noise_response(
        summary, baselines,
        Path(args.out_dir) / f"exp_e_noise_response_{args.metric}.png",
        metric=args.metric)
    print(f"[analysis_e_noise] 图 -> {figure}")

    # 终端表：直接贴回来即可
    columns = ["alpha", "group", "perturb_mode", "n_clients", "asr_mean",
               "asr_std", "baseline_mean", "lift_mean", "lift_std",
               "mw_p_lift_benign_vs_malicious"]
    print(f"\n=== {args.cell} / {args.metric} ===")
    print(summary[columns].to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))

    alphas = [float(a) for a in _alpha_order(summary)]
    if len(alphas) < 2:
        print(f"\n⚠️ 只有一个 alpha（{alphas}），横轴退化为单点，读不出趋势。")
        print("   需要在其余 alpha 上补跑 E1：")
        print("   python -m diag.exp_e --cell E1 --alpha <a> --seed 0")
    n_mal = summary[summary["is_malicious"]]["n_clients"].max()
    print(f"\n注：恶意客户端 n={n_mal}，组间检验功效很低，"
          f"p 值只作参考，不构成结论。")
    missing = int(summary["n_missing_baseline"].sum())
    if missing:
        print(f"⚠️ 有 {missing} 个 (客户端, 扰动) 条目找不到对应的 '{BASELINE_MODE}' "
              f"基线行，其 lift 记为 nan（未填 0）。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
