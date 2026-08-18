"""跨恶意客户端数量的 ACC / ASR 对比图（图 J-6）。

一张图两个面板，横轴是恶意客户端总数，每条线是一种防御：

- 上panel：ASR —— 后门到底植没植进去；
- 下panel：ACC —— 为此付出的干净精度代价。

回答的是"Bad-PFL 需要多少曝光量才能植入后门"这个问题。若 ASR 随恶意客户端
数单调下降，说明它**不是**一个"只要有一个对抗噪声生成器就能成功"的过程，
而是需要累积一定数量的恶意更新进入聚合。

# 三件在这张图上很容易做错的事

1. **密度不能从文件名猜。** 优先读 CSV 里的 ``bad_client_num`` 列；
   没有这一列的旧 CSV 才回退到解析文件名里的 ``_bad{N}``；两者都没有时
   **报错并列出文件**，不套用 config 的默认值 —— 静默归错密度会让整张图失效。

2. **误差棒只能是跨 seed 的。** 尾部 K 轮来自同一条训练轨迹，彼此高度自相关，
   它们的 std 会**显著低估**不确定性。这个数仍然算出来放进 CSV
   （``tail_std_within_run``），但**绝不画成误差棒**，也不当不确定度用。
   只有一个 seed 时不画误差棒，并在图注里说明。

3. **缺格不能连线。** 某个防御缺中间某个密度时，matplotlib 会把两端直接连起来,
   看上去就像测过。这里用 nan 断开线段，并在终端列出缺格。

用法::

    python -m diag.analysis_density \\
        --implantation-glob "results/raw/exp_ij_implantation_*.csv" \\
        --out results/figs/exp_ij_density.png \\
        --summary-out results/exp_ij_density.csv --tail 5

⚠️ 图中一律英文，``assert_no_cjk_in_figure`` 在保存前扫描整张图。
"""

from __future__ import annotations

import argparse
import glob as globlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import assert_no_cjk_in_figure
from .analysis_ij import DEFENSE_STYLE, _style

__all__ = ["parse_bad_num", "load_implantation", "tail_summary",
           "accuracy_cost", "plot_density", "main"]

_BAD_IN_NAME = re.compile(r"_bad(\d+)")

# 排除类防御的上界对照不是"一种方法"，画成黑色虚线与其余曲线区分
_REFERENCE_DEFENSES = ("oracle_exclude",)


def parse_bad_num(path: str) -> Optional[int]:
    """从文件名里取 ``_bad{N}``；取不到返回 ``None``（**不猜默认值**）。"""
    match = _BAD_IN_NAME.search(Path(path).name)
    return int(match.group(1)) if match else None


def load_implantation(pattern: str,
                      assume_bad_num: Optional[int] = None) -> pd.DataFrame:
    """读入植入 CSV 并补齐 ``bad_client_num``。

    ``assume_bad_num`` 只作用于**既没有该列、文件名里也没有 tag** 的文件 ——
    早于该列存在时跑出来的 run 就属于这种。不传就报错。
    """
    paths = sorted(globlib.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"没有匹配 {pattern} 的植入 CSV")

    frames: List[pd.DataFrame] = []
    unresolved: List[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_file"] = Path(path).name

        column = frame["bad_client_num"] if "bad_client_num" in frame else None
        has_column = (column is not None
                      and pd.to_numeric(column, errors="coerce").ge(0).all())
        if has_column:
            frame["bad_client_num"] = pd.to_numeric(column).astype(int)
        else:
            from_name = parse_bad_num(path)
            if from_name is not None:
                frame["bad_client_num"] = int(from_name)
            elif assume_bad_num is not None:
                frame["bad_client_num"] = int(assume_bad_num)
            else:
                unresolved.append(Path(path).name)
                continue
        frames.append(frame)

    if unresolved:
        raise ValueError(
            "以下 CSV 既没有 bad_client_num 列、文件名里也没有 _bad{N} tag，"
            "无法确定恶意客户端数：\n  " + "\n  ".join(unresolved)
            + "\n用 --assume-bad-num N 显式指定，或者带 --run-tag 重跑。"
              "不套用 config 默认值 —— 归到错误的密度上整张图就废了。")
    if not frames:
        raise ValueError(f"匹配到 {len(paths)} 个文件，但全都是空表")
    return pd.concat(frames, ignore_index=True)


def tail_summary(frame: pd.DataFrame, tail: int = 5,
                 asr_column: str = "asr_personalized_targeted",
                 acc_column: str = "acc_personalized",
                 group_by: Sequence[str] = ("defense", "bad_client_num")
                 ) -> pd.DataFrame:
    """每个 run 取**末尾 tail 轮**的均值，再按 ``group_by`` 跨 seed 汇总。

    两级聚合是刻意的：先在 run 内部把尾部轮次压成一个数（消掉轮间抖动），
    再跨 seed 求均值与 std。直接把所有尾部行混在一起做 std，会把 run 内的
    自相关抖动当成 seed 间差异。

    ``group_by`` 决定第二级怎么并。默认 ``(defense, bad_client_num)`` —— 同一
    格子的多个 seed 并成一行。图 J-6 要的是这个。把 ``run_id`` 加进去就变成
    一行一个 run（``analysis_exposure`` 要按 run 与 E 对齐时用）。
    """
    group_by = list(group_by)
    for column in ("round", "defense", "bad_client_num", "seed",
                   asr_column, acc_column, *group_by):
        if column not in frame.columns:
            raise KeyError(f"植入 CSV 缺列 '{column}'；实际列：{list(frame.columns)}")

    run_keys = list(dict.fromkeys([*group_by, "seed"]))
    per_run: List[Dict[str, Any]] = []
    short_runs: List[str] = []
    for values, group in frame.groupby(run_keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        labels = dict(zip(run_keys, values))
        group = group.sort_values("round")
        if len(group) < tail:
            short_runs.append(
                f"{labels}（只有 {len(group)} 个评估行，需要 {tail}）")
            continue
        block = group.iloc[-tail:]
        per_run.append({
            **{k: (int(v) if k in ("bad_client_num", "seed") else str(v))
               for k, v in labels.items()},
            "asr": float(block[asr_column].mean()),
            "acc": float(block[acc_column].mean()),
            # 同一条轨迹内的离散度。**不是不确定度**，只用来发现尾部还没稳定。
            "asr_within": float(block[asr_column].std(ddof=1)),
            "acc_within": float(block[acc_column].std(ddof=1)),
            "last_round": int(block["round"].max()),
        })
    if short_runs:
        raise ValueError("以下 run 的评估行数不足 tail，拒绝用更短的窗口凑数"
                         "（会和其他 run 不同口径）：\n  " + "\n  ".join(short_runs))
    if not per_run:
        raise ValueError("没有任何 run 满足 tail 要求")

    runs = pd.DataFrame(per_run)
    rows: List[Dict[str, Any]] = []
    for values, group in runs.groupby(group_by):
        values = values if isinstance(values, tuple) else (values,)
        n_seeds = int(len(group))
        rows.append({
            **dict(zip(group_by, values)),
            "asr_mean": float(group["asr"].mean()),
            "acc_mean": float(group["acc"].mean()),
            # 跨 seed 的 std —— 只有 >=2 个 seed 时才有定义，才配当误差棒
            "asr_std_across_seeds": (float(group["asr"].std(ddof=1))
                                     if n_seeds >= 2 else np.nan),
            "acc_std_across_seeds": (float(group["acc"].std(ddof=1))
                                     if n_seeds >= 2 else np.nan),
            "tail_std_within_run": float(group["asr_within"].mean()),
            "n_seeds": n_seeds, "tail": int(tail),
            "last_round": int(group["last_round"].max()),
            "seeds": ",".join(str(s) for s in sorted(group["seed"])),
        })
    return pd.DataFrame(rows).sort_values(group_by).reset_index(drop=True)


def accuracy_cost(summary: pd.DataFrame,
                  baseline: str = "fedavg") -> pd.DataFrame:
    """每压低 1 个 ASR 点，要付出多少个 ACC 点 —— 同密度下与无防御基线比。

    ASR 没有被压低（差 <= 0）时该值**无定义，留空**，不填 0：
    "代价为 0" 和 "根本没起作用" 是两件完全不同的事。
    """
    summary = summary.copy()
    summary["asr_reduction"] = np.nan
    summary["acc_drop"] = np.nan
    summary["acc_points_per_asr_point"] = np.nan

    for bad, group in summary.groupby("bad_client_num"):
        ref = group[group["defense"] == baseline]
        if ref.empty:
            continue
        asr_ref = float(ref["asr_mean"].iloc[0])
        acc_ref = float(ref["acc_mean"].iloc[0])
        for index in group.index:
            if summary.at[index, "defense"] == baseline:
                continue
            reduction = asr_ref - float(summary.at[index, "asr_mean"])
            drop = acc_ref - float(summary.at[index, "acc_mean"])
            summary.at[index, "asr_reduction"] = reduction
            summary.at[index, "acc_drop"] = drop
            if reduction > 0:
                summary.at[index, "acc_points_per_asr_point"] = drop / reduction
    return summary


def _series(summary: pd.DataFrame, defense: str, xs: Sequence[int],
            column: str) -> np.ndarray:
    """按 ``xs`` 的顺序取值，缺格填 nan —— matplotlib 会在 nan 处断线。"""
    block = summary[summary["defense"] == defense].set_index("bad_client_num")
    return np.array([float(block[column].get(x, np.nan)) for x in xs])


def plot_density(summary: pd.DataFrame, out_path,
                 paper_asr: Optional[float] = None,
                 paper_bad_num: Optional[int] = None) -> Path:
    """图 J-6：上 ASR / 下 ACC，横轴恶意客户端数（分类位置，不按数值刻度）。"""
    xs = sorted(summary["bad_client_num"].unique().tolist())
    positions = np.arange(len(xs), dtype=float)
    multi_seed = bool((summary["n_seeds"] >= 2).any())

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    gaps: List[str] = []

    for defense in sorted(summary["defense"].unique()):
        label, color = _style(defense)
        reference = defense in _REFERENCE_DEFENSES
        kwargs = dict(color=color, marker="o", markersize=5,
                      linestyle="--" if reference else "-",
                      linewidth=1.6, label=label)
        for axis, metric in ((top, "asr"), (bottom, "acc")):
            values = _series(summary, defense, xs, f"{metric}_mean")
            errors = _series(summary, defense, xs, f"{metric}_std_across_seeds")
            if multi_seed and np.isfinite(errors).any():
                axis.errorbar(positions, values, yerr=errors, capsize=3,
                              **kwargs)
            else:
                axis.plot(positions, values, **kwargs)
            kwargs.pop("label", None)          # 图例只画一次
        missing = [x for x, v in zip(xs, _series(summary, defense, xs, "asr_mean"))
                   if not np.isfinite(v)]
        if missing:
            gaps.append(f"{defense}: 缺 bad={missing}")

    if paper_asr is not None and paper_bad_num is not None:
        if paper_bad_num in xs:
            # 单个标记，不画横线 —— 论文的数字只对它自己那个密度成立
            top.plot([xs.index(paper_bad_num)], [paper_asr], marker="*",
                     markersize=14, color="crimson", linestyle="none",
                     label=f"Bad-PFL paper ({paper_asr:.2%} @ {paper_bad_num})")

    top.set_ylabel("Backdoor ASR (personalized)")
    top.set_ylim(0.0, 1.0)
    top.grid(alpha=0.3)
    # 图例放到坐标区**外面**。loc="best" 在实测里压住了 bad=5 处的 FLAME /
    # Invariant 两条线 —— 而那正是这张图最关键的一段。放外面才不依赖数据形状。
    top.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0.0)
    top.set_title("Backdoor implantation vs. number of malicious clients")

    bottom.set_ylabel("Clean accuracy (personalized)")
    bottom.set_xlabel("Number of malicious clients (out of 100)")
    bottom.grid(alpha=0.3)
    bottom.set_xticks(positions)
    bottom.set_xticklabels([str(x) for x in xs])

    note = (f"Each point = mean over the last {int(summary['tail'].iloc[0])} "
            f"evaluation rounds of a run.\n")
    note += ("Error bars = std across seeds."
             if multi_seed else
             "Single seed per point: no error bars are shown (within-run tail "
             "scatter is autocorrelated and is not an uncertainty estimate).")
    fig.tight_layout()
    # tight_layout 之后再放脚注，否则它会被当成坐标区的一部分重新排版
    fig.subplots_adjust(bottom=0.15)
    # va="top" 且 y 太小时文字会伸到画布**外面**，bbox_inches="tight" 于是
    # 把画布向下撑开一大片空白。放在 xlabel 正下方（约 0.085）才贴得住。
    fig.text(0.5, 0.085, note, ha="center", va="top", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_cjk_in_figure(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if gaps:
        print("[analysis_density] 曲线在这些密度上断开（未测，不插值）：")
        for gap in gaps:
            print(f"  {gap}")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="跨恶意客户端数量的 ACC/ASR 对比图（图 J-6）")
    parser.add_argument("--implantation-glob",
                        default="results/raw/exp_ij_implantation_*.csv")
    parser.add_argument("--out", default="results/figs/exp_ij_density.png")
    parser.add_argument("--summary-out", default="results/exp_ij_density.csv")
    parser.add_argument("--tail", type=int, default=5,
                        help="尾部取几个评估行做均值（默认 5，与其余实验同口径）")
    parser.add_argument("--asr-column", default="asr_personalized_targeted",
                        help="默认个性化模型 + 排除目标类原样本")
    parser.add_argument("--acc-column", default="acc_personalized")
    parser.add_argument("--baseline", default="fedavg",
                        help="算 ACC 代价时的参照防御")
    parser.add_argument("--assume-bad-num", type=int, default=None,
                        help="仅用于既无 bad_client_num 列、文件名也无 tag 的旧 CSV")
    parser.add_argument("--paper-asr", type=float, default=None,
                        help="论文报告的 ASR，画成单个星号标记（不画横线）")
    parser.add_argument("--paper-bad-num", type=int, default=None)
    args = parser.parse_args(argv)

    frame = load_implantation(args.implantation_glob, args.assume_bad_num)
    summary = tail_summary(frame, tail=args.tail,
                           asr_column=args.asr_column,
                           acc_column=args.acc_column)
    summary = accuracy_cost(summary, baseline=args.baseline)

    out_summary = Path(args.summary_out)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"[analysis_density] 汇总 {len(summary)} 行 -> {out_summary}")

    figure = plot_density(summary, args.out, paper_asr=args.paper_asr,
                          paper_bad_num=args.paper_bad_num)
    print(f"[analysis_density] 图 -> {figure}")

    print(f"\n=== 尾部 {args.tail} 轮均值（{args.asr_column}）===")
    header = (f"  {'defense':<22}{'bad':>4}{'ASR':>9}{'ACC':>9}"
              f"{'ΔASR':>9}{'ΔACC':>9}{'ACC/ASR':>9}{'seeds':>7}")
    print(header)
    for _, row in summary.sort_values(["bad_client_num", "defense"]).iterrows():
        def _fmt(key: str, width: int = 9) -> str:
            value = row.get(key, np.nan)
            # 无定义就留空。填 0 会被当成"代价为零"，那是完全相反的结论。
            return (f"{value:>{width}.4f}" if np.isfinite(value)
                    else f"{'':>{width}}")
        print(f"  {row['defense']:<22}{int(row['bad_client_num']):>4}"
              f"{_fmt('asr_mean')}{_fmt('acc_mean')}"
              f"{_fmt('asr_reduction')}{_fmt('acc_drop')}"
              f"{_fmt('acc_points_per_asr_point')}"
              f"{int(row['n_seeds']):>7}")

    single = summary[summary["n_seeds"] < 2]
    if not single.empty:
        print(f"\n  ⚠️ {len(single)}/{len(summary)} 个格子只有 1 个 seed，"
              f"没有不确定度可言。ΔASR 的差异是否超出 seed 抖动**无法判断** —— "
              f"要下结论得至少补到 3 个 seed。")
    unknown = sorted(set(summary["defense"]) - set(DEFENSE_STYLE))
    if unknown:
        print(f"  ⚠️ 这些防御没有配色，图里统一是灰色：{unknown}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
