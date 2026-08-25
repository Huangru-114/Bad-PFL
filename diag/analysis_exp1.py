"""实验 1 / 1B 的分析与图。

======  ==================================================  ================
图      内容                                                  回答的问题
======  ==================================================  ================
E1-1    ASR vs round，每个剂量一条跨 seed 均值线                后门什么时候形成
E1-2    ASR vs MTA 散点（色=剂量, marker=seed）                MTA* 存在吗
E1-4    剂量-响应：各子图只动一个变量（真单轴扫描）             有效边界在哪
E1-5    剂量-响应热力图（N_m × ρ_p 的最终 ASR）                有效边界的形状（全网格）
E1B-1   各调度的 ASR_t，投毒轮次加底色                          时间结构的影响
E1B-2   停攻后对齐的 ASR_{t+k}                                后门能自持吗
======  ==================================================  ================

# 为什么 ASR-vs-MTA 单独证明不了因果

单个 run 里 MTA 随 t 单调上升，所以 ``ASR~MTA`` 与 ``ASR~t`` 是同一张图换了
个横轴 —— 两者的相关性都会很高，而这**说明不了** MTA 是不是因。

本模块因此做两件不同的事：

1. **``threshold_verdict``（相关性证据，非因果）**：比较"ASR 越过某水平时的
   MTA"与"越过时的轮次"哪个跨 run 更集中。若 MTA 明显更集中，说明 MTA 是更好
   的**预测量**。这仍然不是因果 —— 只是把"该看哪个变量"这件事定下来。

2. **``onset_analysis``（1B 的识别策略）**：这才是能分开 ``t→MTA`` 与
   ``MTA→ASR`` 的地方。不同调度让攻击在**不同的 MTA 水平**上开始，同时可以
   控制住"投毒了多少轮"。在投毒轮数相当的前提下，若起始 MTA 高的那些 run
   ASR 明显更高，才是 ``MTA→ASR`` 的证据。

**不要**把第 1 步的结论写成因果 —— 代码里的措辞已经区分开了，报告里也要。

⚠️ 图中一律英文。
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import assert_no_cjk_in_figure

__all__ = ["load_runs", "run_key", "crossing_table", "threshold_verdict",
           "restrict_to_common_dose", "resolve_asr_column",
           "onset_analysis", "dose_response", "persistence_table",
           "plot_e1_1", "plot_e1_2", "plot_e1_4", "plot_e1_5",
           "plot_e1b_1", "plot_e1b_2", "main"]

ASR_COLUMN = "asr_personalized_targeted"
MTA_COLUMN = "mta_personalized"
LOSS_COLUMN = "clean_loss_personalized"
_REQUIRED = ("round", "seed", "bad_client_num", "poison_rate", "schedule",
             ASR_COLUMN, MTA_COLUMN)


def resolve_asr_column(frame: pd.DataFrame, chosen: str) -> Tuple[str, str]:
    """选定当 ASR 用的列，返回 ``(列名, 提示语)``。

    优先用 ``chosen``（默认论文口径 ``asr_paper_all``）；该列不存在或整列 nan
    时回退到旧的 perturb 分解口径 ``asr_personalized_targeted``（仅良性、会低估,
    见 HANDOFF），并给出告警语。两者都没有则抛 ``KeyError`` —— 绝不静默乱选。
    """
    if chosen in frame.columns and bool(np.isfinite(frame[chosen]).any()):
        note = ("（论文尺：原始触发器 + 各客户端自己 test loader）"
                if chosen.startswith("asr_paper") else "")
        return chosen, note
    if "asr_personalized_targeted" in frame.columns:
        return "asr_personalized_targeted", (
            f"⚠️ 数据里没有可用的 '{chosen}' 列，回退到旧口径 "
            f"asr_personalized_targeted（perturb 分解，仅良性，会低估）。"
            f"这多半是 2026-08 之前的 run —— 重跑才有论文口径。")
    raise KeyError(
        f"数据里既没有 '{chosen}' 也没有 asr_personalized_targeted，无法确定 "
        f"ASR 列。可用列：{sorted(frame.columns)}")


def load_runs(pattern: str) -> pd.DataFrame:
    """读入植入 CSV。剂量与调度都是**列**，不从文件名解析。"""
    paths = sorted(globlib.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"没有匹配 {pattern} 的植入 CSV")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        missing = [c for c in _REQUIRED if c not in frame.columns]
        if missing:
            raise KeyError(
                f"{Path(path).name} 缺列 {missing}。这些列是 2026-08 之后的 run "
                f"才有的（track.py 的 IMPLANTATION_COLUMNS），旧 CSV 不能混进来 "
                f"—— 缺 poison_rate 会让所有剂量点挤成一个。")
        frame["source_file"] = Path(path).name
        frame["run_id"] = (Path(path).stem
                           .replace("exp_ij_implantation_", ""))
        frames.append(frame)
    if not frames:
        raise ValueError(f"匹配到 {len(paths)} 个文件，但全是空表")
    return pd.concat(frames, ignore_index=True)


def run_key(row: pd.Series) -> str:
    return (f"Nm={int(row['bad_client_num'])} "
            f"rho={float(row['poison_rate']):.2f}")


# ---------------------------------------------------------------------------
# 阈值：ASR 越过某水平时，MTA 与轮次哪个更集中
# ---------------------------------------------------------------------------
def crossing_table(frame: pd.DataFrame, level: float = 0.5) -> pd.DataFrame:
    """每个 run 首次 ``ASR >= level`` 时的 (round, MTA, clean loss)。

    从未越过的 run **保留在表里但值为 nan** —— 丢掉它们会让阈值看起来比实际
    更普遍（只统计成功的那些，当然都在同一个位置附近）。
    """
    rows: List[Dict[str, Any]] = []
    for run_id, group in frame.groupby("run_id"):
        group = group.sort_values("round")
        asr = group[ASR_COLUMN].to_numpy(dtype=float)
        crossed = np.flatnonzero(np.isfinite(asr) & (asr >= level))
        first = group.iloc[0]
        row = {
            "run_id": str(run_id), "seed": int(first["seed"]),
            "bad_client_num": int(first["bad_client_num"]),
            "poison_rate": float(first["poison_rate"]),
            "schedule": str(first["schedule"]),
            "level": float(level), "crossed": bool(crossed.size),
            "final_asr": float(asr[np.isfinite(asr)][-1])
            if np.isfinite(asr).any() else np.nan,
        }
        if crossed.size:
            at = group.iloc[int(crossed[0])]
            row.update({"round_at_cross": float(at["round"]),
                        "mta_at_cross": float(at[MTA_COLUMN]),
                        "loss_at_cross": float(at.get(LOSS_COLUMN, np.nan))})
        else:
            row.update({"round_at_cross": np.nan, "mta_at_cross": np.nan,
                        "loss_at_cross": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["bad_client_num", "poison_rate", "seed"]).reset_index(drop=True)


def threshold_verdict(crossings: pd.DataFrame,
                      frame: pd.DataFrame) -> Dict[str, Any]:
    """MTA 是不是比轮次更好的**预测量**（相关性证据，不是因果）。

    两个量的单位不同，直接比 std 没有意义，所以各自除以全局跨度得到无量纲的
    相对离散::

        spread_mta   = std(mta_at_cross)   / (MTA 的全局跨度)
        spread_round = std(round_at_cross) / (轮次的全局跨度)

    # 为什么光比这两个数会得出假阳性

    MTA(t) 是**饱和**的：训练后期一大段轮次映射到很窄的一段 MTA。于是即便
    起飞完全由时间/剂量决定、与 MTA 无关，``spread_mta`` 也会机械地比
    ``spread_round`` 小。实测：在一份 ASR 纯粹由剂量驱动、各 run 共用同一条
    MTA 曲线的合成数据上，这个朴素比较照样报出"MTA 更集中"。

    所以必须有一个**零假设**：把观测到的 ``round_at_cross`` 喂进各 run 汇总出
    的 MTA(t) 中位曲线，得到"纯时间驱动时 MTA-at-cross 会有多集中"
    （``null_spread_mta``）。只有实测比这个零假设**还要**明显更集中，
    MTA 才算带来了轮次之外的信息。
    """
    used = crossings[crossings["crossed"]]
    result: Dict[str, Any] = {
        "level": float(crossings["level"].iloc[0]) if len(crossings) else np.nan,
        "n_runs": int(len(crossings)),
        "n_crossed": int(len(used)),
        "n_never_crossed": int((~crossings["crossed"]).sum()),
    }
    if len(used) < 3:
        result["verdict"] = (
            f"未能确定：只有 {len(used)} 个 run 越过了 ASR={result['level']}，"
            f"谈不上'集中'。要么降低 level，要么补更强的剂量。")
        return result

    mta_range = float(np.nanmax(frame[MTA_COLUMN]) - np.nanmin(frame[MTA_COLUMN]))
    round_range = float(frame["round"].max() - frame["round"].min())
    spread_mta = (float(used["mta_at_cross"].std(ddof=1)) / mta_range
                  if mta_range > 0 else np.nan)
    spread_round = (float(used["round_at_cross"].std(ddof=1)) / round_range
                    if round_range > 0 else np.nan)
    # 零假设：纯时间驱动 + MTA(t) 的饱和形状，会给出多集中的 mta_at_cross
    curve = frame.groupby("round")[MTA_COLUMN].median().sort_index()
    predicted = np.interp(used["round_at_cross"].to_numpy(dtype=float),
                          curve.index.to_numpy(dtype=float),
                          curve.to_numpy(dtype=float))
    null_spread_mta = (float(np.std(predicted, ddof=1)) / mta_range
                       if mta_range > 0 and len(predicted) > 1 else np.nan)

    result.update({
        "mta_at_cross_mean": float(used["mta_at_cross"].mean()),
        "mta_at_cross_std": float(used["mta_at_cross"].std(ddof=1)),
        "round_at_cross_mean": float(used["round_at_cross"].mean()),
        "round_at_cross_std": float(used["round_at_cross"].std(ddof=1)),
        "relative_spread_mta": spread_mta,
        "relative_spread_round": spread_round,
        "null_relative_spread_mta": null_spread_mta,
    })

    if not (np.isfinite(spread_mta) and np.isfinite(spread_round)
            and np.isfinite(null_spread_mta)):
        result["verdict"] = "未能确定：跨度为 0，无法归一化"
    elif spread_mta < 0.5 * null_spread_mta:
        result["verdict"] = (
            f"MTA 是更紧的预测量：越过 ASR={result['level']} 时 MTA="
            f"{result['mta_at_cross_mean']:.3f}±{result['mta_at_cross_std']:.3f}"
            f"（相对离散 {spread_mta:.3f}），明显小于纯时间驱动的零假设 "
            f"{null_spread_mta:.3f}。**这是相关性证据，不是因果** —— "
            f"因果要看 1B 的 onset_analysis。")
    elif spread_mta > 1.5 * null_spread_mta:
        result["verdict"] = (
            f"MTA 比零假设还**松**（{spread_mta:.3f} vs {null_spread_mta:.3f}）"
            f"—— 各 run 的 MTA 轨迹本身差异很大，而起飞点并不跟着 MTA 走。"
            f"MTA* 的说法不成立。")
    else:
        result["verdict"] = (
            f"未能确定：实测相对离散 {spread_mta:.3f} 与纯时间驱动的零假设 "
            f"{null_spread_mta:.3f} 相当 —— MTA 看起来集中，只是因为 MTA(t) "
            f"在训练后期饱和，并不能说明它带来了轮次之外的信息。"
            f"要分开必须用 1B 的调度变化。")
    return result


# ---------------------------------------------------------------------------
# 1B：识别策略
# ---------------------------------------------------------------------------
def onset_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """每个 run 的"攻击起始时的 MTA"与"投毒轮数"，以及最终 ASR。

    这是**唯一**能把 ``t→MTA`` 与 ``MTA→ASR`` 分开的表：不同调度让攻击在不同
    的 MTA 水平上开始，同时投毒轮数可以对齐。
    """
    rows: List[Dict[str, Any]] = []
    for run_id, group in frame.groupby("run_id"):
        group = group.sort_values("round")
        active = group["attack_active_this_round"].astype(bool) \
            if "attack_active_this_round" in group.columns \
            else pd.Series(True, index=group.index)
        first = group.iloc[0]
        onset = group[active]
        asr = group[ASR_COLUMN].to_numpy(dtype=float)
        finite = asr[np.isfinite(asr)]
        rows.append({
            "run_id": str(run_id), "seed": int(first["seed"]),
            "schedule": str(first["schedule"]),
            "bad_client_num": int(first["bad_client_num"]),
            "poison_rate": float(first["poison_rate"]),
            # 起始 MTA：第一个投毒轮**之前**已知的 MTA
            "mta_at_onset": (float(onset.iloc[0][MTA_COLUMN])
                             if len(onset) else np.nan),
            "round_at_onset": (float(onset.iloc[0]["round"])
                               if len(onset) else np.nan),
            # 评估轮里有多少轮是投毒的。**不是**真实投毒轮数（评估是抽样的），
            # 所以只能用来在调度之间做相对比较，不能当绝对剂量。
            "n_active_eval_rounds": int(active.sum()),
            "final_asr": float(finite[-1]) if finite.size else np.nan,
            "peak_asr": float(np.nanmax(asr)) if finite.size else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["schedule", "seed"]).reset_index(
        drop=True)


def persistence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """停攻之后 ASR 的衰减，按"距最后一个投毒轮多少轮"对齐。

    只包含**确实停过**的 run（early / burst / intermittent）。continuous
    永远不停，放进来会在 k=0 处堆一堆与衰减无关的点。
    """
    rows: List[Dict[str, Any]] = []
    for run_id, group in frame.groupby("run_id"):
        group = group.sort_values("round")
        if "attack_active_this_round" not in group.columns:
            continue
        active = group["attack_active_this_round"].astype(bool).to_numpy()
        if not active.any() or active[-1]:
            continue                    # 从没攻击过，或到最后仍在攻击
        last_active = int(np.flatnonzero(active)[-1])
        first = group.iloc[0]
        for offset in range(last_active, len(group)):
            row = group.iloc[offset]
            rows.append({
                "run_id": str(run_id), "seed": int(first["seed"]),
                "schedule": str(first["schedule"]),
                "rounds_since_attack_stopped": int(
                    row["round"] - group.iloc[last_active]["round"]),
                "asr": float(row[ASR_COLUMN]),
                "mta": float(row[MTA_COLUMN]),
            })
    return pd.DataFrame(rows)


def dose_response(frame: pd.DataFrame, tail: int = 3) -> pd.DataFrame:
    """每个 (N_m, ρ_p, seed) 的尾部平均 ASR 与 MTA。"""
    rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(["bad_client_num", "poison_rate", "seed"]):
        group = group.sort_values("round")
        if len(group) < tail:
            continue
        block = group.iloc[-tail:]
        rows.append({
            "bad_client_num": int(keys[0]), "poison_rate": float(keys[1]),
            "seed": int(keys[2]),
            "asr": float(block[ASR_COLUMN].mean()),
            "mta": float(block[MTA_COLUMN].mean()),
            "n_tail": int(tail),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _finish(fig, out_path, note: str = "") -> Path:
    if note:
        fig.tight_layout()
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7.5)
    else:
        fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_cjk_in_figure(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _dose_colors(frame: pd.DataFrame) -> Dict[Tuple[int, float], Any]:
    """每个剂量 (N_m, ρ_p) 一个**对比色**（tab10 定性色板），而不是同色系深浅。

    单色系渐变让相邻剂量几乎分不开；定性色板每条曲线一眼可辨。剂量数
    ≤10 用 tab10，更多则退到 tab20。
    """
    keys = sorted({(int(b), float(p)) for b, p in
                   zip(frame["bad_client_num"], frame["poison_rate"])})
    palette = plt.get_cmap("tab10" if len(keys) <= 10 else "tab20")
    return {key: palette(i % palette.N) for i, key in enumerate(keys)}


def _dose_label(key: Tuple[int, float]) -> str:
    return f"Nm={key[0]}, rho={key[1]:g}"


def _seed_mean_curve(group: pd.DataFrame) -> pd.DataFrame:
    """把一个剂量下多 seed 的 (round, ASR) 汇成 round -> 均值/最小/最大。

    各 seed 的评估轮次一致（同一个 eval_every），按 round 对齐再求统计。
    """
    agg = group.groupby("round")[ASR_COLUMN].agg(["mean", "min", "max"])
    return agg.sort_index()


_VAR_SHORT = {"bad_client_num": "Nm", "poison_rate": "rho"}


def _fmt_level(var: str, value: Any) -> str:
    """把一个变量的取值格式化成 legend/标题用的短标签。"""
    short = _VAR_SHORT.get(var, var)
    return (f"{short}={int(value)}" if var == "bad_client_num"
            else f"{short}={float(value):g}")


def _level_colors(values: Sequence[Any]) -> Dict[Any, Any]:
    """给一组取值分配对比色（tab10），颜色在所有分面里保持一致。"""
    keys = sorted(set(values))
    palette = plt.get_cmap("tab10" if len(keys) <= 10 else "tab20")
    return {key: palette(i % palette.N) for i, key in enumerate(keys)}


def _facet_layout(n: int) -> Tuple[int, int]:
    ncol = min(3, max(1, n))
    nrow = (n + ncol - 1) // ncol
    return nrow, ncol


def _fig_axis_labels(fig, xlabel: str, ylabel: str) -> None:
    """给整张分面图加公共 x/y 轴标签。

    用 ``fig.text`` 而非 ``fig.supxlabel/supylabel`` —— 后者要 matplotlib>=3.4，
    集群版本未知，为免在出图阶段直接崩掉，退回到到处都有的 ``fig.text``。
    """
    fig.text(0.5, 0.015, xlabel, ha="center", fontsize=10)
    fig.text(0.008, 0.5, ylabel, va="center", rotation="vertical", fontsize=10)


def _facet_grid(facet_by: str):
    """(nrow, ncol, fig, axes_flat) —— 供 E1/E2 分面复用的画布。"""
    def build(n: int):
        nrow, ncol = _facet_layout(n)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        return fig, axes.ravel()
    return build


def plot_e1_1(frame: pd.DataFrame, out_path, *,
              facet_by: str = "bad_client_num",
              line_by: str = "poison_rate") -> Path:
    """E1-1：ASR vs round，**分面小多图**。

    每个分面固定 ``facet_by`` 的一个取值，面内按 ``line_by`` 的每个取值画一条
    跨 seed 均值线（min-max 包络带）。颜色按 ``line_by`` 统一，跨面可比。
    默认按 N_m 分面、面内不同 ρ；调用方再传 ``facet_by="poison_rate"`` 出另一版。
    """
    facets = sorted(frame[facet_by].unique())
    colors = _level_colors(frame[line_by].tolist())
    fig, axes = _facet_grid(facet_by)(len(facets))
    for axis, fval in zip(axes, facets):
        block = frame[frame[facet_by] == fval]
        for lval in sorted(block[line_by].unique()):
            sub = block[block[line_by] == lval]
            curve = _seed_mean_curve(sub)
            rounds = curve.index.to_numpy(dtype=float)
            axis.fill_between(rounds, curve["min"], curve["max"],
                              color=colors[lval], alpha=0.15, linewidth=0)
            axis.plot(rounds, curve["mean"], color=colors[lval], linewidth=1.8,
                      marker="o", markersize=2.5, label=_fmt_level(line_by, lval))
        axis.set_title(_fmt_level(facet_by, fval), fontsize=9)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
    for axis in axes[len(facets):]:      # 多出来的空面隐藏
        axis.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center right",
               title=_VAR_SHORT.get(line_by, line_by))
    _fig_axis_labels(fig, "Communication round",
                     f"Backdoor ASR [{ASR_COLUMN}]")
    fig.suptitle(f"E1-1  Backdoor formation (faceted by "
                 f"{_VAR_SHORT.get(facet_by, facet_by)}, mean over seeds)")
    return _finish(fig, out_path,
                   "One panel per "
                   f"{_VAR_SHORT.get(facet_by, facet_by)}; one line per "
                   f"{_VAR_SHORT.get(line_by, line_by)} (mean over seeds, band "
                   "= min-max).")


def plot_e1_2(frame: pd.DataFrame, out_path,
              verdict: Optional[Dict[str, Any]] = None, *,
              facet_by: str = "bad_client_num",
              line_by: str = "poison_rate") -> Path:
    """E1-2：ASR vs MTA 散点，**分面小多图**。

    面内不连线（run 内 MTA 随 t 单调，连线只是把 ASR~t 换个横轴）；颜色按
    ``line_by`` 统一、marker 按 seed。阈值判据的 MTA 竖线在每个面里都画。
    """
    facets = sorted(frame[facet_by].unique())
    colors = _level_colors(frame[line_by].tolist())
    markers = ["o", "s", "^", "D", "v", "P"]
    seeds = sorted(frame["seed"].unique())
    fig, axes = _facet_grid(facet_by)(len(facets))
    for axis, fval in zip(axes, facets):
        block = frame[frame[facet_by] == fval]
        seen = set()
        for run_id, group in block.groupby("run_id"):
            first = group.iloc[0]
            lval = first[line_by]
            marker = markers[seeds.index(int(first["seed"])) % len(markers)]
            label = _fmt_level(line_by, lval) if lval not in seen else None
            seen.add(lval)
            axis.scatter(group[MTA_COLUMN], group[ASR_COLUMN],
                         color=colors[lval], marker=marker, s=14, alpha=0.75,
                         edgecolors="none", label=label)
        if verdict and np.isfinite(verdict.get("mta_at_cross_mean", np.nan)):
            axis.axvline(verdict["mta_at_cross_mean"], color="crimson",
                         linestyle=":", linewidth=1.2)
        axis.set_title(_fmt_level(facet_by, fval), fontsize=9)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
    for axis in axes[len(facets):]:
        axis.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center right",
               title=_VAR_SHORT.get(line_by, line_by))
    _fig_axis_labels(fig, "Main task accuracy (MTA, personalized)",
                     f"Backdoor ASR [{ASR_COLUMN}]")
    facet_short = _VAR_SHORT.get(facet_by, facet_by)
    has_line = bool(verdict and np.isfinite(
        verdict.get("mta_at_cross_mean", np.nan)))
    title = f"E1-2  ASR vs MTA (faceted by {facet_short})"
    if has_line:
        title += f"; dotted line = MTA at ASR={verdict['level']:g}"
    fig.suptitle(title)
    return _finish(fig, out_path,
                   "Scatter (no connecting lines); colour = "
                   f"{_VAR_SHORT.get(line_by, line_by)}, marker = seed.  Within "
                   "a run MTA is monotone in t, so this cannot separate t->MTA "
                   "from MTA->ASR; that needs the 1B schedules.")


def _infer_cross_center(response: pd.DataFrame) -> Tuple[int, float]:
    """从十字扫描数据里推断固定点 (Nm_fixed, ρ_fixed)。

    十字扫描里：ρ 扫描那条线上的所有点共用同一个 Nm（= Nm_fixed），Nm 扫描
    那条线上的所有点共用同一个 ρ（= ρ_fixed）。所以 Nm_fixed = 与最多个不同 ρ
    共现的那个 Nm；ρ_fixed 同理。这样无需读 config 就能定位单轴。
    """
    nm_fixed = int(response.groupby("bad_client_num")["poison_rate"]
                   .nunique().idxmax())
    rho_fixed = float(response.groupby("poison_rate")["bad_client_num"]
                      .nunique().idxmax())
    return nm_fixed, rho_fixed


def plot_e1_4(response: pd.DataFrame, out_path) -> Path:
    """E1-4：剂量-响应，**每条子图只动一个变量**（真正的单轴扫描）。

    左：固定 ρ=ρ_fixed，扫 N_m；右：固定 N_m=N_m_fixed，扫 ρ。原实现把每个
    边际都用**全部**数据分组，于是两条线各自混进了另一个变量——没有意义。
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    nm_fixed, rho_fixed = _infer_cross_center(response)
    panels = (
        (left, "bad_client_num", "poison_rate", rho_fixed,
         "Number of malicious clients", f"rho fixed = {rho_fixed:g}"),
        (right, "poison_rate", "bad_client_num", nm_fixed,
         "Poisoning probability rho", f"Nm fixed = {nm_fixed}"),
    )
    for axis, xcol, other, other_fixed, xlabel, sub in panels:
        arm = response[response[other] == other_fixed]     # 只保留单轴那条线
        grouped = arm.groupby(xcol)[["asr"]]
        xs = sorted(arm[xcol].unique())
        means = [float(grouped.get_group(x)["asr"].mean()) for x in xs]
        axis.plot(range(len(xs)), means, marker="o", color="tab:blue",
                  linewidth=1.8, zorder=2)
        for index, x in enumerate(xs):
            values = grouped.get_group(x)["asr"].to_numpy()
            axis.scatter([index] * len(values), values, s=22, color="0.5",
                         zorder=3, alpha=0.85)
            axis.annotate(f"n={len(values)}", (index, 0.02), fontsize=7,
                          ha="center", color="0.4")
        axis.set_xticks(range(len(xs)))
        axis.set_xticklabels([f"{x:g}" for x in xs])
        axis.set_xlabel(f"{xlabel}\n({sub})")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
    left.set_ylabel("Backdoor ASR (tail mean)")
    fig.suptitle("E1-4  Dose-response along each single axis")
    return _finish(fig, out_path,
                   "Each panel holds the other variable fixed at the cross "
                   "centre, so it is a true single-axis sweep.  Grey dots are "
                   "individual seeds; blue line is their mean.")


def plot_e1_5(response: pd.DataFrame, out_path) -> Path:
    """E1-5：剂量-响应**热力图**（N_m × ρ_p 的最终 ASR），全因子网格的正确读法。

    十字扫描下网格是稀疏的（大量空格），热力图照样能画——缺的格子留白，不用
    0 填（铁律：无定义留空）。全因子填满后，这张图一眼看出有效边界的形状。
    """
    pivot = response.pivot_table(index="bad_client_num", columns="poison_rate",
                                 values="asr", aggfunc="mean")
    pivot = pivot.sort_index(ascending=True).sort_index(axis=1)
    rows = [int(r) for r in pivot.index]
    cols = [float(c) for c in pivot.columns]
    data = pivot.to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(1.4 * len(cols) + 2.6,
                                      0.7 * len(rows) + 2.2))
    mesh = axis.imshow(data, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(cols)))
    axis.set_xticklabels([f"{c:g}" for c in cols])
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([str(r) for r in rows])
    axis.set_xlabel("Poisoning probability rho")
    axis.set_ylabel("Number of malicious clients Nm")
    # 每格标数值；缺失格留白并写 "-"，让"没跑"与"跑出来是 0"区分开
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = data[i, j]
            if np.isfinite(value):
                axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                          fontsize=8,
                          color="white" if value < 0.55 else "black")
            else:
                axis.text(j, i, "-", ha="center", va="center", fontsize=8,
                          color="0.6")
    fig.colorbar(mesh, ax=axis, label="Backdoor ASR (tail mean)", shrink=0.85)
    axis.set_title("E1-5  Dose-response surface (ASR over the Nm x rho grid)")
    return _finish(fig, out_path,
                   "Cells are the tail-mean ASR averaged over seeds.  Blank/'-' "
                   "cells were not run (cross sweep leaves them empty; a full "
                   "grid fills them).")


def restrict_to_common_dose(frame: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """把 1B 的对比限制在**同一个剂量**上。

    实验 1 的剂量扫描全是 ``continuous``，直接并进来会让 continuous 那一栏
    塞进十几条不同剂量的曲线，与其余每栏两条并排 —— 看起来像是 continuous
    的方差特别大，其实是剂量在变。1B 要问的是"时间结构的影响"，
    剂量必须固定，否则时间和剂量混在一起。

    取非 continuous 调度里最常见的那个 (N_m, ρ_p) 作为基准剂量。
    """
    others = frame[frame["schedule"] != "continuous"]
    if others.empty:
        return frame, ""
    dose = (others.groupby(["bad_client_num", "poison_rate"])["run_id"]
            .nunique().idxmax())
    keep = ((frame["bad_client_num"] == dose[0])
            & (frame["poison_rate"] == dose[1]))
    dropped = frame.loc[~keep, "run_id"].nunique()
    note = (f"Nm={int(dose[0])}, rho={float(dose[1]):g}"
            + (f"（另有 {dropped} 个其它剂量的 run 未纳入本图）"
               if dropped else ""))
    return frame[keep], note


def plot_e1b_1(frame: pd.DataFrame, out_path) -> Path:
    """E1B-1：各调度的 ASR_t，投毒轮次加底色。剂量已统一。"""
    frame, dose_note = restrict_to_common_dose(frame)
    schedules = sorted(frame["schedule"].unique())
    fig, axes = plt.subplots(len(schedules), 1, sharex=True,
                             figsize=(7.6, 1.7 * len(schedules) + 1.0),
                             squeeze=False)
    for axis, kind in zip(axes[:, 0], schedules):
        block = frame[frame["schedule"] == kind]
        for run_id, group in block.groupby("run_id"):
            group = group.sort_values("round")
            axis.plot(group["round"], group[ASR_COLUMN], linewidth=1.3,
                      color="tab:red", alpha=0.85)
            axis.plot(group["round"], group[MTA_COLUMN], linewidth=1.0,
                      color="tab:blue", alpha=0.6, linestyle="--")
            if "attack_active_this_round" in group.columns:
                active = group["attack_active_this_round"].astype(bool)
                rounds = group["round"].to_numpy()
                for index in np.flatnonzero(active.to_numpy()):
                    left = rounds[index]
                    width = (rounds[index + 1] - left
                             if index + 1 < len(rounds) else 1)
                    axis.axvspan(left, left + width, color="0.75", alpha=0.35,
                                 zorder=0, linewidth=0)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(kind, fontsize=8)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Communication round")
    title = "E1B-1  ASR (red) and MTA (blue dashed) per schedule"
    if dose_note:
        title += f"\nall panels at a single dose: {dose_note.split('（')[0]}"
    axes[0, 0].set_title(title)
    return _finish(fig, out_path,
                   "Grey bands mark rounds where poisoning was active "
                   "(sampled at the evaluation cadence, so band edges are "
                   "accurate only to +/- eval_every).\nDose is held fixed "
                   "across panels: comparing schedules at different doses "
                   "would confound timing with dose.")


def plot_e1b_2(persistence: pd.DataFrame, out_path) -> Path:
    """E1B-2：停攻后对齐的衰减曲线。"""
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    if persistence.empty:
        axis.text(0.5, 0.5,
                  "no run stopped attacking before the end of training",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return _finish(fig, out_path)
    for kind, block in persistence.groupby("schedule"):
        grouped = block.groupby("rounds_since_attack_stopped")["asr"]
        xs = sorted(grouped.groups)
        axis.plot(xs, [float(grouped.get_group(x).mean()) for x in xs],
                  marker="o", markersize=4, linewidth=1.5, label=str(kind))
    axis.set_xlabel("Rounds since the attack stopped")
    axis.set_ylabel(f"Backdoor ASR [{ASR_COLUMN}]")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title("E1B-2  Does the backdoor persist without fresh poisoning?")
    return _finish(fig, out_path,
                   "Curves are aligned at the last poisoned round, then "
                   "averaged over seeds.  'continuous' never stops and is "
                   "absent by construction.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 1 / 1B 的分析与图")
    parser.add_argument("--implantation-glob",
                        # run_fl 写出的文件名是
                        # exp_ij_implantation_<defense>_<mode>_a<alpha>_s<seed>_<run-tag>.csv，
                        # run-tag（e1… / e1b…）在 _s<seed>_ 之后，不在开头，所以
                        # 用 *_e1* 而非 e1* —— 后者匹配不到任何真实文件。
                        default="results/raw/exp_ij_implantation_*_e1*.csv")
    parser.add_argument("--out-dir", default="results/figs")
    parser.add_argument("--summary-prefix", default="results/exp1")
    parser.add_argument("--level", type=float, default=0.5,
                        help="判定'后门已植入'的 ASR 水平")
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--asr-column", default="asr_paper_all",
                        help="用哪一列当 ASR。默认 asr_paper_all —— 论文口径"
                             "（原始触发器 + 各客户端自己 test loader，全体均值，"
                             "对齐 main.py 的 Avg ASR）。可选 asr_paper_benign / "
                             "asr_paper_malicious，或旧口径 asr_personalized_targeted"
                             "（perturb 分解，仅良性）。缺列时自动回退到旧口径。")
    args = parser.parse_args(argv)

    frame = load_runs(args.implantation_glob)
    out_dir, prefix = Path(args.out_dir), Path(args.summary_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # 选定 ASR 列（论文口径优先，缺列回退旧口径）。所有下游函数读模块级
    # ASR_COLUMN，这里在调用它们之前改掉即可。
    global ASR_COLUMN
    ASR_COLUMN, asr_note = resolve_asr_column(frame, str(args.asr_column))
    if asr_note:
        print(f"  {asr_note}")
    print(f"[analysis_exp1] ASR 口径 = {ASR_COLUMN}")

    stage1 = frame[frame["schedule"] == "continuous"]
    stage1b = frame

    crossings = crossing_table(frame, args.level)
    crossings.to_csv(f"{prefix}_crossings.csv", index=False)
    verdict = threshold_verdict(crossings, frame)

    response = dose_response(stage1 if not stage1.empty else frame, args.tail)
    response.to_csv(f"{prefix}_dose_response.csv", index=False)
    onset = onset_analysis(frame)
    onset.to_csv(f"{prefix}_onset.csv", index=False)
    persistence = persistence_table(restrict_to_common_dose(frame)[0])
    if not persistence.empty:
        persistence.to_csv(f"{prefix}_persistence.csv", index=False)

    print(f"[analysis_exp1] {frame['run_id'].nunique()} 个 run，"
          f"{len(frame)} 个评估行")
    print(f"\n=== ASR >= {args.level} 的越过点 ===")
    print(crossings.to_string(index=False))

    print(f"\n=== 阈值判据（相关性证据，**不是因果**）===")
    for key in ("n_runs", "n_crossed", "n_never_crossed", "mta_at_cross_mean",
                "mta_at_cross_std", "round_at_cross_mean",
                "round_at_cross_std", "relative_spread_mta",
                "relative_spread_round"):
        if key in verdict:
            value = verdict[key]
            print(f"  {key:<24} "
                  + (f"{value:.4g}" if isinstance(value, float) else str(value)))
    print(f"  {verdict['verdict']}")

    if not response.empty:
        print(f"\n=== 剂量-响应（尾部 {args.tail} 个评估行）===")
        print(response.to_string(index=False))

    if onset["schedule"].nunique() > 1:
        print(f"\n=== 1B：起始 MTA vs 最终 ASR（识别策略）===")
        print(onset.to_string(index=False))
        print("  读法：在 n_active_eval_rounds 相当的前提下比较不同 "
              "mta_at_onset 的 final_asr。\n"
              "  投毒轮数差很多时不能直接比 —— 那就回到了剂量差异。")
    else:
        print(f"\n  ⚠️ 只有一种调度（{onset['schedule'].iloc[0]}），"
              f"1B 的识别策略用不上。跑 `python -m diag.run_exp1 --stage 1b`。")

    e1frame = stage1 if not stage1.empty else frame
    figures = [
        # E1 / E2 各出两版：按 N_m 分面、按 ρ 分面
        plot_e1_1(e1frame, out_dir / "exp1_E1_asr_vs_round_by_nm.png",
                  facet_by="bad_client_num", line_by="poison_rate"),
        plot_e1_1(e1frame, out_dir / "exp1_E1_asr_vs_round_by_rho.png",
                  facet_by="poison_rate", line_by="bad_client_num"),
        plot_e1_2(e1frame, out_dir / "exp1_E2_asr_vs_mta_by_nm.png", verdict,
                  facet_by="bad_client_num", line_by="poison_rate"),
        plot_e1_2(e1frame, out_dir / "exp1_E2_asr_vs_mta_by_rho.png", verdict,
                  facet_by="poison_rate", line_by="bad_client_num"),
        # E3（ASR vs clean loss）已移除：与 E2 冗余，且离群 loss 会把点挤成一竖线
    ]
    if not response.empty:
        figures.append(plot_e1_4(response, out_dir / "exp1_E4_dose_response.png"))
        figures.append(plot_e1_5(response, out_dir / "exp1_E5_dose_heatmap.png"))
    if stage1b["schedule"].nunique() > 1:
        figures.append(plot_e1b_1(stage1b, out_dir / "exp1b_B1_schedules.png"))
    figures.append(plot_e1b_2(persistence, out_dir / "exp1b_B2_persistence.png"))
    for path in figures:
        print(f"[analysis_exp1] 图 -> {path}")

    if frame["seed"].nunique() < 2:
        print(f"\n  ⚠️ 只有 1 个 seed，散点图上的任何'转折'都无法与随机波动区分。")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
