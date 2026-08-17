"""实验 I / J 的图。

五张图，编号沿用实验 J 的规格；实验 I 的 I-4 / I-5 / I-6 **就是** J-1 / J-3 / J-2
在同一坐标轴与配色下的同一套函数 —— 规格要求两个实验的图能直接并起来，
所以这里不做第二套实现。

======  =======================================  =========================
图      内容                                      需要的输入
======  =======================================  =========================
J-1     ASR vs 轮次（实线全局 / 虚线个性化）       植入 CSV（+ 检出 CSV 画攻击轮次）
J-2     恶意客户端在 l2_to_median 上的名次分布      检出 CSV 或 ranks CSV
J-3     影响力对比（恶意 vs 良性），分防御          检出 CSV
J-4     ASR 相对 ΔT 的衰减                        植入 CSV
J-5     FLAME 的余弦分布（恶意 vs 良性）           检出 CSV（需含 cos_to_others_*）
======  =======================================  =========================

⚠️ **图中一律使用英文**（出版要求）。``diag.analysis.assert_no_cjk_in_figure``
会在保存前扫描整张图，发现 CJK 直接抛异常。

# 一条读图的前提

J-4 在高恶意密度下**画不出东西**：10 个恶意 / 100 客户端 / 每轮选 10 时，
``rounds_since_last_malicious`` 实测只取到 {0,1,2,3}。要观察稀释机制必须先降
密度，见 ``diag/run_density_sweep.py`` 打印的期望间隔表。
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import assert_no_cjk_in_figure

__all__ = ["DEFENSE_STYLE", "GROUP_COLOR", "load_frames", "plot_j1", "plot_j2",
           "plot_j3", "plot_j4", "plot_j5", "main"]

# 配色在 I 与 J 之间共用，最终合并成跨防御对比图时才对得上（规格 §7.4 陷阱 14）
DEFENSE_STYLE: Dict[str, Tuple[str, str]] = {
    "fedavg": ("FedAvg (no defence)", "0.35"),
    "median": ("Median", "tab:green"),
    "multi_krum": ("Multi-Krum", "tab:orange"),
    "flame": ("FLAME", "tab:purple"),
    "invariant": ("Invariant Aggregator", "tab:blue"),
    "invariant_mask_only": ("Invariant: AND-mask only", "tab:cyan"),
    "invariant_trim_only": ("Invariant: trimmed-mean only", "tab:brown"),
}
GROUP_COLOR = {"malicious": "tab:red", "benign": "tab:blue"}


def _style(defense: str) -> Tuple[str, str]:
    return DEFENSE_STYLE.get(str(defense), (str(defense), "tab:gray"))


def _finish(fig, out_path) -> Path:
    """保存前扫描 CJK；用 bbox_inches='tight' 以容纳坐标区外的图例与脚注。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_cjk_in_figure(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_frames(pattern) -> Dict[str, pd.DataFrame]:
    """按 ``defense`` 列把匹配到的 CSV 拆成 ``{防御名: DataFrame}``。"""
    if isinstance(pattern, pd.DataFrame):
        frames = [pattern]
    else:
        paths = sorted(globlib.glob(str(pattern)))
        if not paths:
            raise FileNotFoundError(f"没有匹配的 CSV: {pattern}")
        frames = [pd.read_csv(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True)
    if "defense" not in merged.columns:
        raise KeyError("CSV 缺少 'defense' 列")
    return {str(name): block.sort_values("round").reset_index(drop=True)
            for name, block in merged.groupby("defense")}


# ---------------------------------------------------------------------------
# J-1 / I-4：ASR vs 轮次
# ---------------------------------------------------------------------------
def plot_j1(implantation_pattern, out_path, detection_pattern=None) -> Path:
    """实线 = 全局模型，虚线 = 个性化模型；底部竖线标注有恶意客户端的轮次。

    ``detection_pattern`` 给了就用它画攻击轮次（逐轮，准确）；不给则退回植入
    CSV 的采样点 —— 那只有每 10 轮一个，**会漏掉大部分攻击轮次**，图注会说明。
    """
    by_defense = load_frames(implantation_pattern)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for name, block in by_defense.items():
        label, color = _style(name)
        if "asr_global_targeted" in block:
            ax.plot(block["round"], block["asr_global_targeted"], color=color,
                    linestyle="-", linewidth=1.6, label=f"{label} — global")
        if "asr_personalized_targeted" in block:
            ax.plot(block["round"], block["asr_personalized_targeted"],
                    color=color, linestyle="--", linewidth=1.6,
                    label=f"{label} — personalized")

    # 攻击轮次的地毯图
    rug_note = ""
    attack_rounds: Sequence[int] = ()
    if detection_pattern is not None:
        detection = pd.concat(
            [pd.read_csv(p) for p in sorted(globlib.glob(str(detection_pattern)))],
            ignore_index=True)
        attack_rounds = sorted(
            detection.loc[detection["n_malicious_in_round"] > 0, "round"].unique())
    else:
        sample = next(iter(by_defense.values()))
        if "n_malicious_this_round" in sample:
            attack_rounds = sorted(
                sample.loc[sample["n_malicious_this_round"] > 0, "round"].unique())
            rug_note = ("rug marks only the evaluated rounds, so most attack "
                        "rounds are missing; pass a detection CSV for the exact set. ")
    if len(attack_rounds):
        ax.vlines(list(attack_rounds), -0.045, -0.012, color="tab:red",
                  linewidth=0.5, alpha=0.55)
        ax.text(0.005, -0.058, "rounds with >=1 malicious client",
                transform=ax.get_yaxis_transform(), fontsize=7, color="tab:red")

    ax.set_xlabel("Round")
    ax.set_ylabel("ASR_targeted")
    ax.set_ylim(-0.08, 1.02)
    ax.set_title("J-1 / I-4: backdoor implantation over training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.5, -0.02,
             rug_note + "Solid = global model (BN borrowed from a benign client; "
             "FedBN never aggregates BN, so the raw global model sits at chance). "
             "Dashed = personalized models, the paper's ASR definition.",
             fontsize=7, style="italic", color="dimgray", ha="center", va="top",
             wrap=True)
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# J-2 / I-6：名次分布
# ---------------------------------------------------------------------------
def plot_j2(detection_pattern, out_path, signal: str = "l2") -> Path:
    """恶意客户端按信号降序的名次直方图。

    只用**恰好 1 个恶意客户端**的轮次：此时 ``malicious_rank_*_mean`` 就是那个
    客户端的确切名次；有 2 个以上时它是均值，进不了直方图。被排除的轮数会写进
    图注，不藏起来。

    完美判别力 -> 全部集中在名次 1；无判别力 -> 均匀分布（图中画出该水平线）。
    """
    column = f"malicious_rank_{signal}_mean"
    by_defense = load_frames(detection_pattern)
    usable = {k: v for k, v in by_defense.items() if column in v.columns}
    if not usable:
        raise KeyError(f"检出 CSV 里没有 '{column}'")

    fig, axes = plt.subplots(1, len(usable), figsize=(4.2 * len(usable), 4.2),
                            sharey=True, squeeze=False)
    for ax, (name, block) in zip(axes[0], usable.items()):
        label, color = _style(name)
        single = block[block["n_malicious_in_round"] == 1]
        ranks = single[column].dropna().astype(int)
        n_clients = int(block["n_selected"].mode().iloc[0])
        counts = np.bincount(ranks, minlength=n_clients + 1)[1:n_clients + 1]
        share = counts / max(counts.sum(), 1)
        ax.bar(np.arange(1, n_clients + 1), share, color=color, alpha=0.85)
        ax.axhline(1.0 / n_clients, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{label}\n{len(single)} single-malicious rounds", fontsize=9)
        ax.set_xlabel(f"Rank by {signal}_to_median (1 = largest)")
        ax.set_xticks(np.arange(1, n_clients + 1))
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Share of malicious clients")

    # y>1 配合 bbox_inches="tight"：否则总标题会压住子图标题
    fig.suptitle("J-2 / I-6: where the malicious client ranks each round",
                 fontsize=11, y=1.06)
    fig.text(0.5, -0.03,
             "Dashed line = uniform, i.e. no discriminative power. Rounds with "
             "two or more malicious clients are excluded because the recorded "
             "rank is their mean, not an individual rank.",
             fontsize=7, style="italic", color="dimgray", ha="center", va="top")
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# J-3 / I-5：影响力对比
# ---------------------------------------------------------------------------
def plot_j3(detection_pattern, out_path) -> Path:
    """恶意 vs 良性的平均影响力 I^(k)，分防御。比值 ≈ 1 表示这一环没起作用。"""
    by_defense = load_frames(detection_pattern)
    names, malicious, benign, errors_m, errors_b = [], [], [], [], []
    for name, block in by_defense.items():
        used = block[block["n_malicious_in_round"] > 0]
        if used.empty:
            continue
        names.append(name)
        malicious.append(float(used["influence_malicious_mean"].mean()))
        benign.append(float(used["influence_benign_mean"].mean()))
        errors_m.append(float(used["influence_malicious_mean"].std(ddof=1)))
        errors_b.append(float(used["influence_benign_mean"].std(ddof=1)))

    def _clamped(values: List[float], spreads: List[float]) -> np.ndarray:
        """I(k) ∈ [0, 1]，误差棒不得越界。

        对二元决策的防御（FLAME / Multi-Krum）I(k) 只取 0 或 1，跨轮标准差
        天然接近 0.5 —— 画成对称误差棒会伸到负数区，那是坐标轴上不存在的值。
        这里按 [0,1] 截断，形状本身也在传达"这些防御是全进或全出"。
        """
        values_arr = np.asarray(values, dtype=float)
        spreads_arr = np.nan_to_num(np.asarray(spreads, dtype=float))
        return np.vstack([np.minimum(spreads_arr, values_arr),
                          np.minimum(spreads_arr, 1.0 - values_arr)])

    positions = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(2.1 * max(len(names), 3) + 3, 5.0))
    err_m = _clamped(malicious, errors_m)
    err_b = _clamped(benign, errors_b)
    ax.bar(positions - width / 2, malicious, width, yerr=err_m, capsize=3,
           color=GROUP_COLOR["malicious"], alpha=0.85, label="malicious clients")
    ax.bar(positions + width / 2, benign, width, yerr=err_b, capsize=3,
           color=GROUP_COLOR["benign"], alpha=0.85, label="benign clients")

    for index, (m, b) in enumerate(zip(malicious, benign)):
        ratio = m / b if b else np.nan
        top = max(m + err_m[1][index], b + err_b[1][index])
        ax.text(index, min(top + 0.05, 1.04), f"ratio {ratio:.2f}", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([_style(n)[0] for n in names], fontsize=9)
    ax.set_ylabel("Influence  I(k)  on the aggregate")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("J-3 / I-5: how much each group still influences aggregation")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.5, -0.04,
             "A ratio near 1 means the rule does not separate the two groups. "
             "Error bars are the between-round standard deviation, clipped to "
             "[0, 1]; for FLAME and Multi-Krum I(k) is binary, so a wide bar "
             "means the client is either fully kept or fully dropped. Rounds "
             "with no malicious client are excluded.",
             fontsize=7, style="italic", color="dimgray", ha="center", va="top",
             wrap=True)
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# J-4：ΔT 衰减
# ---------------------------------------------------------------------------
def plot_j4(implantation_pattern, out_path, min_round: Optional[int] = None
            ) -> Path:
    """ASR 相对"距上次有恶意客户端的轮数"的变化。

    衰减速率是防御有效性的**独立**度量 —— 好的防御应加快衰减，而不只是压低峰值。

    # 两个必须处理的混杂

    1. **ΔT 与训练进度混在一起。** ASR 在 500 轮里从 0.1 涨到 0.99，而 ΔT 与
       轮次几乎无关。把整段训练按 ΔT 汇总，误差棒里装的主要是"训练到第几轮"，
       不是 ΔT 的效应。因此默认**只用后半段**（ASR 已进入平台期），
       ``min_round`` 可显式指定；用了哪个窗口会写在图注里。
    2. **高恶意密度下 ΔT 没有取值范围。** 10 恶意 / 100 客户端 / 每轮选 10 时
       实测只到 3 轮。此时图上直接写出这一点，而不是让读者从 n= 标注里去猜。
    """
    by_defense = load_frames(implantation_pattern)
    if min_round is None:
        all_rounds = pd.concat([b["round"] for b in by_defense.values()])
        min_round = int(all_rounds.max() // 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_gap, total_points = 0, 0
    for name, block in by_defense.items():
        label, color = _style(name)
        used = block[(block["rounds_since_last_malicious"] >= 0)
                     & (block["round"] >= min_round)]
        if used.empty:
            continue
        grouped = used.groupby("rounds_since_last_malicious")[
            "asr_personalized_targeted"].agg(["mean", "std", "count"])
        max_gap = max(max_gap, int(grouped.index.max()))
        total_points += len(grouped)
        # ASR ∈ [0,1]，误差棒不越界
        spread = grouped["std"].fillna(0.0).to_numpy()
        mean = grouped["mean"].to_numpy()
        yerr = np.vstack([np.minimum(spread, mean),
                          np.minimum(spread, 1.0 - mean)])
        ax.errorbar(grouped.index, mean, yerr=yerr, marker="o", capsize=3,
                    color=color, label=label)
        for gap, row in grouped.iterrows():
            ax.annotate(f"n={int(row['count'])}", (gap, row["mean"]),
                        textcoords="offset points", xytext=(0, 9),
                        fontsize=6, ha="center", color=color)

    ax.set_xlabel("rounds since a malicious client last participated")
    ax.set_ylabel("ASR_targeted (personalized)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("J-4: does the backdoor decay while the attacker is absent?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    note = (f"Restricted to rounds >= {min_round} so that the gap effect is not "
            f"confounded with training progress — ASR climbs from ~0.1 to ~0.99 "
            f"across the run, while the gap is uncorrelated with the round. "
            f"Decay rate is an independent measure of a defence: a good one "
            f"should speed up decay, not merely lower the peak.")
    if max_gap <= 3:
        note = (f"WARNING: the largest observed gap is {max_gap} rounds, so this "
                f"panel has almost no dynamic range and no decay can be read "
                f"from it. With 10 malicious clients out of 100 and 10 sampled "
                f"per round the expected gap is 0.5 rounds; at 1 malicious "
                f"client it is 9. Lower the malicious count first — see "
                f"diag/run_density_sweep.py. ") + note
    fig.text(0.5, -0.03, note, fontsize=7, style="italic", color="dimgray",
             ha="center", va="top", wrap=True)
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# J-5：FLAME 的余弦分布
# ---------------------------------------------------------------------------
def plot_j5(detection_pattern, out_path) -> Optional[Path]:
    """FLAME 聚类环节的直接证据：恶意与良性的平均余弦相似度是否可分。

    重叠 = 聚类那一环没有信号可用。同时把**裁剪**环节的比率并排画出来 ——
    两环节的信号强弱正是 §4.2 要定位的东西。
    """
    by_defense = load_frames(detection_pattern)
    block = by_defense.get("flame")
    if block is None or "cos_to_others_malicious_mean" not in block.columns:
        return None
    used = block[block["n_malicious_in_round"] > 0]
    if used.empty:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))

    bins = np.linspace(
        float(np.nanmin(used[["cos_to_others_malicious_mean",
                              "cos_to_others_benign_mean"]].to_numpy())),
        float(np.nanmax(used[["cos_to_others_malicious_mean",
                              "cos_to_others_benign_mean"]].to_numpy())), 40)
    for column, group in (("cos_to_others_malicious_mean", "malicious"),
                          ("cos_to_others_benign_mean", "benign")):
        left.hist(used[column].dropna(), bins=bins, alpha=0.55,
                  color=GROUP_COLOR[group], label=group, density=True)
    left.set_xlabel("mean cosine similarity to the other updates")
    left.set_ylabel("density")
    left.set_title("Clustering stage: is there any signal?")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    if "clip_rate_malicious" in used.columns:
        rates = [float(used["clip_rate_malicious"].mean()),
                 float(used["clip_rate_benign"].mean())]
        right.bar(["malicious", "benign"], rates,
                  color=[GROUP_COLOR["malicious"], GROUP_COLOR["benign"]],
                  alpha=0.85)
        for index, value in enumerate(rates):
            right.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
        right.set_ylim(0, 1.05)
        right.set_ylabel("fraction of updates clipped")
        right.set_title("Clipping stage: is there any signal?")
        right.grid(alpha=0.3, axis="y")

    fig.suptitle("J-5: FLAME stage by stage", fontsize=11, y=1.02)
    fig.text(0.5, -0.04,
             "The stage that carries the signal (clipping) only rescales; the "
             "stage that excludes clients (clustering) is the weaker one.",
             fontsize=7, style="italic", color="dimgray", ha="center", va="top")
    return _finish(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 I / J 的五张图")
    parser.add_argument("--detection-glob",
                        default="results/exp_*_detection.csv")
    parser.add_argument("--implantation-glob",
                        default="results/raw/exp_ij_implantation_*.csv")
    parser.add_argument("--out-dir", default="results/figs")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    produced: List[Path] = []
    attempts = [
        ("J-1", lambda: plot_j1(args.implantation_glob, out / "exp_ij_J1.png",
                                detection_pattern=args.detection_glob)),
        ("J-2", lambda: plot_j2(args.detection_glob, out / "exp_ij_J2.png")),
        ("J-3", lambda: plot_j3(args.detection_glob, out / "exp_ij_J3.png")),
        ("J-4", lambda: plot_j4(args.implantation_glob, out / "exp_ij_J4.png")),
        ("J-5", lambda: plot_j5(args.detection_glob, out / "exp_ij_J5.png")),
    ]
    for name, func in attempts:
        try:
            path = func()
        except (FileNotFoundError, KeyError) as exc:
            print(f"[analysis_ij] {name} 跳过：{exc}")
            continue
        if path is None:
            print(f"[analysis_ij] {name} 跳过：所需数据不在这批 CSV 里")
            continue
        produced.append(path)
        print(f"[analysis_ij] {name} -> {path}")
    return 0 if produced else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
