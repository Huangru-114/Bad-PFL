"""汇总与绘图。

本模块只做**聚合与可视化**，不做测量。所有输入来自 ``results/raw/*.csv``。

⚠️ 本模块不实现 t-SNE。所有定量结论必须来自 ``knn_overlap`` 这类可量化指标；
t-SNE 只适合展示，不能用于任何定量判断。若将来为了论文配图需要 t-SNE，
必须在图注中明确标注"该图不参与任何指标计算"。
"""

from __future__ import annotations

import glob as globlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # 无显示环境；必须在 pyplot 之前设置
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

__all__ = ["load_raw", "auc_score", "pearson_spearman", "mannwhitney",
           "summarize", "summarize_c", "summarize_d",
           "plot_a1", "plot_a2", "plot_a3", "plot_a4",
           "plot_c1", "plot_c2", "plot_c3", "plot_d1", "plot_d2"]

_GROUP_STYLE = {
    "real": ("正对照: 真实目标类图片", "tab:green", "o", "-"),
    "delta_only": ("待测: 非目标类 + δ", "tab:red", "s", "-"),
    "delta_plus_xi": ("待测: 非目标类 + δ + ξ", "tab:orange", "^", "-"),
    "random_noise": ("负对照: 同预算随机噪声", "tab:gray", "x", "--"),
    "xi_only": ("参考: 非目标类 + ξ", "tab:blue", "v", ":"),
}


# ---------------------------------------------------------------------------
# 载入与统计工具
# ---------------------------------------------------------------------------
def load_raw(raw_csv_glob) -> pd.DataFrame:
    """读入并拼接所有匹配的原始 CSV。

    ``raw_csv_glob`` 可以是 glob 字符串、路径，或 DataFrame（直接返回）。
    """
    if isinstance(raw_csv_glob, pd.DataFrame):
        return raw_csv_glob.copy()
    paths = sorted(globlib.glob(str(raw_csv_glob)))
    if not paths:
        raise FileNotFoundError(f"没有匹配的 CSV: {raw_csv_glob}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def auc_score(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """二分类 AUC（Mann-Whitney U 的秩形式，正确处理并列）。

    nan 行为：nan 分数的样本被剔除；剔除后任一类为空则返回 nan。
    不依赖 scikit-learn。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # 并列取平均秩
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pearson_spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """返回 ``(pearson_r, spearman_rho)``，成对剔除 nan。不足 3 个点返回 nan。"""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan"), float("nan")
    return (float(stats.pearsonr(x[keep], y[keep])[0]),
            float(stats.spearmanr(x[keep], y[keep])[0]))


def mannwhitney(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Mann-Whitney U 检验，返回 ``(U, p)``（双侧）。任一组为空返回 nan。"""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _cv(series: pd.Series) -> float:
    """变异系数 std/mean，nan-safe；均值退化时返回 nan 而不是 inf。"""
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    mean = values.mean()
    if abs(mean) < 1e-12:
        return float("nan")
    return float(values.std(ddof=1) / abs(mean))


def _alpha_order(frame: pd.DataFrame) -> List[float]:
    """异构度从弱到强（alpha 从大到小）排列，与图的横轴语义一致。"""
    return sorted(frame["alpha"].unique(), reverse=True)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(raw_csv_glob, metric: str = "overlap") -> pd.DataFrame:
    """实验 A 的汇总：按 ``(alpha, seed, group)`` 计算跨客户端变异系数。

    ::

        CV^g(alpha) = std_k(metric) / mean_k(metric)
        NG(alpha)   = CV^delta_plus_xi(alpha) - CV^real(alpha)

    用变异系数而不是标准差，是为了避免"均值高的组方差自然大"这一伪影。

    返回的 DataFrame 每行是一个 ``(alpha, seed, group)``，列包含
    ``cv`` / ``mean`` / ``std`` / ``n_clients``，并附带同 ``(alpha, seed)`` 下的
    ``ng_vs_real``（该组 CV 减去 real 组 CV）。
    """
    raw = load_raw(raw_csv_glob)
    for column in ("alpha", "seed", "group", metric):
        if column not in raw.columns:
            raise KeyError(f"原始 CSV 缺少列 '{column}'")

    rows = []
    for (alpha, seed, group), block in raw.groupby(["alpha", "seed", "group"]):
        values = block[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "alpha": float(alpha), "seed": int(seed), "group": str(group),
            "metric": metric,
            "cv": _cv(block[metric]),
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "n_clients": int(len(block)),
            "n_valid": int(len(finite)),
        })
    summary = pd.DataFrame(rows)

    real_cv = (summary[summary["group"] == "real"]
               .set_index(["alpha", "seed"])["cv"].to_dict())
    summary["ng_vs_real"] = [
        row["cv"] - real_cv.get((row["alpha"], row["seed"]), float("nan"))
        for _, row in summary.iterrows()]
    return summary.sort_values(["alpha", "seed", "group"]).reset_index(drop=True)


def summarize_c(raw_csv_glob,
                signals: Sequence[str] = ("A_observable", "l2_to_median",
                                          "sign_consistency")) -> pd.DataFrame:
    """实验 C 的汇总：每个 ``(alpha, seed, signal)`` 区分良性/恶意的 AUC。"""
    raw = load_raw(raw_csv_glob)
    rows = []
    for (alpha, seed), block in raw.groupby(["alpha", "seed"]):
        labels = block["is_malicious"].astype(bool).to_numpy()
        for signal in signals:
            if signal not in block.columns:
                continue
            rows.append({"alpha": float(alpha), "seed": int(seed),
                         "signal": signal,
                         "auc": auc_score(block[signal].to_numpy(dtype=float), labels),
                         "n_clients": int(len(block)),
                         "n_malicious": int(labels.sum())})
    return pd.DataFrame(rows)


def summarize_d(raw_csv_glob) -> pd.DataFrame:
    """实验 D 的汇总：用 ``max_c v_c^(k)`` 区分良性/恶意的 AUC。"""
    raw = load_raw(raw_csv_glob)
    rows = []
    for (alpha, seed), block in raw.groupby(["alpha", "seed"]):
        per_client = (block.groupby(["client_id", "is_malicious", "poison_ratio"])["v"]
                      .max().reset_index())
        rows.append({
            "alpha": float(alpha), "seed": int(seed),
            "auc_max_v": auc_score(per_client["v"].to_numpy(dtype=float),
                                   per_client["is_malicious"].astype(bool).to_numpy()),
            "poison_ratio": float(per_client["poison_ratio"].max()),
            "n_clients": int(len(per_client)),
            "n_malicious": int(per_client["is_malicious"].astype(bool).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _finish(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_a1(summary: pd.DataFrame, out_path) -> Path:
    """图 A1（主图）：横轴 alpha，纵轴 CV，三条线，误差棒来自不同 seed。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = _alpha_order(summary)
    positions = np.arange(len(alphas))
    for group in summary["group"].unique():
        label, color, marker, linestyle = _GROUP_STYLE.get(
            group, (group, None, "o", "-"))
        means, errors = [], []
        for alpha in alphas:
            block = summary[(summary["group"] == group) & (summary["alpha"] == alpha)]
            values = block["cv"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.errorbar(positions, means, yerr=errors, label=label, color=color,
                    marker=marker, linestyle=linestyle, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha（越往右异构度越高）")
    ax.set_ylabel("跨客户端变异系数 CV(overlap)")
    ax.set_title("图 A1: δ / 真实目标类 / 随机噪声 的跨客户端离散度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_a2(summary: pd.DataFrame, out_path,
            group: str = "delta_plus_xi") -> Path:
    """图 A2：横轴 alpha，纵轴自然度差距 ``NG(alpha) = CV^δ - CV^real``。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = _alpha_order(summary)
    positions = np.arange(len(alphas))
    for candidate in ("delta_only", "delta_plus_xi"):
        block_all = summary[summary["group"] == candidate]
        if block_all.empty:
            continue
        means, errors = [], []
        for alpha in alphas:
            values = block_all[block_all["alpha"] == alpha]["ng_vs_real"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        style = _GROUP_STYLE.get(candidate, (candidate, None, "o", "-"))
        ax.errorbar(positions, means, yerr=errors, label=f"NG ({style[0]})",
                    color=style[1], marker=style[2], capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.1, color="crimson", linestyle=":", linewidth=0.8,
               label="判据阈值 NG = 0.1")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha（越往右异构度越高）")
    ax.set_ylabel("自然度差距 NG(alpha)")
    ax.set_title("图 A2: 自然度差距随异构度的变化")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_a3(raw: pd.DataFrame, out_path, group: str = "delta_plus_xi") -> Path:
    """图 A3：小提琴图，展示 overlap 在客户端间的**完整分布**（而不只是方差）。"""
    raw = load_raw(raw)
    block = raw[raw["group"] == group]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = _alpha_order(block) if len(block) else []
    datasets, positions, labels = [], [], []
    for index, alpha in enumerate(alphas):
        values = block[block["alpha"] == alpha]["overlap"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        datasets.append(values)
        positions.append(index)
        labels.append(str(alpha))
    if datasets:
        ax.violinplot(datasets, positions=positions, showmeans=True, widths=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel(f"overlap（组: {group}）")
    ax.set_title("图 A3: overlap 在客户端间的分布")
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_a4(raw, out_path, group: str = "delta_plus_xi") -> Path:
    """图 A4（机制证据）：``overlap`` vs **本地目标类样本数**。

    这张图从"混杂控制"升级成了主要的机制证据：一个从未见过目标类的客户端模型，
    会不会仍然觉得 δ 像目标类？
    - 若会（无相关）→ δ 的语义是数据集/架构层面的普遍属性，H1 存疑。
    - 若不会（强相关）→ δ 的语义依赖本地见过多少目标类，H1 得到支持并有机制解释。
    """
    raw = load_raw(raw)
    block = raw[raw["group"] == group]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in _alpha_order(block):
        sub = block[block["alpha"] == alpha]
        ax.scatter(sub["n_target_samples_local"], sub["overlap"], s=18,
                   alpha=0.7, label=f"alpha={alpha}")
    r, rho = pearson_spearman(block["n_target_samples_local"], block["overlap"])
    ax.set_xlabel("该客户端【本地】目标类样本数")
    ax.set_ylabel(f"overlap（组: {group}，在共享探针集上测量）")
    ax.set_title(f"图 A4: 混杂/机制控制  Pearson r={r:.3f}, Spearman ρ={rho:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_c1(raw, out_path) -> Path:
    """图 C1（主图）：散点 ``E_k`` vs ``A^(k)``，良性/恶意分色。"""
    raw = load_raw(raw)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for is_malicious, color, label in ((False, "tab:blue", "良性客户端"),
                                       (True, "tab:red", "恶意客户端")):
        sub = raw[raw["is_malicious"].astype(bool) == is_malicious]
        ax.scatter(sub["A_observable"], sub["E"], s=22, alpha=0.75,
                   color=color, label=label)
    r, rho = pearson_spearman(raw["A_observable"], raw["E"])
    ax.set_xlabel("可观测量 A^(k)（仅用类原型 + 分类头）")
    ax.set_ylabel("超额响应 E_k = ASR_k - NatRate_k（ground truth）")
    ax.set_title(f"图 C1: 可观测量与真实后门强度  Pearson r={r:.3f}, Spearman ρ={rho:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_c2(summary: pd.DataFrame, out_path) -> Path:
    """图 C2（最终写进论文的那张）：三条信号的 AUC 随 alpha 的变化。"""
    style = {
        "A_observable": ("A^(k)（本方法：边距矩阵）", "tab:red", "o", "-"),
        "l2_to_median": ("参数 L2 距中位数（Krum / FLAME 类）", "tab:blue", "s", "--"),
        "sign_consistency": ("梯度符号一致性（Invariant Aggregator 类）",
                             "tab:green", "^", ":"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = _alpha_order(summary)
    positions = np.arange(len(alphas))
    for signal in summary["signal"].unique():
        label, color, marker, linestyle = style.get(signal, (signal, None, "o", "-"))
        means, errors = [], []
        for alpha in alphas:
            values = summary[(summary["signal"] == signal)
                             & (summary["alpha"] == alpha)]["auc"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.errorbar(positions, means, yerr=errors, label=label, color=color,
                    marker=marker, linestyle=linestyle, capsize=3)
    ax.axhline(0.5, color="black", linewidth=0.8, label="随机猜测")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Dirichlet alpha（越往右异构度越高）")
    ax.set_ylabel("区分良性/恶意的 AUC")
    ax.set_title("图 C2: 三种信号在不同异构度下的检测能力")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_c3(raw, out_path) -> Path:
    """图 C3：``observable_score`` 对 lambda 的敏感性。"""
    raw = load_raw(raw)
    columns = sorted([c for c in raw.columns if c.startswith("A_lambda_")],
                     key=lambda name: float(name.rsplit("_", 1)[1]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if columns:
        labels = raw["is_malicious"].astype(bool).to_numpy()
        lambdas = [float(name.rsplit("_", 1)[1]) for name in columns]
        for alpha in _alpha_order(raw):
            mask = raw["alpha"] == alpha
            aucs = [auc_score(raw.loc[mask, name].to_numpy(dtype=float),
                              labels[mask.to_numpy()]) for name in columns]
            ax.plot(lambdas, aucs, marker="o", label=f"alpha={alpha}")
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.set_xlabel("lambda_std")
    ax.set_ylabel("AUC")
    ax.set_title("图 C3: A^(k) 对 lambda 的敏感性")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_d1(raw, out_path, poisoned_source_classes: Optional[Iterable[int]] = None) -> Path:
    """图 D1：箱线图，``v`` 按 (目标类 / 被投毒源类 / 其余类) × (良性 / 恶意) 分组。

    ``poisoned_source_classes`` 未提供时，把所有非目标类都归为"其余类" ——
    因为 Bad-PFL 的投毒是对**全部**非目标类样本施加 ξ，不存在特定源类。
    需要区分时由调用方显式传入。
    """
    raw = load_raw(raw)
    target_classes = set(raw.loc[raw["is_target_class"].astype(bool), "class_id"].unique()) \
        if "is_target_class" in raw.columns else set()
    poisoned = set(poisoned_source_classes or [])

    def _bucket(class_id: int) -> str:
        if class_id in target_classes:
            return "目标类"
        if class_id in poisoned:
            return "被投毒源类"
        return "其余类"

    raw = raw.copy()
    raw["bucket"] = raw["class_id"].map(_bucket)

    buckets = ["目标类", "被投毒源类", "其余类"]
    buckets = [b for b in buckets if (raw["bucket"] == b).any()]
    datasets, positions, tick_positions, tick_labels, colors = [], [], [], [], []
    for index, bucket in enumerate(buckets):
        for offset, (is_malicious, color) in enumerate(
                ((False, "tab:blue"), (True, "tab:red"))):
            values = raw[(raw["bucket"] == bucket)
                         & (raw["is_malicious"].astype(bool) == is_malicious)]["v"]
            values = values.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            datasets.append(values)
            positions.append(index * 2.5 + offset * 0.8)
            colors.append(color)
        tick_positions.append(index * 2.5 + 0.4)
        tick_labels.append(bucket)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if datasets:
        artists = ax.boxplot(datasets, positions=positions, widths=0.65,
                             patch_artist=True, showfliers=False)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("v = log(J / 良性中位数)")
    ax.set_title("图 D1: 类内散度异常（蓝=良性，红=恶意）")
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, out_path)


def plot_d2(summary: pd.DataFrame, out_path) -> Path:
    """图 D2：横轴 alpha，纵轴用 ``max_c v_c^(k)`` 区分良性/恶意的 AUC。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = _alpha_order(summary)
    positions = np.arange(len(alphas))
    means, errors = [], []
    for alpha in alphas:
        values = summary[summary["alpha"] == alpha]["auc_max_v"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        means.append(values.mean() if len(values) else np.nan)
        errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
    ax.errorbar(positions, means, yerr=errors, marker="o", color="tab:purple",
                capsize=3, label="max_c v_c^(k)")
    ax.axhline(0.5, color="black", linewidth=0.8, label="随机猜测")
    ax.axhline(0.7, color="crimson", linestyle=":", linewidth=0.8, label="判据阈值 0.7")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("AUC")
    ax.set_title("图 D2: 类内散度异常的检测能力")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)
