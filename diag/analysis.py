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


# ---------------------------------------------------------------------------
# 图中一律使用英文
#
# 出版用图不得出现中文。与其配一个 CJK 字体让中文"看起来正常"，不如让违规
# 直接失败 —— 字体兜底只会把问题藏起来（默认的 DejaVu Sans 不含 CJK 字形，
# 中文会渲染成方框 tofu，而且不报错）。
# ``_finish`` 在保存前会扫描整张图的文字，发现 CJK 立即抛异常。
# ---------------------------------------------------------------------------
_CJK_RANGES = ((0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
               (0xF900, 0xFAFF), (0xFF00, 0xFF65))


def contains_cjk(text: str) -> bool:
    """字符串里是否含中日韩字符（含全角标点）。"""
    return any(any(low <= ord(ch) <= high for low, high in _CJK_RANGES)
               for ch in str(text))


def assert_no_cjk_in_figure(fig) -> None:
    """扫描图中所有文字，发现 CJK 就抛 ``ValueError``。

    覆盖 figure 级文字（``suptitle`` / ``supxlabel`` / ``fig.text``）、
    以及每个 axes 的标题、坐标轴标签、刻度标签、图例与 ``ax.text`` 注释。
    """
    offenders = [text.get_text() for text in fig.texts
                 if contains_cjk(text.get_text())]
    for ax in fig.get_axes():
        candidates = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        candidates += [label.get_text() for label in ax.get_xticklabels()]
        candidates += [label.get_text() for label in ax.get_yticklabels()]
        candidates += [child.get_text() for child in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            candidates += [entry.get_text() for entry in legend.get_texts()]
        offenders += [text for text in candidates if contains_cjk(text)]
    if offenders:
        raise ValueError(
            "图中不允许出现中文，但发现: "
            + "; ".join(repr(item) for item in dict.fromkeys(offenders)))


__all__ = ["load_raw", "auc_score", "pearson_spearman", "mannwhitney",
           "contains_cjk", "assert_no_cjk_in_figure",
           "SIGNAL_DIRECTION", "signal_direction",
           "summarize", "summarize_c", "summarize_d",
           "plot_a1", "plot_a2", "plot_a3", "plot_a4",
           "plot_c1", "plot_c2", "plot_c3", "plot_d1", "plot_d2"]

# ---------------------------------------------------------------------------
# 检测分数的极性约定
#
# ``auc_score(score, labels=is_malicious)`` 的约定是**分数越大越像恶意**。
# 但并非每个信号天然满足这个方向，必须逐个显式声明，否则会静默地把 AUC 算反：
#
#   A_observable      +1  构造上就是"越大越可疑"（-mean - lambda*std 的最大值）
#   l2_to_median      +1  离中位数更新越远 = 越离群 = 越可疑
#   sign_consistency  -1  越大表示越贴近多数派方向 = 越像**良性**，故须取负
#
# 曾经的 bug：sign_consistency 未取负直接送进 auc_score，导致四个 alpha 的 AUC
# 全部落在 0.30-0.38（正确值是它们的 1 - x，即 0.62-0.70）。
# ---------------------------------------------------------------------------
SIGNAL_DIRECTION: Dict[str, int] = {
    "A_observable": +1,
    "l2_to_median": +1,
    "sign_consistency": -1,
}


def signal_direction(name: str) -> Tuple[int, bool]:
    """返回 ``(direction, is_declared)``。

    ``direction`` 为 +1/-1，乘到原始值上再送进 ``auc_score``。
    ``is_declared`` 为 False 表示该信号没有显式声明极性、按 +1 处理 ——
    调用方应当把这一点如实报告出来，而不是当作已知。

    ``A_lambda_*``（lambda 敏感性分析的各列）与 ``A_observable`` 同向。
    """
    if name in SIGNAL_DIRECTION:
        return SIGNAL_DIRECTION[name], True
    if name.startswith("A_lambda_"):
        return +1, True
    return +1, False

_GROUP_STYLE = {
    "real": ("Positive control: real target-class images", "tab:green", "o", "-"),
    "delta_only": ("Test: non-target + delta", "tab:red", "s", "-"),
    "delta_plus_xi": ("Test: non-target + delta + xi", "tab:orange", "^", "-"),
    "random_noise": ("Negative control: random noise, same budget", "tab:gray", "x", "--"),
    "xi_only": ("Reference: non-target + xi", "tab:blue", "v", ":"),
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
def overlap_chance_level(n_ref: int, n_other_per_class: int,
                         num_classes: int) -> float:
    """``knn_overlap`` 的**随机基准**：参照库里 ref 所占的比例。

    一个与两个分布都无关的查询点，其 k 近邻里 ref 的期望占比就是这个数。
    正式配置下 = 500 / (500 + 9×200) = **0.217**。

    ⚠️ 不报它，overlap 的绝对值就没法读。2026-09 审计发现实验 A 的三个扰动组
    （0.067–0.162）**全部低于**这个基准 —— 所以「δ = 经验基率」这个说法本身
    不准确：它们比基率还低（非目标类图片本来就更靠近 `other`）。
    """
    bank = int(n_ref) + int(n_other_per_class) * (int(num_classes) - 1)
    return float(n_ref) / bank if bank > 0 else float("nan")


def summarize(raw_csv_glob, metric: str = "overlap") -> pd.DataFrame:
    """实验 A 的汇总：按 ``(alpha, seed, group)`` 计算跨客户端变异系数。

    ::

        CV^g(alpha) = std_k(metric) / mean_k(metric)
        NG(alpha)   = CV^delta_plus_xi(alpha) - CV^real(alpha)

    用变异系数而不是标准差，是为了避免"均值高的组方差自然大"这一伪影。

    返回的 DataFrame 每行是一个 ``(alpha, seed, group)``，列包含
    ``cv`` / ``mean`` / ``std`` / ``n_clients``，并附带同 ``(alpha, seed)`` 下的
    ``ng_vs_real``（该组 CV 减去 real 组 CV）。

    ⚠️ **``metric`` 要两个都跑**（2026-09 审计）：``exp_a`` 每行同时算了
    ``overlap``（特征空间 kNN 邻域构成）与 ``nat_rate``（被判为目标类的比例），
    但 `results/summary/expA_summary.csv` 里**只有 overlap** —— nat_rate
    从来没被汇总过。而"干净模型对 δ 有没有响应"这个问题，**nat_rate 才是直接
    读数**，overlap 回答的是另一个问题。零机时可补：对同一批 raw CSV 再跑一次
    ``summarize(..., metric="nat_rate")``。
    """
    raw = load_raw(raw_csv_glob)
    for column in ("alpha", "seed", "group", metric):
        if column not in raw.columns:
            raise KeyError(
                f"原始 CSV 缺少列 '{column}'"
                + ("。exp_a 的行里应当同时有 overlap 与 nat_rate；"
                   "缺 nat_rate 说明这批 raw 是更早的版本跑的。"
                   if metric == "nat_rate" else ""))

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
    """实验 C 的汇总：每个 ``(alpha, seed, signal)`` 区分良性/恶意的 AUC。

    每个信号先乘以 ``signal_direction()`` 给出的极性再算 AUC —— 见本模块顶部
    ``SIGNAL_DIRECTION`` 的说明。列含义：

    ``auc``
        **修正极性后**的 AUC，这是应当被引用的数字。
    ``auc_raw``
        未修正的原始值，仅供追溯（对 direction=-1 的信号，``auc = 1 - auc_raw``）。
    ``direction`` / ``direction_declared``
        实际使用的极性，以及它是显式声明的还是按 +1 假定的。
    """
    raw = load_raw(raw_csv_glob)
    rows = []
    for (alpha, seed), block in raw.groupby(["alpha", "seed"]):
        labels = block["is_malicious"].astype(bool).to_numpy()
        for signal in signals:
            if signal not in block.columns:
                continue
            values = block[signal].to_numpy(dtype=float)
            direction, declared = signal_direction(signal)
            rows.append({"alpha": float(alpha), "seed": int(seed),
                         "signal": signal,
                         "auc": auc_score(direction * values, labels),
                         "auc_raw": auc_score(values, labels),
                         "direction": direction,
                         "direction_declared": declared,
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
    assert_no_cjk_in_figure(fig)   # 出版用图不得含中文，违规直接失败
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
    ax.set_xlabel("Dirichlet alpha (heterogeneity increases to the right)")
    ax.set_ylabel("Cross-client coefficient of variation, CV(overlap)")
    ax.set_title("A1: cross-client dispersion of delta vs real target vs noise")
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
               label="Decision threshold NG = 0.1")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha (heterogeneity increases to the right)")
    ax.set_ylabel("Naturalness gap NG(alpha)")
    ax.set_title("A2: naturalness gap vs heterogeneity")
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
    ax.set_ylabel(f"overlap (group: {group})")
    ax.set_title("A3: distribution of overlap across clients")
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
    ax.set_xlabel("Local target-class sample count of the client")
    ax.set_ylabel(f"overlap (group: {group}, measured on shared probe set)")
    ax.set_title(f"A4: confounder / mechanism control  Pearson r={r:.3f}, Spearman rho={rho:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_c1(raw, out_path) -> Path:
    """图 C1（主图）：散点 ``E_k`` vs ``A^(k)``，良性/恶意分色。"""
    raw = load_raw(raw)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for is_malicious, color, label in ((False, "tab:blue", "Benign clients"),
                                       (True, "tab:red", "Malicious clients")):
        sub = raw[raw["is_malicious"].astype(bool) == is_malicious]
        ax.scatter(sub["A_observable"], sub["E"], s=22, alpha=0.75,
                   color=color, label=label)
    r, rho = pearson_spearman(raw["A_observable"], raw["E"])
    ax.set_xlabel("Observable A^(k) (class prototypes + classifier head only)")
    ax.set_ylabel("Excess response E_k = ASR_k - NatRate_k (ground truth)")
    ax.set_title(f"C1: observable vs true backdoor strength  Pearson r={r:.3f}, Spearman rho={rho:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def plot_c2(summary: pd.DataFrame, out_path) -> Path:
    """图 C2（最终写进论文的那张）：三条信号的 AUC 随 alpha 的变化。"""
    style = {
        "A_observable": ("A^(k) (ours: margin matrix)", "tab:red", "o", "-"),
        "l2_to_median": ("Param L2 to median (Krum / FLAME family)", "tab:blue", "s", "--"),
        "sign_consistency": ("Gradient sign consistency (Invariant Aggregator)",
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
    ax.axhline(0.5, color="black", linewidth=0.8, label="Chance")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Dirichlet alpha (heterogeneity increases to the right)")
    ax.set_ylabel("AUC separating benign from malicious")
    ax.set_title("C2: detection power of three signals vs heterogeneity")
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
    ax.set_title("C3: sensitivity of A^(k) to lambda")
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
            return "Target class"
        if class_id in poisoned:
            return "Poisoned source classes"
        return "Other classes"

    raw = raw.copy()
    raw["bucket"] = raw["class_id"].map(_bucket)

    buckets = ["Target class", "Poisoned source classes", "Other classes"]
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
    ax.set_ylabel("v = log(J / benign median)")
    ax.set_title("D1: within-class scatter anomaly (blue = benign, red = malicious)")
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
    ax.axhline(0.5, color="black", linewidth=0.8, label="Chance")
    ax.axhline(0.7, color="crimson", linestyle=":", linewidth=0.8, label="Threshold 0.7")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("AUC")
    ax.set_title("D2: detection power of the within-class scatter anomaly")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)
