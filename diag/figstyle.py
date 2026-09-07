"""出版级图样式的**共用件**（回应导师意见 #3）。

导师逐条点的问题都在「谁来放标签、放在哪」这一层，而不在数据上：

  - y 轴标签**与刻度数字重叠**：`fig.text` 写在固定坐标上，而 `tight_layout`
    根本不知道它存在、不会给它留位置。
  - `fig.legend(loc="center right")` **压在最右侧 panel 上**。
  - y 轴直接写内部列名 `asr_paper_benign` —— 那是代码里的名字，不是人话。
  - 4 个 panel 排成 3+1，第二行大片空白；**中间 panel 没有 x 刻度标签**
    （`sharex=True` 只给最下面一行标签，而它下面那格是空的）。

**为什么单独一个模块**：这里的布局算术与标签映射是纯 Python，
`analysis_exp1.py` 一 import matplotlib 本机就跑不了了。抽出来之后 L1 秒级覆盖
（与 tf-dpfl 把纯算术抽进 `server/participation.py` 是同一个理由）。
需要 matplotlib 的函数**在函数体里**才 import。

⚠️ 铁律 #4：图里一律英文。本模块产出的每一个标签都是英文。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["facet_layout", "pretty_label", "asr_axis_label",
           "apply_publication_style", "fig_axis_labels", "finish_facets",
           "outside_legend", "PUBLICATION_RCPARAMS"]


# ---------------------------------------------------------------------------
# 分面布局
# ---------------------------------------------------------------------------
# 显式表而不是「ncol = min(3, n)」：后者在 n=4 时给 2x3、空两格，正是报告里
# E1-1b / E1-2 / E1-6 第二行大片空白的来源。这里每个 n 单独定，可预期、可测试。
# 取舍：优先不留空格，其次偏横向（论文版面是横的，竖长条会被 PDF 从中间截断
# —— E1B-1 就是这么断在 p.6/p.7 之间的）。
_LAYOUT: Dict[int, Tuple[int, int]] = {
    1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2),
    5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3),
}


def facet_layout(n: int) -> Tuple[int, int]:
    """``n`` 个分面的 (nrow, ncol)。n>9 时按 3 列铺开。"""
    if n <= 0:
        return 1, 1
    if n in _LAYOUT:
        return _LAYOUT[n]
    ncol = 3
    return (n + ncol - 1) // ncol, ncol


# ---------------------------------------------------------------------------
# 标签：内部列名 -> 人话
# ---------------------------------------------------------------------------
# 轴上出现内部列名（asr_paper_benign）是最容易被审稿人抓的一类问题：
# 读者不知道 benign 指的是「在良性客户端的个性化模型上测」，也不知道
# filtered 指「已排除真标签就是目标类的样本」。这里一次性说清楚。
_PRETTY: Dict[str, str] = {
    # ASR —— 三档 scope x 两种口径
    "asr_paper_benign":
        "Backdoor ASR (benign clients, target class included)",
    "asr_paper_malicious":
        "Backdoor ASR (malicious clients, target class included)",
    "asr_paper_all":
        "Backdoor ASR (all clients, target class included)",
    "asr_paper_filtered_benign":
        "Backdoor ASR (benign clients, target class excluded)",
    "asr_paper_filtered_malicious":
        "Backdoor ASR (malicious clients, target class excluded)",
    "asr_paper_filtered_all":
        "Backdoor ASR (all clients, target class excluded)",
    "asr_paper_frozen_benign":
        "Backdoor ASR, frozen trigger (benign clients)",
    "asr_paper_frozen_malicious":
        "Backdoor ASR, frozen trigger (malicious clients)",
    "asr_paper_frozen_all":
        "Backdoor ASR, frozen trigger (all clients)",
    "asr_personalized_targeted":
        "Backdoor ASR (personalized models, perturb decomposition)",
    "asr_global_targeted":
        "Backdoor ASR (global model)",
    "asr_unfiltered":
        "Backdoor ASR (target class included)",
    # 主任务
    "mta_personalized": "Main-task accuracy (personalized models)",
    "mta_global": "Main-task accuracy (global model)",
    "acc_personalized":
        "Clean accuracy, non-target classes (personalized models)",
    "acc_global": "Clean accuracy, non-target classes (global model)",
    "clean_loss_personalized": "Clean loss (personalized models)",
    "clean_loss_global": "Clean loss (global model)",
    # 轴
    "round": "Communication round",
    "bad_client_num": "Number of malicious clients $N_m$",
    "poison_rate": r"Poisoning probability $\rho$",
}

# legend / 分面标题上的短名（轴标签太长，这里要短的）
_SHORT: Dict[str, str] = {
    "bad_client_num": "$N_m$",
    "poison_rate": r"$\rho$",
    "seed": "seed",
    "schedule": "schedule",
}


def pretty_label(column: str, *, short: bool = False) -> str:
    """内部列名 -> 图上用的英文标签。

    没登记的列名**原样返回**（而不是报错或编一个）—— 出图不该因为多了一列
    就崩掉，但也不该假装知道那是什么。
    """
    if short and column in _SHORT:
        return _SHORT[column]
    return _PRETTY.get(column, column)


def asr_axis_label(column: str, *, suffix: str = "") -> str:
    """ASR 轴的标签。``suffix`` 如 "(tail mean)"。"""
    base = pretty_label(column)
    return f"{base} {suffix}".strip()


# ---------------------------------------------------------------------------
# 需要 matplotlib 的部分（**函数体内**才 import，保持本模块本机可导入）
# ---------------------------------------------------------------------------
PUBLICATION_RCPARAMS: Dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,     # 我们自己调 tight_layout，别让它抢
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def apply_publication_style() -> None:
    """把出版级 rcParams 应用到全局。出图入口调一次即可。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def fig_axis_labels(fig, xlabel: str, ylabel: str) -> str:
    """整张分面图的公共 x/y 轴标签。返回实际用的机制名（便于测试/日志）。

    优先 `fig.supxlabel` / `fig.supylabel`（matplotlib >= 3.4）——它们**参与
    布局**，`tight_layout` 会给它们留位置。旧版本退回 `fig.text`，
    但这时必须**自己留边距**（`subplots_adjust`），否则就是原来那个
    「标签压在刻度数字上」的样子。旧实现只做了退路、没留边距，所以哪怕在
    新 matplotlib 上也是压着的。
    """
    if hasattr(fig, "supxlabel") and hasattr(fig, "supylabel"):
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        return "suplabel"
    fig.subplots_adjust(left=0.11, bottom=0.11)
    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=10)
    fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=10)
    return "figtext"


def finish_facets(axes_flat: Sequence[Any], n_used: int,
                  nrow: int, ncol: int) -> None:
    """隐藏多余的空面，并把**每列最下面那个可见 panel** 的 x 刻度标签打开。

    `sharex=True` 只给最后一行留 x 刻度标签。当最后一行有空格时，它上面那个
    panel 就成了那一列的底部，却没有标签 —— 报告里 E1-1b/E1-2 中间那个 panel
    没有 x 轴刻度就是这么来的（**不是 matplotlib 的 bug，是我们没处理**）。
    """
    for axis in list(axes_flat)[n_used:]:
        axis.set_axis_off()
    for col in range(ncol):
        bottom = None
        for row in range(nrow):
            idx = row * ncol + col
            if idx < n_used:
                bottom = axes_flat[idx]
        if bottom is not None:
            bottom.tick_params(axis="x", labelbottom=True)
            for label in bottom.get_xticklabels():
                label.set_visible(True)


def outside_legend(fig, handles: Sequence[Any], labels: Sequence[str],
                   title: Optional[str] = None, *, ncol: int = 1):
    """把 legend 放在**画布之外**的右侧，不压任何 panel。

    旧写法 `fig.legend(loc="center right")` 是在画布**内部**靠右居中 ——
    正好盖住最右侧那个 panel。`bbox_to_anchor=(1.0, 0.5)` + `loc="center left"`
    把它挪到画布右缘之外；`savefig(bbox_inches="tight")` 会把它一起收进来，
    所以不会被裁掉。
    """
    return fig.legend(handles, labels, title=title, ncol=ncol,
                      loc="center left", bbox_to_anchor=(1.0, 0.5),
                      borderaxespad=0.0)
