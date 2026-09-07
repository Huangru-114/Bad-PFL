"""`diag.figstyle` —— 出版级图样式的纯逻辑部分（回应导师意见 #3）。

**为什么这个模块能在本地跑**：`figstyle` 刻意不在模块层 import matplotlib，
布局算术与标签映射都是纯 Python。`analysis_exp1.py` 一 import matplotlib
本机就跑不了，那边的图只能在集群上看 —— 但「4 个 panel 排成 3+1」
「y 轴写内部列名」这两类问题的**判据**在这里，本地就能钉死。

不覆盖的部分（如实说）：实际渲染出来标签有没有重叠、legend 有没有压住 panel，
本地看不到。那要在集群上跑 `analysis_exp1` 之后目检 PNG。
"""

from __future__ import annotations

from diag.figstyle import (PUBLICATION_RCPARAMS, asr_axis_label, facet_layout,
                           pretty_label)


# ══════════════════════════════════════════════════════════════════════════
# 分面布局：不要 3+1
# ══════════════════════════════════════════════════════════════════════════
def test_four_panels_are_two_by_two_not_three_plus_one():
    """报告里 E1-1b / E1-2 / E1-6 第二行大片空白的直接原因。

    旧公式 `ncol = min(3, n)` 在 n=4 时给 (2, 3) —— 两行三列、空两格。
    """
    assert facet_layout(4) == (2, 2)


def test_layout_never_wastes_more_than_two_cells():
    """空格多了不只是难看：空格上方那个 panel 会丢掉 x 刻度标签
    （sharex 只给最后一行），也就是「中间 panel 没有 x 轴」那件事。"""
    for n in range(1, 13):
        nrow, ncol = facet_layout(n)
        assert nrow * ncol >= n, (n, nrow, ncol)
        assert nrow * ncol - n <= 2, f"n={n} 浪费了 {nrow * ncol - n} 格"


def test_layout_prefers_landscape_over_a_tall_strip():
    """竖长条会被 PDF 从中间截断 —— 报告 Figure 8（E1B-1，6 panel 竖排）
    就断在 p.6 / p.7 之间。同样的 n，行数不该多于列数太多。"""
    for n in (4, 5, 6, 7, 8, 9):
        nrow, ncol = facet_layout(n)
        assert nrow <= ncol, f"n={n} 排成了 {nrow}x{ncol}（竖）"


def test_layout_is_defined_for_degenerate_inputs():
    assert facet_layout(1) == (1, 1)
    assert facet_layout(0) == (1, 1)
    assert facet_layout(-3) == (1, 1)


def test_layout_falls_back_to_three_columns_when_large():
    assert facet_layout(10) == (4, 3)
    assert facet_layout(12) == (4, 3)


# ══════════════════════════════════════════════════════════════════════════
# 标签：图上不出现内部列名
# ══════════════════════════════════════════════════════════════════════════
def test_asr_columns_get_human_readable_labels():
    """y 轴上写 `asr_paper_benign` 读者读不出两件关键信息：
    benign 指「在良性客户端的个性化模型上测」，filtered 指「已排除真标签
    就是目标类的样本」。两者都要在标签里说出来。"""
    label = pretty_label("asr_paper_filtered_benign")
    assert "asr_paper" not in label
    assert "benign" in label.lower()
    assert "excluded" in label.lower(), "没说清楚 filtered 是什么意思"

    unfiltered = pretty_label("asr_paper_benign")
    assert "included" in unfiltered.lower(), "没说清楚 unfiltered 是什么意思"
    assert unfiltered != label, "filtered 与 unfiltered 的标签必须不同"


def test_every_tier_column_is_mapped():
    """三档 × 两口径 = 六列，一个都不能漏 —— 漏掉的那个会把内部列名画上去。"""
    for scope in ("benign", "malicious", "all"):
        for stem in ("asr_paper_{}", "asr_paper_filtered_{}"):
            col = stem.format(scope)
            assert pretty_label(col) != col, f"{col} 没有人话标签"


def test_unknown_column_passes_through_instead_of_crashing():
    """多了一列不该让出图崩掉；但也不该编一个标签出来。"""
    assert pretty_label("some_future_column") == "some_future_column"


def test_short_labels_are_used_for_legends():
    assert pretty_label("bad_client_num", short=True) == "$N_m$"
    assert pretty_label("poison_rate", short=True) == r"$\rho$"
    # 长名仍然是给轴用的，两者必须不同
    assert pretty_label("bad_client_num") != pretty_label("bad_client_num",
                                                          short=True)


def test_asr_axis_label_appends_a_suffix():
    base = asr_axis_label("asr_paper_filtered_benign")
    with_suffix = asr_axis_label("asr_paper_filtered_benign",
                                 suffix="(tail mean)")
    assert with_suffix == f"{base} (tail mean)"
    assert asr_axis_label("asr_paper_benign", suffix="") == \
        pretty_label("asr_paper_benign"), "空 suffix 不该留下尾随空格"


def test_no_cjk_anywhere_in_the_label_tables():
    """铁律 #4：图里绝不出现 CJK。标签表是唯一会被画上去的文本来源。

    `assert_no_cjk_in_figure` 在保存前扫整张图会拦截，但那时已经画完了 ——
    这条把它提前到本地。
    """
    from diag import figstyle
    for table in (figstyle._PRETTY, figstyle._SHORT):
        for key, value in table.items():
            for ch in value:
                assert not ("一" <= ch <= "鿿"), f"{key}: {value}"


def test_rcparams_do_not_enable_autolayout():
    """`figure.autolayout=True` 会和我们自己调的 tight_layout / 画布外 legend
    打架（它每次 draw 都重排，把画布外的 legend 挤没）。"""
    assert PUBLICATION_RCPARAMS["figure.autolayout"] is False
    assert PUBLICATION_RCPARAMS["savefig.bbox"] == "tight", \
        "画布外的 legend 靠 bbox_inches='tight' 才不会被裁掉"
