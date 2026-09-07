"""`diag.viz_trigger` —— 触发器可视化的可测部分（回应导师意见 #1）。

**验证范围如实**：本机没有 torch / torchvision / matplotlib，画不出图。
这里测的是**会决定图对不对**的那几件纯计算：放大映射、幅度统计、
panel 标题里的数字、缺文件的提前报错。真正的 PNG 要在集群上出、目检。

盯的两件事：
1. 扰动图**必须写明放大倍数**。ξ 与 δ 的 L∞ 上界是 4/255 ≈ 0.0157，
   不放大就是一块灰；放大了不写倍数，读者无法判断它是 4/255 还是 40/255。
2. 超范围**截断**而不是重新归一化。重新归一化会让「这张扰动更大」看不出来，
   而跨 panel 可比正是这张图存在的意义。
"""

from __future__ import annotations

import math
from pathlib import Path

from diag.viz_trigger import (EPS, _display_value, _flatten,
                              amplify_for_display, missing_files, panel_titles,
                              perturbation_stats, required_files)


# ══════════════════════════════════════════════════════════════════════════
# 放大映射
# ══════════════════════════════════════════════════════════════════════════
def test_zero_maps_to_neutral_grey():
    """0 必须是 0.5：读者靠「比灰亮/比灰暗」读出扰动的符号。"""
    assert amplify_for_display(0.0, EPS) == 0.5


def test_full_scale_maps_to_the_ends():
    assert amplify_for_display(EPS, EPS) == 1.0
    assert amplify_for_display(-EPS, EPS) == 0.0


def test_out_of_range_is_clipped_not_renormalised():
    """两倍量程仍然是 1.0（截断），不是被压回 1.0 以内后整体缩放。

    若改成重新归一化，同一个 panel 看起来会和满量程的一样 ——
    「总残差可以到 8/255」这件事就画不出来了。
    """
    assert amplify_for_display(2 * EPS, EPS) == 1.0
    assert amplify_for_display(-5 * EPS, EPS) == 0.0


def test_mapping_is_linear_between_the_ends():
    assert math.isclose(amplify_for_display(EPS / 2, EPS), 0.75)
    assert math.isclose(amplify_for_display(-EPS / 2, EPS), 0.25)


def test_nested_lists_are_mapped_elementwise():
    got = amplify_for_display([[0.0, EPS], [-EPS, 3 * EPS]], EPS)
    assert got == [[0.5, 1.0], [0.0, 1.0]]


def test_scale_must_be_positive():
    for bad in (0.0, -1.0):
        try:
            amplify_for_display(0.0, bad)
        except ValueError:
            continue
        raise AssertionError(f"scale={bad} 应该报错")


def test_numpy_path_agrees_with_the_python_path():
    """两条路的公式由 `_display_value` 唯一定义 —— 这条在集群上逐元素对拍。

    本机没有 numpy 时自动跳过（不是假绿：跳过会被报成 skip）。
    """
    try:
        import numpy as np
    except ImportError:
        import unittest
        raise unittest.SkipTest("本机无 numpy；这条在集群上跑")
    arr = np.linspace(-3 * EPS, 3 * EPS, 41)
    vec = amplify_for_display(arr, EPS)
    for a, v in zip(arr.tolist(), vec.tolist()):
        assert math.isclose(v, _display_value(a, EPS), abs_tol=1e-12), a


# ══════════════════════════════════════════════════════════════════════════
# 幅度统计
# ══════════════════════════════════════════════════════════════════════════
def test_stats_report_linf_and_the_fraction_of_the_budget():
    s = perturbation_stats([[0.0, EPS], [-EPS / 2, 0.0]])
    assert math.isclose(s["linf"], EPS)
    assert math.isclose(s["frac_of_eps"], 1.0)
    assert math.isclose(s["mean_abs"], (EPS + EPS / 2) / 4)


def test_stats_detect_a_budget_violation():
    """总残差可以到 8/255（ξ 与 δ 相加、δ 之后没有再 clip）。
    统计量要能把这件事**报出来**，而不是悄悄截断。"""
    s = perturbation_stats([2 * EPS])
    assert math.isclose(s["frac_of_eps"], 2.0), \
        "超预算没有被如实报出来 —— 报告里 8/255 那句就没有证据了"


def test_empty_input_is_nan_not_zero():
    """0 会被读成「扰动为零」，那是个强结论（铁律 #5）。"""
    s = perturbation_stats([])
    assert all(math.isnan(v) for v in s.values())


def test_flatten_handles_arrays_lists_and_scalars():
    assert _flatten(1.5) == [1.5]
    assert _flatten([[1, 2], [3]]) == [1.0, 2.0, 3.0]


# ══════════════════════════════════════════════════════════════════════════
# panel 标题：放大倍数与实测幅度都要出现
# ══════════════════════════════════════════════════════════════════════════
def test_titles_state_the_amplification_factor():
    s = perturbation_stats([EPS])
    titles = panel_titles(s, s, s, EPS)
    assert len(titles) == 5
    amplified = [t for t in titles if "amplified" in t]
    assert len(amplified) == 3, "三个扰动 panel 都要写放大倍数"
    for t in amplified:
        assert "x32" in t, f"没写出倍数：{t}"


def test_titles_carry_the_measured_magnitude():
    s = perturbation_stats([EPS / 2])
    titles = panel_titles(s, s, s, EPS)
    joined = " ".join(titles)
    assert "L_\\infty" in joined and "0.0078" in joined
    assert "of 4/255" in joined, "没有把幅度换算成预算的比例"


def test_titles_are_ascii_only():
    """铁律 #4：图里一律英文。标题是会被画上去的文本。"""
    s = perturbation_stats([EPS])
    for t in panel_titles(s, s, s, EPS):
        for ch in t:
            assert not ("一" <= ch <= "鿿"), t


# ══════════════════════════════════════════════════════════════════════════
# 缺文件要提前报，不要等 GPU 排到
# ══════════════════════════════════════════════════════════════════════════
def test_required_files_cover_generator_model_and_meta():
    names = {p.name for p in required_files("/tmp/x", 3)}
    assert names == {"generator.pt", "client_3.pt", "meta.json"}


def test_missing_files_are_all_reported_at_once():
    """一次报齐，而不是修一个再发现下一个。"""
    gone = missing_files("/tmp/definitely-not-here", 0)
    assert len(gone) == 3
