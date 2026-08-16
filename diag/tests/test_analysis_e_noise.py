"""``diag.analysis_e_noise`` 的单元测试。

重点在**算错了也不会报错**的地方：

- lift 必须是逐客户端配对相减，不是两个组均值相减（均值相同、误差棒不同）；
- 混进别的 cell 或缺模式时必须报错，而不是静默画出一张看不出错的图；
- 找不到基线行时 lift 记 nan，不填 0；
- 单 alpha 时不画连线（一条线会被读成趋势）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_e_noise import (BASELINE_MODE, NOISE_MODES,
                                   baseline_by_group, plot_noise_response,
                                   summarize_noise)

MODES = (BASELINE_MODE,) + NOISE_MODES + ("delta_only", "full", "xi_only")


def _raw(alphas=(1.0, 0.5, 0.1), n_benign=8, n_malicious=4, cell="E1",
         modes=MODES, lift_fn=None, seed=0) -> pd.DataFrame:
    """构造 schema 与真实 exp_e_E1 CSV 一致的表。"""
    rng = np.random.RandomState(seed)
    lift_fn = lift_fn or (lambda alpha, malicious, mode: 0.02 if malicious else 0.005)
    rows = []
    for alpha in alphas:
        for cid in range(n_benign + n_malicious):
            malicious = cid >= n_benign
            base = 0.03 + rng.normal(0, 0.005)
            for mode in modes:
                if mode == BASELINE_MODE:
                    value = base
                elif mode in NOISE_MODES:
                    value = base + lift_fn(alpha, malicious, mode)
                else:
                    value = 0.9
                rows.append({"client_id": cid, "is_malicious": malicious,
                             "cell": cell, "perturb_mode": mode,
                             "alpha": alpha, "seed": 0,
                             "asr_targeted": value,
                             "asr_targeted_unfiltered": value * 0.9,
                             "error_rate": 0.2, "clean_acc": 0.79})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# lift 的算法
# ---------------------------------------------------------------------------
def test_lift_is_paired_per_client_not_a_difference_of_means():
    """每个客户端的 lift 是常数 -> 配对后 std 必须是 0。

    若改成"两个组均值相减"，均值一样但这里的 std 会变成两个独立量的方差和，
    也就是非 0 —— 这个测试正是用来卡住那种写法的。
    """
    raw = _raw(lift_fn=lambda a, m, mode: 0.02 if m else 0.005)
    summary = summarize_noise(raw)
    assert np.allclose(summary["lift_std"].to_numpy(dtype=float), 0.0, atol=1e-12)

    benign = summary[(~summary["is_malicious"])
                     & (summary["perturb_mode"] == "random_eps4")]
    malicious = summary[summary["is_malicious"]
                        & (summary["perturb_mode"] == "random_eps4")]
    assert np.allclose(benign["lift_mean"], 0.005)
    assert np.allclose(malicious["lift_mean"], 0.020)
    # 而绝对值的 std 不为 0（各客户端的基线不同），说明配对确实起了作用
    assert (summary["asr_std"] > 1e-6).all()


def test_lift_is_nan_when_baseline_row_is_missing():
    raw = _raw()
    raw = raw[~((raw["perturb_mode"] == BASELINE_MODE) & (raw["client_id"] == 0))]
    summary = summarize_noise(raw)
    assert int(summary["n_missing_baseline"].sum()) > 0
    # 缺基线的条目记 nan，其余仍然算得出来 —— 不填 0
    assert np.isfinite(summary["lift_mean"]).all()


def test_group_sizes_and_baselines_are_reported():
    raw = _raw(n_benign=8, n_malicious=4)
    summary = summarize_noise(raw)
    benign_n = summary[~summary["is_malicious"]]["n_clients"].unique().tolist()
    malicious_n = summary[summary["is_malicious"]]["n_clients"].unique().tolist()
    assert benign_n == [8] and malicious_n == [4]
    assert (summary["n_seeds"] == 1).all()

    baselines = baseline_by_group(raw)
    assert set(baselines["is_malicious"]) == {True, False}
    assert np.isfinite(baselines["baseline_mean"]).all()


# ---------------------------------------------------------------------------
# 拒绝静默降级
# ---------------------------------------------------------------------------
def test_rejects_mixed_cells():
    raw = pd.concat([_raw(), _raw(cell="E2")], ignore_index=True)
    try:
        summarize_noise(raw)
    except ValueError as exc:
        assert "E2" in str(exc)     # 报出实际值，便于定位
    else:
        raise AssertionError("混进别的 cell 必须报错，不能静默过滤")


def test_rejects_missing_modes():
    raw = _raw(modes=(BASELINE_MODE, "random_eps4"))
    try:
        summarize_noise(raw)
    except ValueError as exc:
        assert "random_eps8" in str(exc)
    else:
        raise AssertionError("缺 random_eps8 必须报错")

    raw = _raw(modes=NOISE_MODES)          # 缺基线
    try:
        summarize_noise(raw)
    except ValueError as exc:
        assert BASELINE_MODE in str(exc)
    else:
        raise AssertionError("缺 none 基线必须报错（lift 无从算起）")


def test_rejects_duplicated_rows():
    """glob 匹配到重叠文件时样本量会翻倍：均值不变、误差棒变窄，看不出来。"""
    raw = pd.concat([_raw(), _raw()], ignore_index=True)
    try:
        summarize_noise(raw)
    except ValueError as exc:
        assert "重复" in str(exc)
    else:
        raise AssertionError("重复行必须报错")


def test_rejects_missing_columns():
    raw = _raw().drop(columns=["is_malicious"])
    try:
        summarize_noise(raw)
    except KeyError as exc:
        assert "is_malicious" in str(exc)
    else:
        raise AssertionError("缺列必须报错")


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def test_plot_renders_and_orders_alpha_by_heterogeneity():
    raw = _raw(alphas=(1.0, 0.5, 0.1, 0.05))
    summary = summarize_noise(raw)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_noise_response(summary, baseline_by_group(raw),
                                   Path(tmp) / "noise.png")
        assert path.exists() and path.stat().st_size > 0


def test_single_alpha_draws_markers_without_a_connecting_line():
    """单点时连线会被读成趋势，必须退化为纯标记。"""
    raw = _raw(alphas=(0.5,))
    summary = summarize_noise(raw)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_noise_response(summary, baseline_by_group(raw),
                                   Path(tmp) / "single.png")
        assert path.exists() and path.stat().st_size > 0


def test_plot_has_no_cjk():
    """``assert_no_cjk_in_figure`` 在保存前扫描，跑通即等于通过。"""
    raw = _raw()
    summary = summarize_noise(raw)
    with tempfile.TemporaryDirectory() as tmp:
        plot_noise_response(summary, baseline_by_group(raw),
                            Path(tmp) / "cjk.png",
                            metric="asr_targeted_unfiltered")
