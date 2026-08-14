"""``diag.analysis_e`` 的测试 + 「图中禁止中文」的强制检查。

本文件生成的数字是随机的，**没有任何科学含义**，只用于验证绘图与汇总能跑通。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diag.analysis import assert_no_cjk_in_figure, contains_cjk
from diag.analysis_e import (CELL_STYLE, EXP_E_MODE_ORDER, MODE_LABEL,
                             classify_floor, decomposition_e, plot_e1, plot_e2,
                             plot_e3, summarize_e)

ALPHAS = [1.0, 0.5, 0.1, 0.05]
SEEDS = [0, 1]
CELLS = ["E1", "E2", "E3"]


def _fake_exp_e(n_clients: int = 12) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for cell in CELLS:
                for mode in EXP_E_MODE_ORDER:
                    for cid in range(n_clients):
                        is_target_only = mode == "real_target"
                        rows.append({
                            "client_id": cid,
                            "is_malicious": cid >= n_clients - 2,
                            "cell": cell, "perturb_mode": mode,
                            "generator_source": "attack/final",
                            "xi_source": f"self/client_{cid}",
                            "alpha": alpha, "seed": seed, "probe_set": "shared",
                            "n_eval_samples": 0 if is_target_only else 1800,
                            "asr_targeted": (np.nan if is_target_only
                                             else float(rng.rand())),
                            "asr_targeted_unfiltered": float(rng.rand()),
                            "error_rate": (np.nan if is_target_only
                                           else float(rng.rand())),
                            "clean_acc": float(rng.rand()),
                            "linf_budget": 8 / 255, "poison_ratio": 0.2,
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图中禁止中文
# ---------------------------------------------------------------------------
def test_contains_cjk_detects_chinese_and_ignores_ascii():
    assert contains_cjk("对抗地板")
    assert contains_cjk("alpha=0.05 异构")
    assert not contains_cjk("ASR_targeted (alpha=0.05)")
    assert not contains_cjk("delta + xi, 8/255")


def test_assert_no_cjk_catches_title():
    fig, ax = plt.subplots()
    ax.set_title("对抗地板")
    try:
        assert_no_cjk_in_figure(fig)
    except ValueError as exc:
        assert "不允许出现中文" in str(exc)
    else:
        raise AssertionError("标题含中文时必须抛异常")
    finally:
        plt.close(fig)


def test_assert_no_cjk_catches_legend_and_ticks():
    for setter in ("legend", "ticks", "text"):
        fig, ax = plt.subplots()
        if setter == "legend":
            ax.plot([0, 1], [0, 1], label="良性")
            ax.legend()
        elif setter == "ticks":
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["目标类", "其余"])
        else:
            ax.text(0.5, 0.5, "注释")
        try:
            assert_no_cjk_in_figure(fig)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{setter} 含中文时必须抛异常")
        finally:
            plt.close(fig)


def test_assert_no_cjk_passes_on_english_figure():
    fig, ax = plt.subplots()
    ax.set_title("E-1: attack success by perturbation mode")
    ax.set_xlabel("Dirichlet alpha")
    ax.plot([0, 1], [0, 1], label="ASR_attacked (E1)")
    ax.legend()
    assert_no_cjk_in_figure(fig)     # 不应抛异常
    plt.close(fig)


def test_all_exp_e_labels_are_ascii_only():
    """图里用到的常量标签必须全英文 —— 这是常量层面的守卫。"""
    for label in MODE_LABEL.values():
        assert not contains_cjk(label), label
    for label, _color in CELL_STYLE.values():
        assert not contains_cjk(label), label


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def test_summarize_e_shape_and_columns():
    summary = summarize_e(_fake_exp_e())
    expected = len(ALPHAS) * len(SEEDS) * len(CELLS) * len(EXP_E_MODE_ORDER)
    assert len(summary) == expected
    assert {"asr_targeted_mean", "asr_targeted_std",
            "asr_targeted_unfiltered_mean", "error_rate_mean",
            "clean_acc_mean", "n_clients"}.issubset(summary.columns)


def test_summarize_e_keeps_real_target_asr_as_nan():
    """real_target 的分母为空 -> 聚合后仍必须是 nan，不能变成 0。"""
    summary = summarize_e(_fake_exp_e())
    block = summary[summary["perturb_mode"] == "real_target"]
    assert block["asr_targeted_mean"].isna().all()
    assert block["asr_targeted_unfiltered_mean"].notna().all(), (
        "该组有意义的数字是不排除口径，必须存在")


def test_summarize_e_missing_column_raises():
    try:
        summarize_e(pd.DataFrame({"alpha": [1.0], "seed": [0]}))
    except KeyError as exc:
        assert "cell" in str(exc) or "perturb_mode" in str(exc)
    else:
        raise AssertionError("缺列时必须抛 KeyError")


def test_decomposition_delta_equals_e1_minus_e2():
    summary = summarize_e(_fake_exp_e())
    table = decomposition_e(summary, mode="full")
    assert len(table) == len(ALPHAS) * len(SEEDS)
    computed = table["ASR_attacked"] - table["ASR_clean_transfer"]
    assert np.allclose(table["Delta"], computed, equal_nan=True)


def test_decomposition_reports_floor_fraction_and_verdict():
    table = decomposition_e(summarize_e(_fake_exp_e()), mode="full")
    assert "floor_fraction" in table.columns
    assert table["verdict"].notna().all()


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------
def test_classify_floor_bands():
    assert "真后门" in classify_floor(0.05)
    assert "真后门" in classify_floor(0.15)
    assert "混合体" in classify_floor(0.30)
    assert "对抗攻击" in classify_floor(0.85)


def test_classify_floor_reports_undefined_gaps_honestly():
    """spec 的三段判据留了两个空隙，落进去必须如实说，不能靠拢邻近区间。"""
    for value in (0.17, 0.60):
        verdict = classify_floor(value)
        assert "未定义" in verdict, f"{value} -> {verdict}"


def test_classify_floor_handles_nan():
    assert "未能确定" in classify_floor(float("nan"))


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def test_all_exp_e_plots_produce_non_empty_files():
    raw = _fake_exp_e()
    summary = summarize_e(raw)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        produced = [plot_e1(summary, out / "e1.png"),
                    plot_e1(summary, out / "e1_a05.png", alpha=0.5),
                    plot_e2(summary, out / "e2.png"),
                    plot_e3(raw, out / "e3.png"),
                    plot_e3(raw, out / "e3_e2only.png", cells=["E2"])]
        for path in produced:
            assert path.exists() and path.stat().st_size > 1000, path


def test_plots_reject_chinese_via_finish():
    """_finish 的检查必须真的挂在保存路径上（防止有人日后加回中文标题）。"""
    from diag.analysis import _finish

    fig, ax = plt.subplots()
    ax.set_title("对抗地板")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _finish(fig, Path(tmp) / "x.png")
        except ValueError as exc:
            assert "不允许出现中文" in str(exc)
        else:
            raise AssertionError("_finish 必须在保存前拦下中文")


def test_plot_e1_handles_missing_cell():
    """只有 E1/E2 时（E3 尚未跑）也要能出图。"""
    raw = _fake_exp_e()
    raw = raw[raw["cell"].isin(["E1", "E2"])]
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_e1(summarize_e(raw), Path(tmp) / "e1.png")
        assert path.stat().st_size > 1000
