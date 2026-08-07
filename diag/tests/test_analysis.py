"""``diag.analysis`` 的测试：汇总数学 + 全部绘图函数能在**合成数据**上跑通。

绘图函数必须不依赖真实结果的存在 —— 这是冒烟测试的硬要求。
本文件生成的数字是随机的，**没有任何科学含义**。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis import (auc_score, mannwhitney, pearson_spearman, plot_a1,
                           plot_a2, plot_a3, plot_a4, plot_c1, plot_c2, plot_c3,
                           plot_d1, plot_d2, summarize, summarize_c, summarize_d)

ALPHAS = [1.0, 0.5, 0.1, 0.05]
SEEDS = [0, 1, 2]
GROUPS = ["real", "delta_only", "delta_plus_xi", "random_noise"]


def _fake_exp_a(n_clients: int = 12) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for group in GROUPS:
                for cid in range(n_clients):
                    rows.append({
                        "client_id": cid, "is_malicious": cid >= n_clients - 2,
                        "is_malicious_slot": cid >= n_clients - 2,
                        "poison_ratio": 0.2, "n_local_samples": 500,
                        "n_target_samples_local": int(rng.randint(0, 120)),
                        "n_participations": 30,
                        "alpha": alpha, "seed": seed, "mode": "clean",
                        "group": group,
                        "overlap": float(rng.rand()),
                        "nat_rate": float(rng.rand()),
                        "n_probe_samples": 100, "knn_k": 20,
                        "xi_source": "own_model",
                    })
    return pd.DataFrame(rows)


def _fake_exp_c(n_clients: int = 12) -> pd.DataFrame:
    rng = np.random.RandomState(1)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for cid in range(n_clients):
                is_bad = cid >= n_clients - 2
                rows.append({
                    "client_id": cid, "is_malicious": is_bad,
                    "is_malicious_slot": is_bad, "poison_ratio": 0.2,
                    "n_local_samples": 500,
                    "n_target_samples_local": int(rng.randint(0, 120)),
                    "n_participations": 30, "alpha": alpha, "seed": seed,
                    "mode": "attack",
                    "A_observable": float(rng.randn() + (2.0 if is_bad else 0.0)),
                    "A_lambda_0.0": float(rng.randn()),
                    "A_lambda_0.5": float(rng.randn()),
                    "A_lambda_1.0": float(rng.randn()),
                    "A_lambda_2.0": float(rng.randn()),
                    "l2_to_median": float(abs(rng.randn())),
                    "sign_consistency": float(rng.rand()),
                    "n_valid_prototypes": 10,
                    "nat_rate": float(rng.rand() * 0.2),
                    "asr": float(rng.rand()),
                    "E": float(rng.randn() + (0.5 if is_bad else 0.0)),
                })
    return pd.DataFrame(rows)


def _fake_exp_d(n_clients: int = 12, num_classes: int = 10) -> pd.DataFrame:
    rng = np.random.RandomState(2)
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for cid in range(n_clients):
                is_bad = cid >= n_clients - 2
                for cls in range(num_classes):
                    rows.append({
                        "client_id": cid, "is_malicious": is_bad,
                        "is_malicious_slot": is_bad, "poison_ratio": 0.2,
                        "n_local_samples": 500,
                        "n_target_samples_local": 50, "n_participations": 30,
                        "alpha": alpha, "seed": seed, "mode": "attack",
                        "class_id": cls, "is_target_class": cls == 0,
                        "n_class_samples_probe": 100,
                        "n_class_samples_local": 50,
                        "J": float(abs(rng.randn()) + 0.1),
                        "v": float(rng.randn() + (0.5 if is_bad else 0.0)),
                        "benign_baseline_J": 1.0,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------
def test_auc_score_perfect_separation():
    assert auc_score([0.0, 0.1, 0.9, 1.0], [False, False, True, True]) == 1.0
    assert auc_score([0.9, 1.0, 0.0, 0.1], [False, False, True, True]) == 0.0


def test_auc_score_ties_give_half():
    assert abs(auc_score([1.0, 1.0], [True, False]) - 0.5) < 1e-9


def test_pearson_spearman_perfect_correlation():
    r, rho = pearson_spearman([1, 2, 3, 4], [2, 4, 6, 8])
    assert abs(r - 1.0) < 1e-9 and abs(rho - 1.0) < 1e-9


def test_mannwhitney_returns_pvalue():
    u, p = mannwhitney([1, 2, 3], [10, 11, 12])
    assert math.isfinite(u) and 0.0 <= p <= 1.0
    assert math.isnan(mannwhitney([], [1, 2])[1])


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def test_summarize_shape_and_cv_definition():
    raw = _fake_exp_a()
    summary = summarize(raw)
    assert len(summary) == len(ALPHAS) * len(SEEDS) * len(GROUPS)
    assert set(summary.columns) >= {"alpha", "seed", "group", "cv", "mean",
                                    "std", "n_clients", "ng_vs_real"}

    row = summary.iloc[0]
    block = raw[(raw["alpha"] == row["alpha"]) & (raw["seed"] == row["seed"])
                & (raw["group"] == row["group"])]["overlap"].to_numpy()
    expected = block.std(ddof=1) / abs(block.mean())
    assert abs(row["cv"] - expected) < 1e-9


def test_summarize_ng_is_zero_for_real_group():
    summary = summarize(_fake_exp_a())
    real = summary[summary["group"] == "real"]
    assert (real["ng_vs_real"].abs() < 1e-12).all(), "real 组相对自身的 NG 必须是 0"


def test_summarize_missing_column_raises():
    try:
        summarize(pd.DataFrame({"alpha": [1.0], "seed": [0]}))
    except KeyError as exc:
        assert "group" in str(exc)
    else:
        raise AssertionError("缺列时必须抛 KeyError")


def test_summarize_c_and_d_shapes():
    summary_c = summarize_c(_fake_exp_c())
    assert len(summary_c) == len(ALPHAS) * len(SEEDS) * 3
    assert (summary_c["auc"].between(0.0, 1.0)).all()

    summary_d = summarize_d(_fake_exp_d())
    assert len(summary_d) == len(ALPHAS) * len(SEEDS)
    assert (summary_d["auc_max_v"].between(0.0, 1.0)).all()


# ---------------------------------------------------------------------------
# 绘图（合成数据）
# ---------------------------------------------------------------------------
def test_all_plots_produce_non_empty_files():
    raw_a, raw_c, raw_d = _fake_exp_a(), _fake_exp_c(), _fake_exp_d()
    summary_a = summarize(raw_a)
    summary_c = summarize_c(raw_c)
    summary_d = summarize_d(raw_d)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        produced = [
            plot_a1(summary_a, out / "a1.png"),
            plot_a2(summary_a, out / "a2.png"),
            plot_a3(raw_a, out / "a3.png"),
            plot_a4(raw_a, out / "a4.png"),
            plot_c1(raw_c, out / "c1.png"),
            plot_c2(summary_c, out / "c2.png"),
            plot_c3(raw_c, out / "c3.png"),
            plot_d1(raw_d, out / "d1.png"),
            plot_d2(summary_d, out / "d2.png"),
        ]
        for path in produced:
            assert path.exists(), f"{path} 未生成"
            assert path.stat().st_size > 1000, f"{path} 是空文件"


def test_plot_d1_accepts_explicit_poisoned_source_classes():
    raw_d = _fake_exp_d()
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_d1(raw_d, Path(tmp) / "d1.png", poisoned_source_classes=[1, 2])
        assert path.stat().st_size > 1000
