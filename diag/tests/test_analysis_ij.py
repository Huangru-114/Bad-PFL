"""``diag.analysis_ij`` 与 ``diag.run_density_sweep`` 的单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_ij import (DEFENSE_STYLE, load_frames, plot_j1, plot_j2,
                              plot_j3, plot_j4, plot_j5)
from diag.run_density_sweep import build_commands, expected_gap


def _detection(defense="flame", n_rounds=120, with_cosine=False):
    rng = np.random.RandomState(0)
    rows = []
    for r in range(1, n_rounds + 1):
        n_mal = [0, 1, 2][r % 3]
        row = {"round": r, "defense": defense, "tau": 0.2, "alpha": 0.25,
               "alpha_dirichlet": 0.5, "seed": 0,
               "n_malicious_in_round": n_mal, "n_selected": 10,
               "influence_malicious_mean": 0.3 if n_mal else np.nan,
               "influence_benign_mean": 0.7,
               "malicious_rank_l2_mean": (float(rng.randint(1, 5))
                                          if n_mal else np.nan),
               "tpr": 0.5 if n_mal else None, "fpr": 0.2 if n_mal else None,
               "youden_j": 0.3 if n_mal else None, "n_excluded": 4,
               "clip_rate_malicious": 0.95, "clip_rate_benign": 0.43}
        if with_cosine:
            row["cos_to_others_malicious_mean"] = 0.1 + rng.normal(0, 0.02)
            row["cos_to_others_benign_mean"] = 0.15 + rng.normal(0, 0.02)
        rows.append(row)
    return pd.DataFrame(rows)


def _implantation(defense="flame", n_points=50):
    rows = []
    for i in range(1, n_points + 1):
        r = i * 10
        rows.append({"round": r, "defense": defense,
                     "asr_global_targeted": min(0.99, r / 500),
                     "asr_personalized_targeted": min(0.95, r / 600),
                     "acc_personalized": 0.7, "acc_global": 0.75,
                     "n_malicious_this_round": i % 2,
                     "rounds_since_last_malicious": i % 4})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------
def test_load_frames_splits_by_defense():
    merged = pd.concat([_detection("flame"), _detection("multi_krum")],
                       ignore_index=True)
    frames = load_frames(merged)
    assert set(frames) == {"flame", "multi_krum"}
    assert (frames["flame"]["round"].is_monotonic_increasing)


def test_load_frames_rejects_missing_defense_column():
    try:
        load_frames(pd.DataFrame({"round": [1, 2]}))
    except KeyError as exc:
        assert "defense" in str(exc)
    else:
        raise AssertionError("缺 defense 列必须报错")


# ---------------------------------------------------------------------------
# 图能出且不含 CJK（_finish 内部扫描，跑通即通过）
# ---------------------------------------------------------------------------
def test_all_plots_render_without_cjk():
    detection = pd.concat([_detection("flame", with_cosine=True),
                           _detection("multi_krum")], ignore_index=True)
    implantation = pd.concat([_implantation("flame"),
                              _implantation("multi_krum")], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert plot_j1(implantation, out / "j1.png").exists()
        assert plot_j2(detection, out / "j2.png").exists()
        assert plot_j3(detection, out / "j3.png").exists()
        assert plot_j4(implantation, out / "j4.png").exists()
        assert plot_j5(detection, out / "j5.png").exists()


def test_j5_returns_none_without_the_cosine_columns():
    """老版本的检出 CSV 没有 cos_to_others_*，应当跳过而不是画一张空图。"""
    with tempfile.TemporaryDirectory() as tmp:
        assert plot_j5(_detection("flame", with_cosine=False),
                       Path(tmp) / "j5.png") is None


def test_j2_uses_only_single_malicious_rounds():
    """有 2 个以上恶意时记录的是名次**均值**，不是个体名次，必须排除。"""
    detection = _detection("flame")
    single = int((detection["n_malicious_in_round"] == 1).sum())
    assert single < len(detection)            # 构造里确实有多恶意的轮次
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_j2(detection, Path(tmp) / "j2.png")
        assert path.exists()      # 必须在 with 内断言，出了块临时目录就没了


def test_j4_restricts_to_the_plateau_by_default():
    """默认只用后半段，避免把 ΔT 效应与训练进度混在一起。"""
    implantation = _implantation("flame", n_points=50)     # round 10..500
    with tempfile.TemporaryDirectory() as tmp:
        assert plot_j4(implantation, Path(tmp) / "a.png").exists()
        assert plot_j4(implantation, Path(tmp) / "b.png", min_round=0).exists()


def test_defense_style_covers_every_defense_we_run():
    from diag.defenses import ABLATION_VARIANTS, DEFENSES

    for name in list(DEFENSES) + list(ABLATION_VARIANTS):
        assert name in DEFENSE_STYLE, name


# ---------------------------------------------------------------------------
# 密度扫描
# ---------------------------------------------------------------------------
def test_expected_gap_shrinks_as_density_grows():
    gaps = [expected_gap(100, bad, 10)["expected_gap"] for bad in (1, 2, 5, 10)]
    assert gaps == sorted(gaps, reverse=True)
    # 10 个恶意时期望间隔 < 1 轮 —— 这正是 J-4 画不出衰减的原因
    assert gaps[-1] < 1.0
    # 1 个恶意时期望间隔 ~9 轮，J-4 才有动态范围
    assert gaps[0] > 5.0


def test_expected_gap_matches_the_hypergeometric_pmf():
    stats = expected_gap(100, 10, 10)
    assert abs(stats["p_no_malicious"] - 0.3305) < 1e-3
    assert abs(stats["expected_malicious_per_round"] - 1.0) < 1e-9


def test_build_commands_always_tags_the_run():
    """不加 tag 的话不同密度的 run 会静默覆盖彼此的 checkpoint。"""
    commands = build_commands([1, 10], ["fedavg"], alpha=0.5, seed=0,
                              total_round=None, eval_every=10,
                              ckpt_root="./ckpt", instrument_dir="instr",
                              results_dir=None)
    trains = [c for c in commands if "run_fl" in " ".join(c)]
    assert len(trains) == 2
    for cmd, expected in zip(trains, ("bad1", "bad10")):
        assert "--run-tag" in cmd and cmd[cmd.index("--run-tag") + 1] == expected
        assert "--bad-client-num" in cmd
    # 每个训练命令后面都跟一条对应的汇总命令
    assert len(commands) == 2 * len(trains)
