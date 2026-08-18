"""``diag.analysis_exposure`` 的单元测试。

这个模块的整个价值在于 **E 的权重对不对**。权重错了图还是会画出来，
而且看上去完全正常 —— 所以每种防御的分母都单独手算核对一遍。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_exposure import (clip_scale, collapse_diagnosis,
                                    effective_denominator, estimate_threshold,
                                    exposure_by_run, join_with_asr,
                                    plot_exposure, round_exposure)


def _record(defense, influence, norms, malicious, *, extra=None,
            effective_trim_n=-1, round_index=1):
    return {
        "round_index": round_index, "defense": defense,
        "influence": np.asarray(influence, dtype=float),
        "update_norm": np.asarray(norms, dtype=float),
        "is_malicious": np.asarray(malicious, dtype=bool),
        "effective_trim_n": effective_trim_n,
        "extra": extra or {},
    }


# ---------------------------------------------------------------------------
# 分母：每种防御单独手算
# ---------------------------------------------------------------------------
def test_fedavg_denominator_is_n_and_weights_sum_to_one():
    record = _record("fedavg", [1.0] * 4, [10, 20, 30, 40], [1, 0, 0, 0])
    assert effective_denominator(record) == 4.0
    out = round_exposure(record)
    # w = 1/4，恶意只有 client 0：E = 10/4
    assert np.isclose(out["exposure_malicious"], 2.5)
    assert np.isclose(out["exposure_benign"], (20 + 30 + 40) / 4)
    assert np.isclose(out["weight_sum"], 1.0)


def test_median_denominator_is_two_not_n():
    """逐坐标中位数 = 中间两位的均值，每位权重 1/2 —— 分母是 2，不是 N。"""
    # median_influence 之和恒为 2（每个坐标恰好两个客户端进中间两位）
    record = _record("median", [0.5, 0.5, 0.5, 0.5], [10, 10, 10, 10],
                     [1, 0, 0, 0])
    assert effective_denominator(record) == 2.0
    out = round_exposure(record)
    assert np.isclose(out["weight_sum"], 1.0)
    assert np.isclose(out["exposure_malicious"], 0.5 * 10 / 2)


def test_binary_defenses_use_the_number_selected():
    record = _record("multi_krum", [1.0, 1.0, 0.0, 0.0], [10, 20, 30, 40],
                     [1, 0, 0, 0])
    assert effective_denominator(record) == 2.0
    out = round_exposure(record)
    assert np.isclose(out["exposure_malicious"], 10 / 2)
    assert np.isclose(out["weight_sum"], 1.0)


def test_a_round_that_selects_nobody_contributes_zero():
    record = _record("multi_krum", [0.0] * 4, [10, 20, 30, 40], [1, 0, 0, 0])
    out = round_exposure(record)
    assert out["exposure_malicious"] == 0.0
    assert out["exposure_benign"] == 0.0


def test_invariant_denominator_is_effective_trim_n():
    """掩码是真实的幅度削减，必须留在分子里 —— Σw = mask_keep_ratio < 1。"""
    # influence = trim_survival * mask_keep_ratio；这里 trim_survival 全 1,
    # mask=0.4，effective_trim_n=2（4 个客户端两端各裁 1 个）
    record = _record("invariant", [0.4, 0.4, 0.4, 0.4], [10, 10, 10, 10],
                     [1, 0, 0, 0], effective_trim_n=2)
    assert effective_denominator(record) == 2.0
    out = round_exposure(record)
    assert np.isclose(out["weight_sum"], 0.4 * 4 / 2)   # = 0.8
    assert np.isclose(out["exposure_malicious"], 0.4 * 10 / 2)


def test_invariant_without_effective_trim_n_raises():
    record = _record("invariant", [0.4] * 4, [10] * 4, [1, 0, 0, 0],
                     effective_trim_n=-1)
    raised = None
    try:
        effective_denominator(record)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "effective_trim_n" in raised


# ---------------------------------------------------------------------------
# 裁剪：FLAME 的"不排除、只削弱"必须进 E
# ---------------------------------------------------------------------------
def test_clip_scale_defaults_to_one_without_flame_fields():
    record = _record("fedavg", [1.0] * 3, [1, 1, 1], [1, 0, 0])
    assert np.allclose(clip_scale(record), 1.0)


def test_clipping_reduces_exposure_even_when_nothing_is_excluded():
    """按次数计数看不见裁剪；E 必须看得见 —— 这正是整个改写的理由。"""
    extra = {"norm_before_clip": [100.0, 10.0, 10.0, 10.0],
             "norm_after_clip": [10.0, 10.0, 10.0, 10.0]}
    record = _record("flame", [1.0] * 4, [100.0, 10, 10, 10], [1, 0, 0, 0],
                     extra=extra)
    out = round_exposure(record)
    # s_0 = 10/100 = 0.1 -> E = 0.1 * 100 / 4 = 2.5，而不是 100/4 = 25
    assert np.isclose(out["exposure_malicious"], 2.5)
    assert np.isclose(out["mean_clip_scale_malicious"], 0.1)
    # 全员被选中但恶意被裁 -> 权重和 < 1
    assert out["weight_sum"] < 1.0


def test_clip_scale_length_mismatch_raises():
    extra = {"norm_before_clip": [1.0, 2.0], "norm_after_clip": [1.0, 2.0]}
    record = _record("flame", [1.0] * 4, [1] * 4, [1, 0, 0, 0], extra=extra)
    raised = None
    try:
        clip_scale(record)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "不一致" in raised


def test_oracle_exclude_absorbs_exactly_zero():
    record = _record("oracle_exclude", [0.0, 1.0, 1.0, 1.0], [99, 10, 10, 10],
                     [1, 0, 0, 0])
    out = round_exposure(record)
    assert out["exposure_malicious"] == 0.0


def test_mismatched_array_lengths_raise():
    record = _record("fedavg", [1.0] * 4, [1, 2, 3], [1, 0, 0, 0])
    raised = None
    try:
        round_exposure(record)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "长度不一致" in raised


# ---------------------------------------------------------------------------
# 落盘 -> 读回 -> 累加
# ---------------------------------------------------------------------------
def _write_npz(directory: Path, index: int, defense, influence, norms,
               malicious, extra=None, effective_trim_n=-1):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"round_{index:04d}.npz",
        round_index=index, defense=defense,
        influence=np.asarray(influence, dtype=float),
        update_norm=np.asarray(norms, dtype=float),
        is_malicious=np.asarray(malicious, dtype=bool),
        effective_trim_n=effective_trim_n,
        client_ids=np.arange(len(influence)),
        extra_json=json.dumps(extra or {}))


def test_exposure_accumulates_across_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "fedavg_attack_a0.5_s0_bad1"
        for index in (1, 2, 3):
            _write_npz(run, index, "fedavg", [1.0] * 4, [8, 4, 4, 4],
                       [1, 0, 0, 0])
        out = exposure_by_run(Path(tmp), run.name)
        assert out["n_rounds"] == 3
        assert np.isclose(out["exposure_malicious"], 3 * 8 / 4)
        assert out["exposure_is_exact"] is True
        assert out["n_malicious_observations"] == 3


def test_median_and_invariant_are_flagged_inexact():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "median_attack_a0.5_s0_bad1"
        _write_npz(run, 1, "median", [0.5] * 4, [8, 4, 4, 4], [1, 0, 0, 0])
        assert exposure_by_run(Path(tmp), run.name)["exposure_is_exact"] is False


# ---------------------------------------------------------------------------
# 与 ASR 并表
# ---------------------------------------------------------------------------
def _implantation(run_id: str, defense: str, bad: int, asr: float, acc: float,
                  n: int = 10) -> pd.DataFrame:
    return pd.DataFrame([
        {"round": (i + 1) * 10, "defense": defense, "bad_client_num": bad,
         "seed": 0, "asr_personalized_targeted": asr,
         "acc_personalized": acc}
        for i in range(n)])


def _build_case(tmp: Path, cells):
    """cells: [(defense, bad, norm, asr)] -> 埋点目录 + 植入 CSV。"""
    raw = tmp / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    root = tmp / "instr"
    for defense, bad, norm, asr in cells:
        run_id = f"{defense}_attack_a0.5_s0_bad{bad}"
        for index in range(1, 6):
            _write_npz(root / run_id, index, defense, [1.0] * 4,
                       [norm, 1, 1, 1], [1, 0, 0, 0])
        _implantation(run_id, defense, bad, asr, 0.78).to_csv(
            raw / f"exp_ij_implantation_{run_id}.csv", index=False)
    return root, str(raw / "*.csv")


def test_join_matches_on_run_id():
    with tempfile.TemporaryDirectory() as tmp:
        root, glob = _build_case(Path(tmp), [
            ("fedavg", 1, 40.0, 0.85), ("flame", 5, 8.0, 0.30)])
        merged = join_with_asr(root, glob, tail=5)
        assert len(merged) == 2
        fedavg = merged[merged["defense"] == "fedavg"].iloc[0]
        assert np.isclose(fedavg["exposure_malicious"], 5 * 40 / 4)
        assert np.isclose(fedavg["asr_mean"], 0.85)


def test_join_reports_when_nothing_matches():
    with tempfile.TemporaryDirectory() as tmp:
        root, _ = _build_case(Path(tmp), [("fedavg", 1, 40.0, 0.85)])
        other = Path(tmp) / "other"
        other.mkdir()
        _implantation("x", "flame", 5, 0.3, 0.78).to_csv(
            other / "exp_ij_implantation_flame_attack_a0.5_s9_bad5.csv",
            index=False)
        raised = None
        try:
            join_with_asr(root, str(other / "*.csv"), tail=5)
        except ValueError as error:
            raised = str(error)
        assert raised is not None and "run_id" in raised


# ---------------------------------------------------------------------------
# 判据：不能把噪声报成阈值
# ---------------------------------------------------------------------------
def _merged(pairs, defense="flame"):
    return pd.DataFrame([
        {"run_id": f"r{i}", "defense": defense, "bad_client_num": i + 1,
         "exposure_malicious": e, "asr_mean": a, "acc_mean": 0.78,
         "exposure_is_exact": True, "n_rounds": 10,
         "exposure_benign": 100.0}
        for i, (e, a) in enumerate(pairs)])


def test_too_few_cells_is_undetermined_not_a_correlation():
    result = collapse_diagnosis(_merged([(1.0, 0.1), (2.0, 0.5)]))
    assert "未能确定" in result["verdict"]
    assert "spearman_rho" not in result


def test_monotone_data_reports_collapse():
    merged = _merged([(1.0, 0.05), (2.0, 0.2), (4.0, 0.5), (8.0, 0.85),
                      (16.0, 0.95)])
    result = collapse_diagnosis(merged)
    assert result["spearman_rho"] > 0.9
    assert "坍缩" in result["verdict"]


def test_unrelated_data_does_not_claim_a_threshold():
    merged = _merged([(1.0, 0.8), (2.0, 0.1), (4.0, 0.9), (8.0, 0.2),
                      (16.0, 0.85)])
    result = collapse_diagnosis(merged)
    assert "不坍缩" in result["verdict"]


def test_threshold_needs_exactly_one_crossing():
    single = _merged([(1.0, 0.1), (2.0, 0.3), (4.0, 0.7), (8.0, 0.9)])
    value = estimate_threshold(single)
    assert value is not None and 2.0 < value < 4.0
    # 上下反复穿越 0.5 -> 不报阈值
    wobbly = _merged([(1.0, 0.1), (2.0, 0.7), (4.0, 0.2), (8.0, 0.9)])
    assert estimate_threshold(wobbly) is None


def test_plot_runs_and_passes_the_cjk_scan():
    merged = _merged([(1.0, 0.05), (2.0, 0.2), (4.0, 0.5), (8.0, 0.85)])
    merged.loc[0, "exposure_is_exact"] = False      # 混入一个近似格子
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_exposure(merged, Path(tmp) / "figs" / "e.png",
                             threshold=4.0)
        assert path.exists() and path.stat().st_size > 0
