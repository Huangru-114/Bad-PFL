"""``diag.exp_t0`` 的单元测试（窗口构造 + 单窗口度量 + 判词分支）。

判词是这个模块最危险的部分：它把一堆数翻译成一句会被写进论文的话。
所以四个分支（相当 / 几乎没动±噪声底 / 中间 / 缺参照）各有一个 case，
并且**明确检查它在该说"未能确定"的时候确实说了**。

**不需要 torch**：``exp_t0`` 只在 ``load_state`` 内部延迟导入 torch，
而这里用 ``.npz`` fixture 走 numpy 分支。
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from diag import paramspace as ps
from diag.exp_t0 import (RATIO_COMPARABLE, RATIO_NEGLIGIBLE, analyze,
                         attack_reference, available_global_rounds,
                         build_windows, displacement_verdict, global_path,
                         load_state, load_window_rows, main, random_walk_check,
                         window_report, write_rows)


def _state(conv_second: float = 4.0, linear_first: float = 1.0):
    return {
        "conv1.weight": np.array([[3.0, conv_second]]),
        "bn1.weight": np.array([1.0, 1.0]),
        "bn1.bias": np.array([0.0, 0.0]),
        "bn1.running_mean": np.array([0.0, 0.0]),
        "bn1.running_var": np.array([1.0, 1.0]),
        "bn1.num_batches_tracked": np.array(5, dtype=np.int64),
        "linear.bias": np.array([linear_first, -1.0]),
    }


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------
def test_build_windows_labels_attack_anchor_and_segments():
    windows = build_windows([140, 200, 250, 300], attack_start=140,
                            attack_stop=200)
    by_pair = {(w["round_from"], w["round_to"]): w for w in windows}
    assert set(by_pair) == {(140, 200), (200, 250), (200, 300), (250, 300)}
    # 优先级 attack > anchor > segment：[140,200) 不能被记成 segment，
    # 否则判词找不到参照尺度
    assert by_pair[(140, 200)]["kind"] == "attack"
    assert by_pair[(140, 200)]["phase"] == "attack"
    assert by_pair[(200, 250)]["kind"] == "anchor"
    assert by_pair[(250, 300)]["kind"] == "segment"
    assert all(by_pair[k]["phase"] == "clean"
               for k in [(200, 250), (200, 300), (250, 300)])


def test_build_windows_marks_windows_crossing_the_stop_round_as_mixed():
    windows = build_windows([100, 300], attack_start=140, attack_stop=200)
    assert [w["phase"] for w in windows] == ["mixed"]
    assert windows[0]["span_rounds"] == 200


def test_build_windows_is_sorted_and_deduplicated():
    windows = build_windows([200, 200, 250], attack_start=140, attack_stop=200)
    pairs = [(w["round_from"], w["round_to"]) for w in windows]
    assert pairs == [(200, 250)]        # 重复轮次不产生零长窗口


# ---------------------------------------------------------------------------
# 单窗口度量
# ---------------------------------------------------------------------------
def test_window_report_separates_trainable_from_bn_buffer():
    report = window_report(_state(4.0), _state(8.0), n_bins=4)
    summary = report["summary"]
    assert summary["n_params_total"] == 12
    assert summary["n_params_trainable"] == 8
    assert summary["n_params_bn_buffer"] == 4
    assert summary["excluded_keys"] == "bn1.num_batches_tracked"

    # 只有 conv1.weight 的第二个坐标动了 +4；‖θ_trainable‖ = sqrt(29)
    assert np.isclose(summary["trainable_l2_delta"], 4.0)
    assert np.isclose(summary["trainable_l2_base"], np.sqrt(29.0))
    assert np.isclose(summary["trainable_relative_displacement"],
                      4.0 / np.sqrt(29.0))
    # cos(θ_from, θ_to) = 45 / sqrt(29 * 77)
    assert np.isclose(summary["trainable_cos_from_to"],
                      45.0 / np.sqrt(29.0 * 77.0))
    assert np.isclose(summary["trainable_mean_abs_delta"], 0.5)
    assert np.isclose(summary["trainable_median_abs_delta"], 0.0)
    assert np.isclose(summary["trainable_max_abs_delta"], 4.0)
    # BN buffer 没动，且**单独一行**报，不混进 trainable
    assert np.isclose(summary["bn_buffer_l2_delta"], 0.0)


def test_window_report_reports_aggregated_scope_without_bn_affine():
    """FedBN 下 bn_affine 从不更新，把它算进分母只会稀释相对位移。"""
    summary = window_report(_state(4.0), _state(8.0))["summary"]
    # aggregated = weight(conv1.weight 2) + bias(linear.bias 2) = 4 个坐标
    assert summary["n_params_aggregated"] == 4
    # ‖θ_agg‖ = ‖[3,4,1,-1]‖ = sqrt(27)，位移仍是 4
    assert np.isclose(summary["aggregated_l2_base"], np.sqrt(27.0))
    assert np.isclose(summary["aggregated_l2_delta"], 4.0)
    assert np.isclose(summary["aggregated_relative_displacement"],
                      4.0 / np.sqrt(27.0))
    # 同一个位移，trainable 口径因为多了 bn 的 sqrt(2) 而被稀释
    assert (summary["aggregated_relative_displacement"]
            > summary["trainable_relative_displacement"])
    # bn_affine 单独一行报：这个 fixture 里它确实没动
    assert np.isclose(summary["bn_affine_l2_base"], np.sqrt(2.0))
    assert np.isclose(summary["bn_affine_l2_delta"], 0.0)


def test_window_report_rank_correlation_is_hand_computable():
    summary = window_report(_state(4.0), _state(8.0))["summary"]
    # |Δ| 的秩 = [4]*7 + [8]；|θ| 的秩 = [1.5,1.5,4.5,4.5,7,8,4.5,4.5]
    # 中心化后点积 14，两边范数 sqrt(14) 与 sqrt(36.5)
    assert np.isclose(summary["spearman_absdelta_abstheta"],
                      14.0 / np.sqrt(14.0 * 36.5))
    # 只有一个坐标两侧都非零（conv1.weight 的 4 -> +4），故一致率 1、分母 1
    assert summary["sign_agreement_n_compared"] == 1
    assert np.isclose(summary["sign_agreement_theta_delta"], 1.0)


def test_window_report_layer_rows_cover_only_trainable_tensors():
    report = window_report(_state(4.0), _state(8.0))
    groups = {row["group"] for row in report["layers"]}
    assert groups == {"bn1", "conv1", "linear"}
    conv = [row for row in report["layers"] if row["group"] == "conv1"][0]
    assert conv["n_params"] == 2                    # BN buffer 不在这张表里
    assert np.isclose(conv["energy_share"], 1.0)    # 位移全在 conv1 上
    bn1 = [row for row in report["layers"] if row["group"] == "bn1"][0]
    assert bn1["n_params"] == 4                     # bn1.weight + bn1.bias
    assert bn1["kind"] == "bn_affine"
    # 逐 kind 的表则**包含** BN buffer —— 坑 6 要求它被显式报出来
    kinds = {row["group"] for row in report["kinds"]}
    assert kinds == {"weight", "bias", "bn_affine", "bn_buffer"}


def test_window_report_bins_are_returned_for_every_bin():
    report = window_report(_state(4.0), _state(8.0), n_bins=4)
    assert len(report["bins"]) == 4
    assert sum(row["count"] for row in report["bins"]) == 8


# ---------------------------------------------------------------------------
# 驱动：缺轮次如实跳过
# ---------------------------------------------------------------------------
def test_analyze_skips_missing_rounds_without_interpolating():
    states = {200: _state(4.0), 400: _state(8.0)}
    windows = build_windows([200, 300, 400], attack_start=140, attack_stop=200)
    tables = analyze(states, windows, run_id="fixture")
    done = {(row["round_from"], row["round_to"]) for row in tables["windows"]}
    assert done == {(200, 400)}
    skipped = {(row["round_from"], row["round_to"])
               for row in tables["skipped"]}
    assert skipped == {(200, 300), (300, 400)}
    assert all(row["run_id"] == "fixture" for row in tables["windows"])


# ---------------------------------------------------------------------------
# 判词：四个分支
# ---------------------------------------------------------------------------
def _row(kind, phase, rel, *, start=0, end=100, aggregated=None, l2_delta=None):
    row = {"kind": kind, "phase": phase, "round_from": start, "round_to": end,
           "span_rounds": end - start,
           "trainable_relative_displacement": rel,
           "trainable_cos_from_to": 0.9}
    if aggregated is not None:
        row["aggregated_relative_displacement"] = aggregated
    if l2_delta is not None:
        row["trainable_l2_delta"] = l2_delta
    return row


def test_verdict_calls_clean_phase_comparable_and_refuses_to_kill_a():
    rows = [_row("attack", "attack", 0.20, start=140, end=200),
            _row("anchor", "clean", 0.20, start=200, end=400)]
    out = displacement_verdict(rows)
    assert np.isclose(out["ratio_clean_over_attack"], 1.0)
    assert "相当" in out["verdict"]
    # T0 测全体坐标，判不了载体的生死 —— 判词必须把这句写出来
    assert "S1" in out["verdict"]
    assert "死" not in out["verdict"].replace("生死", "")


def test_verdict_without_noise_floor_says_undetermined():
    rows = [_row("attack", "attack", 1.0, start=140, end=200),
            _row("anchor", "clean", 0.5 * RATIO_NEGLIGIBLE, start=200, end=400)]
    out = displacement_verdict(rows)
    assert "未能确定" in out["verdict"]
    assert "噪声底" in out["verdict"]
    assert "noise_floor_relative_displacement" not in out


def test_verdict_with_noise_floor_reports_the_multiple():
    rows = [_row("attack", "attack", 1.0, start=140, end=200),
            _row("anchor", "clean", 0.02, start=200, end=400)]
    out = displacement_verdict(rows, noise_floor=0.01)
    assert np.isclose(out["clean_over_noise_floor"], 2.0)
    assert "相容" in out["verdict"]
    # 相容不是证实：位移小也可能是符号来回抵消，那要 T1 才分得开
    assert "T1" in out["verdict"]


def test_verdict_middle_case_refuses_to_pick_a_side():
    middle = 0.5 * (RATIO_NEGLIGIBLE + RATIO_COMPARABLE)
    rows = [_row("attack", "attack", 1.0, start=140, end=200),
            _row("anchor", "clean", middle, start=200, end=400)]
    out = displacement_verdict(rows)
    assert "中间情形" in out["verdict"]


def test_verdict_without_attack_window_has_no_reference_scale():
    rows = [_row("anchor", "clean", 0.3, start=200, end=400)]
    out = displacement_verdict(rows)
    assert "未能确定" in out["verdict"]
    assert "ratio_clean_over_attack" not in out


def test_verdict_without_clean_window_says_so():
    rows = [_row("attack", "attack", 0.3, start=140, end=200)]
    out = displacement_verdict(rows)
    assert "未能确定" in out["verdict"]
    assert "clean_relative_displacement" not in out


def test_attack_reference_falls_back_to_a_partial_window():
    """网格上没有 attack_start 时（首跑就是这样），退到植入阶段内最长的窗口。"""
    rows = [_row("segment", "attack", 0.5, start=150, end=200),
            _row("anchor", "clean", 0.3, start=200, end=400)]
    reference, is_full = attack_reference(rows)
    assert (reference["round_from"], reference["round_to"]) == (150, 200)
    assert is_full is False                 # 只覆盖植入窗口的一部分
    # 有精确的那一格时优先用它，并标 is_full
    rows.append(_row("attack", "attack", 0.6, start=140, end=200))
    reference, is_full = attack_reference(rows)
    assert (reference["round_from"], reference["round_to"]) == (140, 200)
    assert is_full is True


def test_verdict_flags_a_partial_reference_window():
    rows = [_row("segment", "attack", 0.5, start=150, end=200),
            _row("anchor", "clean", 0.3, start=200, end=400)]
    out = displacement_verdict(rows)
    assert out["attack_window"] == "150->200"
    assert out["attack_window_is_full"] is False
    # 退化取的是更短的窗口 -> 参照尺度偏小、比值偏大，必须说出来
    assert "偏小" in out["verdict"] and "偏大" in out["verdict"]


def test_verdict_prefers_aggregated_scope_when_available():
    """有 aggregated_* 列就用它 —— trainable 的分母含 FedBN 下恒不动的 BN 仿射。"""
    rows = [_row("attack", "attack", 0.06, start=140, end=200, aggregated=0.12),
            _row("anchor", "clean", 0.08, start=200, end=400, aggregated=0.16)]
    out = displacement_verdict(rows)
    assert out["displacement_scope"] == "aggregated"
    assert np.isclose(out["clean_relative_displacement"], 0.16)
    assert np.isclose(out["ratio_clean_over_attack"], 0.16 / 0.12)
    # 没有该列时回退，并如实标注用的是哪一份
    bare = displacement_verdict([_row("attack", "attack", 0.06, start=140,
                                      end=200),
                                 _row("anchor", "clean", 0.08, start=200,
                                      end=400)])
    assert bare["displacement_scope"] == "trainable"
    assert np.isclose(bare["ratio_clean_over_attack"], 0.08 / 0.06)


def test_verdict_reports_a_span_matched_ratio():
    """整段干净 vs 一小段植入，一半的比值是时长差造成的 —— 同时长的也要报。"""
    rows = [_row("segment", "attack", 0.10, start=150, end=200),
            _row("anchor", "clean", 0.08, start=200, end=250),
            _row("anchor", "clean", 0.20, start=200, end=400)]
    out = displacement_verdict(rows)
    assert out["clean_window"] == "200->400"
    assert out["clean_window_span_matched"] == "200->250"
    assert np.isclose(out["ratio_span_matched"], 0.08 / 0.10)
    assert np.isclose(out["ratio_clean_over_attack"], 0.20 / 0.10)
    assert "同时长比较" in out["verdict"]


# ---------------------------------------------------------------------------
# 随机游走判据
# ---------------------------------------------------------------------------
def test_random_walk_check_detects_orthogonal_increments():
    """3-4-5：两段正交时实测恰好等于平方和开根，比值为 1。"""
    rows = [_row("anchor", "clean", 0.1, start=200, end=400, l2_delta=5.0),
            _row("anchor", "clean", 0.1, start=200, end=300, l2_delta=3.0),
            _row("segment", "clean", 0.1, start=300, end=400, l2_delta=4.0)]
    out = random_walk_check(rows)
    assert out["window"] == "200->400"
    assert out["n_segments"] == 2
    assert np.isclose(out["quadrature_l2_delta"], 5.0)
    assert np.isclose(out["linear_l2_delta"], 7.0)
    assert np.isclose(out["observed_over_quadrature"], 1.0)
    assert np.isclose(out["linear_over_quadrature"], 1.4)


def test_random_walk_check_tiles_by_rounds_not_by_kind():
    """锚定起点后的第一格被标成 anchor（优先级），只收 segment 会永远铺不满。"""
    rows = [_row("anchor", "clean", 0.1, start=200, end=300, l2_delta=5.0),
            _row("anchor", "clean", 0.1, start=200, end=250, l2_delta=3.0),
            _row("segment", "clean", 0.1, start=250, end=300, l2_delta=4.0)]
    assert random_walk_check(rows)["n_segments"] == 2


def test_random_walk_check_returns_none_on_a_gap():
    """铺不满就不比，**不用可用的段硬凑**。"""
    rows = [_row("anchor", "clean", 0.1, start=200, end=400, l2_delta=5.0),
            _row("segment", "clean", 0.1, start=300, end=400, l2_delta=4.0)]
    assert random_walk_check(rows) is None


def test_verdict_prefers_the_longest_clean_window():
    rows = [_row("attack", "attack", 1.0, start=140, end=200),
            _row("anchor", "clean", 0.9, start=200, end=250),
            _row("anchor", "clean", 0.3, start=200, end=400)]
    out = displacement_verdict(rows)
    assert out["clean_window"] == "200->400"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def test_write_rows_unions_columns_and_leaves_gaps_empty():
    """缺的列留**空**，不是 0、不是 "N/A"（CLAUDE.md 铁律 5）。"""
    rows = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    with tempfile.TemporaryDirectory() as tmp:
        path = write_rows(rows, Path(tmp) / "sub" / "out.csv")
        with open(path, encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
    assert list(table[0].keys()) == ["a", "b", "c"]
    assert table[0]["c"] == ""
    assert table[1]["b"] == ""


def test_load_window_rows_round_trips_and_rebuilds_aggregated():
    """写出的 CSV 读回来要能直接进判词；老 CSV 靠 layers 精确重建 aggregated。"""
    states = {200: _state(4.0), 400: _state(8.0)}
    windows = build_windows([200, 400], attack_start=140, attack_stop=200)
    tables = analyze(states, windows, run_id="fixture")
    truth = tables["windows"][0]["aggregated_relative_displacement"]

    with tempfile.TemporaryDirectory() as tmp:
        windows_csv = write_rows(tables["windows"], Path(tmp) / "w.csv")
        layers_csv = write_rows(tables["layers"], Path(tmp) / "l.csv")
        # 模拟首跑的老 CSV：把 aggregated_* 列整列删掉
        stripped = [{k: v for k, v in row.items()
                     if not k.startswith("aggregated_")}
                    for row in tables["windows"]]
        old_csv = write_rows(stripped, Path(tmp) / "old.csv")

        rows = load_window_rows(windows_csv)
        assert rows[0]["round_from"] == 200 and rows[0]["span_rounds"] == 200
        assert np.isclose(rows[0]["aggregated_relative_displacement"], truth)

        without = load_window_rows(old_csv)
        assert "aggregated_relative_displacement" not in without[0]
        rebuilt = load_window_rows(old_csv, layers_csv)
        assert np.isclose(rebuilt[0]["aggregated_relative_displacement"], truth)
        # 判词在两种输入下都能跑，只是标注的 scope 不同
        assert displacement_verdict(without)["displacement_scope"] == "trainable"
        assert (displacement_verdict(rebuilt)["displacement_scope"]
                == "aggregated")


def test_load_window_rows_leaves_aggregated_missing_when_kinds_absent():
    """layers 里缺 kind 行的窗口就保持没有 aggregated_*——不猜、不补。"""
    states = {200: _state(4.0), 400: _state(8.0)}
    tables = analyze(states, build_windows([200, 400], 140, 200),
                     run_id="fixture")
    stripped = [{k: v for k, v in row.items()
                 if not k.startswith("aggregated_")}
                for row in tables["windows"]]
    only_weight = [row for row in tables["layers"]
                   if row.get("scope") == "kind" and row["group"] == "weight"]
    with tempfile.TemporaryDirectory() as tmp:
        rows = load_window_rows(write_rows(stripped, Path(tmp) / "w.csv"),
                                write_rows(only_weight, Path(tmp) / "l.csv"))
    assert "aggregated_relative_displacement" not in rows[0]


def test_load_state_reads_npz_without_torch():
    state = _state()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global.npz"
        np.savez(path, **state)
        loaded = load_state(path)
    assert set(loaded) == set(state)
    assert np.allclose(loaded["conv1.weight"], state["conv1.weight"])
    # 走完整链路：读回来的字典能直接进 paramspace
    index = ps.build_index(loaded)
    assert index.excluded == ["bn1.num_batches_tracked"]


def test_load_state_reports_missing_path():
    try:
        load_state(Path("/nonexistent/round_0200/global.pt"))
    except FileNotFoundError as error:
        assert "global.pt" in str(error)
    else:
        raise AssertionError("缺文件时应当报 FileNotFoundError")


def _write_run(root: Path, rounds, drift):
    """造一个 ``round_XXXX/global.npz`` 的假 run。

    ``drift[r]`` 是该轮次 conv1.weight 第二个坐标的值 —— 位移全部集中在这一个
    坐标上，所以每个窗口的 ‖Δθ‖ 都能手算。
    """
    for round_index in rounds:
        directory = root / f"round_{round_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "global.npz", **_state(drift[round_index]))
    return root


def test_global_path_falls_back_to_npz_and_lists_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp), [200, 400], {200: 4.0, 400: 8.0})
        assert available_global_rounds(root) == [200, 400]
        assert global_path(root, 200).name == "global.npz"
        assert global_path(root, 300) is None


def test_main_runs_end_to_end_on_an_npz_fixture():
    """整条 CLI（读盘 -> 四张 CSV -> 判词 json）在没有 torch 的机器上跑通。"""
    rounds = [140, 200, 250, 300]
    drift = {140: 4.0, 200: 8.0, 250: 8.5, 300: 8.6}
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp) / "attack_fixture", rounds, drift)
        out_dir = Path(tmp) / "out"
        assert main(["--ckpt-dir", str(root), "--attack-start", "140",
                     "--attack-stop", "200", "--out-dir", str(out_dir),
                     "--n-bins", "4"]) == 0

        with open(out_dir / "t0_windows.csv", encoding="utf-8") as handle:
            windows = {(row["round_from"], row["round_to"]): row
                       for row in csv.DictReader(handle)}
        # 植入窗口 [140,200) 位移 4；干净锚定窗口 [200,400] 位移 0.6
        assert np.isclose(float(windows[("140", "200")]["trainable_l2_delta"]),
                          4.0)
        assert np.isclose(float(windows[("200", "300")]["trainable_l2_delta"]),
                          0.6)
        assert windows[("140", "200")]["kind"] == "attack"
        assert windows[("200", "300")]["phase"] == "clean"

        for name in ("t0_layers.csv", "t0_energy.csv", "t0_bins.csv",
                     "t0_verdict.json"):
            assert (out_dir / name).exists(), name
        verdict = json.loads((out_dir / "t0_verdict.json")
                             .read_text(encoding="utf-8"))
        assert verdict["run_id"] == "attack_fixture"
        assert verdict["noise_floor_relative_displacement"] is None
        # rel_clean/rel_attack ≈ (0.6/‖θ200‖)/(4/‖θ140‖) ≈ 0.1 -> 中间情形
        assert "中间情形" in verdict["verdict"]["verdict"]
