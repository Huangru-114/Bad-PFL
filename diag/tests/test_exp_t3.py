"""``diag.exp_t3`` 的单元测试（方向族 + 标定 + 判词）。

T3 的全部说服力挂在**标定对不对**上：如果"随机方向、同幅度"这句里的"同幅度"
是错的，整个对照就没了，而图照样画得出来、数照样是有限值。所以每个族的不变量
都单独钉死：

- ``zero`` 恒为零向量；
- ``real`` 与真实漂移的 cos 恰好 1；
- ``shuffled`` / ``sign_flipped`` **逐层**范数与真实漂移完全相同；
- ``gaussian_layer_matched`` 的逐层剖面偏差 ≈ 0，而 ``gaussian_global`` 不是；
- 定标后 achieved 必须等于 target。

判词部分单独钉「ACC 已经坏了就不许给 ASR 结论」这条 —— 那是这个实验唯一能
立住的读法。

**不需要 torch**：``exp_t3`` 只在 GPU 分支里延迟导入 torch。
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from diag import paramspace as ps
from diag.exp_t3 import (ACC_GUARD, DEFAULT_MULTIPLIERS, FAMILIES,
                         apply_perturbation, build_recipes, calibrate,
                         flatness_verdict, main, make_direction,
                         perturbation_profile, scale_to_relative)


def _index_and_vectors():
    """两层的迷你子空间：``a.weight`` 4 个坐标、``b.weight`` 2 个坐标。

    θ = [1,1,1,1 | 3,4]        ‖θ_a‖ = 2, ‖θ_b‖ = 5
    Δ = [0.1,-0.1,0.2,-0.2 | 1,0]  ‖Δ_a‖ = sqrt(0.1)，‖Δ_b‖ = 1
    """
    state = {"a.weight": np.ones(4), "b.weight": np.array([3.0, 4.0])}
    index = ps.build_index(state)
    theta = ps.flatten(state, index)
    delta = np.array([0.1, -0.1, 0.2, -0.2, 1.0, 0.0])
    return index, theta, delta


def _layer_norms(vector, index):
    labels = np.asarray(index.group_labels("layer"))
    return {label: float(np.linalg.norm(vector[labels == label]))
            for label in sorted(set(labels.tolist()))}


# ---------------------------------------------------------------------------
# 方向族的不变量
# ---------------------------------------------------------------------------
def test_zero_family_is_exactly_zero():
    index, theta, delta = _index_and_vectors()
    direction = make_direction("zero", delta, theta, index, seed=0)
    assert np.allclose(direction, 0.0)
    # 定标一个零向量不该报错，也不该造出非零
    assert np.allclose(scale_to_relative(direction, theta, 0.5), 0.0)


def test_real_family_is_the_drift_itself():
    index, theta, delta = _index_and_vectors()
    direction = make_direction("real", delta, theta, index, seed=0)
    assert np.allclose(direction, delta)
    assert np.isclose(ps.cosine(direction, delta), 1.0)
    # 返回的是副本，改它不能污染输入
    direction[0] = 999.0
    assert delta[0] == 0.1


def test_shuffled_preserves_layer_norms_but_moves_coordinates():
    index, theta, delta = _index_and_vectors()
    direction = make_direction("shuffled", delta, theta, index, seed=0)
    before, after = _layer_norms(delta, index), _layer_norms(direction, index)
    for layer in before:
        assert np.isclose(before[layer], after[layer])      # 层内置换，范数不变
    # 幅度的多重集合逐层不变
    assert sorted(np.abs(direction[:4])) == sorted(np.abs(delta[:4]))


def test_sign_flipped_preserves_magnitudes_coordinate_by_coordinate():
    index, theta, delta = _index_and_vectors()
    direction = make_direction("sign_flipped", delta, theta, index, seed=0)
    assert np.allclose(np.abs(direction), np.abs(delta))    # 逐坐标幅度不变
    assert not np.allclose(direction, delta)                # 但方向变了


def test_gaussian_layer_matched_matches_the_per_layer_profile():
    index, theta, delta = _index_and_vectors()
    direction = make_direction("gaussian_layer_matched", delta, theta, index,
                               seed=0)
    before, after = _layer_norms(delta, index), _layer_norms(direction, index)
    for layer in before:
        assert np.isclose(before[layer], after[layer])
    profile = perturbation_profile(direction, theta, delta, index)
    assert profile["layer_profile_max_rel_dev"] < 1e-12


def test_gaussian_global_does_not_match_the_per_layer_profile():
    """这正是它与 layer_matched 的区别 —— 必须能被读数区分开，不能靠嘴说。"""
    index, theta, delta = _index_and_vectors()
    direction = make_direction("gaussian_global", delta, theta, index, seed=0)
    vector = scale_to_relative(direction, theta,
                               ps.relative_displacement(delta, theta))
    matched = scale_to_relative(
        make_direction("gaussian_layer_matched", delta, theta, index, seed=0),
        theta, ps.relative_displacement(delta, theta))
    global_dev = perturbation_profile(vector, theta, delta,
                                      index)["layer_profile_max_rel_dev"]
    matched_dev = perturbation_profile(matched, theta, delta,
                                       index)["layer_profile_max_rel_dev"]
    assert global_dev > matched_dev
    assert matched_dev < 1e-12


def test_random_families_are_reproducible_from_the_seed():
    """配方不存向量、只存 seed，所以"同 seed 同向量"是硬要求。"""
    index, theta, delta = _index_and_vectors()
    for family in ("gaussian_global", "gaussian_layer_matched", "shuffled",
                   "sign_flipped"):
        first = make_direction(family, delta, theta, index, seed=7)
        again = make_direction(family, delta, theta, index, seed=7)
        other = make_direction(family, delta, theta, index, seed=8)
        assert np.allclose(first, again), family
        assert not np.allclose(first, other), family


def test_make_direction_rejects_a_mismatched_index():
    index, theta, delta = _index_and_vectors()
    try:
        make_direction("shuffled", delta[:3], theta[:3], index, seed=0)
    except ValueError as error:
        assert "长度" in str(error)
    else:
        raise AssertionError("向量与索引长度不一致时应当报错")


def test_unknown_family_raises():
    index, theta, delta = _index_and_vectors()
    try:
        make_direction("brownian", delta, theta, index, seed=0)
    except ValueError as error:
        assert "brownian" in str(error)
    else:
        raise AssertionError("未知方向族应当报错")


# ---------------------------------------------------------------------------
# 定标
# ---------------------------------------------------------------------------
def test_scale_to_relative_hits_the_target_exactly():
    index, theta, delta = _index_and_vectors()
    for family in ("gaussian_global", "shuffled", "real"):
        direction = make_direction(family, delta, theta, index, seed=1)
        vector = scale_to_relative(direction, theta, 0.25)
        # ‖θ‖ = sqrt(4 + 25) = sqrt(29)
        assert np.isclose(ps.l2(vector), 0.25 * np.sqrt(29.0)), family
        assert np.isclose(
            perturbation_profile(vector, theta, delta,
                                 index)["achieved_relative_displacement"],
            0.25), family


def test_profile_reports_cos_with_the_real_drift():
    index, theta, delta = _index_and_vectors()
    scaled = scale_to_relative(make_direction("real", delta, theta, index, 0),
                               theta, 0.3)
    profile = perturbation_profile(scaled, theta, delta, index)
    assert np.isclose(profile["cos_with_real_drift"], 1.0)
    assert profile["n_layers"] == 2


# ---------------------------------------------------------------------------
# 配方表
# ---------------------------------------------------------------------------
def test_build_recipes_gives_zero_exactly_one_cell():
    recipes = build_recipes(seeds=(0, 1, 2))
    zeros = [r for r in recipes if r["family"] == "zero"]
    assert len(zeros) == 1                       # 幅度与 seed 对它没有意义
    assert zeros[0]["target_relative"] == 0.0


def test_build_recipes_does_not_repeat_deterministic_families():
    """`real` 是确定性的，多个 seed 会得到同一个向量 —— 重复评估纯属浪费机时。"""
    recipes = build_recipes(families=("real", "shuffled"),
                            multipliers=(1.0, 2.0), seeds=(0, 1, 2))
    real = [r for r in recipes if r["family"] == "real"]
    shuffled = [r for r in recipes if r["family"] == "shuffled"]
    assert len(real) == 2                        # 两个幅度各一格
    assert len(shuffled) == 6                    # 两个幅度 × 三个 seed


def test_build_recipes_scales_target_by_the_base():
    recipes = build_recipes(families=("real",), multipliers=(0.5, 2.0),
                            base_relative=0.156)
    assert np.isclose(recipes[0]["target_relative"], 0.078)
    assert np.isclose(recipes[1]["target_relative"], 0.312)


def test_build_recipes_rejects_unknown_family():
    try:
        build_recipes(families=("teleport",))
    except ValueError as error:
        assert "teleport" in str(error)
    else:
        raise AssertionError("未知方向族应当报错")


# ---------------------------------------------------------------------------
# 加回 state_dict
# ---------------------------------------------------------------------------
def test_apply_perturbation_only_touches_indexed_keys():
    """BN 不在 aggregated 索引里 -> 客户端自己的 BN 必须原样保留（FedBN）。"""
    index, theta, delta = _index_and_vectors()
    state = {"a.weight": np.ones(4), "b.weight": np.array([3.0, 4.0]),
             "bn.running_mean": np.array([7.0, 7.0]),
             "bn.num_batches_tracked": np.array(5, dtype=np.int64)}
    out = apply_perturbation(state, delta, index)
    assert np.allclose(out["a.weight"], np.array([1.1, 0.9, 1.2, 0.8]))
    assert np.allclose(out["b.weight"], np.array([4.0, 4.0]))
    assert np.allclose(out["bn.running_mean"], 7.0)          # 没被碰
    assert out["bn.num_batches_tracked"].dtype == np.int64   # 整型 buffer 不变
    assert set(out) == set(state)                            # 键集合不变


def test_apply_perturbation_keeps_float32_as_float32():
    """checkpoint 是 float32；加完扰动还得能塞回 load_state_dict。"""
    state = {"a.weight": np.ones(4, dtype=np.float32)}
    index = ps.build_index(state)
    out = apply_perturbation(state, np.full(4, 0.5), index)
    assert out["a.weight"].dtype == np.float32
    assert np.allclose(out["a.weight"], 1.5)


# ---------------------------------------------------------------------------
# 判词
# ---------------------------------------------------------------------------
def _cell(family, multiplier, asr, acc, seed=0):
    return {"family": family, "multiplier": multiplier, "seed": seed,
            "asr": asr, "acc": acc}


def test_verdict_needs_the_zero_baseline():
    out = flatness_verdict([_cell("gaussian_global", 1.0, 0.4, 0.7)])
    assert "未能确定" in out["verdict"]
    assert "baseline_acc" not in out


def test_verdict_calls_a_wide_basin_when_asr_survives():
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 1.0, 0.39, 0.69),
            _cell("shuffled", 1.0, 0.40, 0.68),
            _cell("real", 1.0, 0.38, 0.69)]
    out = flatness_verdict(rows)
    assert np.isclose(out["baseline_asr"], 0.41)
    assert out["largest_usable_multiplier"] == 1.0
    assert "宽盆" in out["verdict"]
    # 宽盆意味着参数空间的定点操作够不到它 —— 这是防御选型的直接后果，要说出来
    assert "剪枝" in out["verdict"]


def test_verdict_calls_the_direction_special_when_random_kills_asr():
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 1.0, 0.10, 0.68),
            _cell("shuffled", 1.0, 0.12, 0.69)]
    out = flatness_verdict(rows)
    assert "方向特殊" in out["verdict"]
    assert out["asr_retained_fraction"] < 0.4


def test_verdict_refuses_to_read_asr_once_acc_is_broken():
    """把模型打坏也能让 ASR 掉 —— 那不是"扰动能消后门"。"""
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 4.0, 0.02, 0.11)]
    out = flatness_verdict(rows)
    assert "未能确定" in out["verdict"]
    assert out["per_multiplier"]["4"]["acc_ok"] is False
    assert "模型本身已经坏了" in out["verdict"]


def test_verdict_uses_the_largest_magnitude_that_acc_survives():
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 1.0, 0.40, 0.69),
            _cell("gaussian_layer_matched", 2.0, 0.39, 0.67),
            _cell("gaussian_layer_matched", 4.0, 0.05, 0.20)]
    out = flatness_verdict(rows)
    assert out["largest_usable_multiplier"] == 2.0      # 4.0 被 ACC 闸门挡掉
    assert out["per_multiplier"]["4"]["acc_ok"] is False
    assert out["per_multiplier"]["2"]["acc_ok"] is True
    assert "宽盆" in out["verdict"]


def test_verdict_middle_case_refuses_to_pick_a_side():
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 1.0, 0.25, 0.69)]
    out = flatness_verdict(rows)
    assert "中间情形" in out["verdict"]


def test_verdict_excludes_real_from_the_random_control_mean():
    """`real` 是正对照，不能混进"随机方向"的均值里。"""
    rows = [_cell("zero", 0.0, 0.41, 0.70),
            _cell("gaussian_layer_matched", 1.0, 0.10, 0.69),
            _cell("real", 1.0, 0.40, 0.69)]
    out = flatness_verdict(rows)
    assert np.isclose(out["per_multiplier"]["1"]["asr_random_mean"], 0.10)
    assert np.isclose(out["per_multiplier"]["1"]["asr_real"], 0.40)


def test_defaults_are_the_ones_the_plan_pins():
    assert ACC_GUARD == 0.05
    assert DEFAULT_MULTIPLIERS == (0.5, 1.0, 2.0, 4.0)
    assert FAMILIES[0] == "zero"                 # 自检锚点排第一


# ---------------------------------------------------------------------------
# 整链路（.npz fixture，不需要 torch）
# ---------------------------------------------------------------------------
def _write_run(root: Path, drift: float):
    """两个轮次的假 run：conv1.weight 与 linear.weight 各挪一点。"""
    for round_index, shift in ((200, 0.0), (400, drift)):
        directory = root / f"round_{round_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "global.npz", **{
            "conv1.weight": np.arange(12, dtype=np.float64).reshape(4, 3) + shift,
            "bn1.weight": np.ones(4), "bn1.bias": np.zeros(4),
            "bn1.running_mean": np.zeros(4), "bn1.running_var": np.ones(4),
            "bn1.num_batches_tracked": np.array(3, dtype=np.int64),
            "linear.weight": np.ones((2, 4)) + shift,
            "linear.bias": np.full(2, 0.5),
        })
    return root


def test_calibrate_ignores_bn_and_measures_the_real_drift():
    """标定必须在 aggregated 子空间上做 —— BN 在 FedBN 下根本不动。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp) / "run", drift=0.1)
        rows, drift = calibrate(root, 200, 400,
                                build_recipes(families=("zero", "real"),
                                              multipliers=(1.0,),
                                              base_relative=0.2))
    # aggregated = conv1.weight(12) + linear.weight(8) + linear.bias(2) = 22
    assert drift["n_params_aggregated"] == 22
    # 每个坐标挪 0.1（bias 不动）-> ‖Δ‖ = 0.1*sqrt(20)
    assert np.isclose(drift["drift_l2"], 0.1 * np.sqrt(20.0))
    real = [row for row in rows if row["family"] == "real"][0]
    assert np.isclose(real["cos_with_real_drift"], 1.0)
    assert np.isclose(real["achieved_relative_displacement"], 0.2)


def test_build_mode_runs_end_to_end_and_the_self_checks_hold():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp) / "run", drift=0.1)
        out_dir = Path(tmp) / "out"
        assert main(["--mode", "build", "--ckpt-dir", str(root),
                     "--multipliers", "1,2", "--seeds", "0,1",
                     "--out-dir", str(out_dir)]) == 0
        with open(out_dir / "t3_manifest.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert (out_dir / "t3_drift.json").exists()

    for row in rows:
        # achieved 必须等于 target，一格都不能差
        assert np.isclose(float(row["achieved_relative_displacement"]),
                          float(row["target_relative"]), atol=1e-12), row
        dev = float(row["layer_profile_max_rel_dev"])
        if row["family"] in ("gaussian_layer_matched", "shuffled",
                             "sign_flipped", "real"):
            assert dev < 1e-12, row               # 形状与真实漂移一致
        elif row["family"] == "gaussian_global":
            assert dev > 1e-6, row                # 形状不一致，正是它的定义
    assert np.isclose(float([r for r in rows
                             if r["family"] == "real"][0]
                            ["cos_with_real_drift"]), 1.0)


def test_eval_dry_run_lists_recipes_without_touching_torch():
    """dry-run 要能在没有 torch / 没有数据的机器上跑，否则没法先看格子数。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp) / "run", drift=0.1)
        assert main(["--mode", "eval", "--ckpt-dir", str(root),
                     "--multipliers", "1", "--seeds", "0",
                     "--out-dir", str(Path(tmp) / "out")]) == 0
        assert not (Path(tmp) / "out" / "t3_results.csv").exists()


def test_calibrate_reports_a_missing_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_run(Path(tmp) / "run", drift=0.1)
        try:
            calibrate(root, 200, 350, build_recipes(families=("zero",)))
        except FileNotFoundError as error:
            assert "round_0350" in str(error)
        else:
            raise AssertionError("缺快照时应当报 FileNotFoundError")
