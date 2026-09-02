"""``diag.paramspace`` 的单元测试。

全部用**手工构造、可手算**的小 state_dict：这个模块下游要支撑
"干净训练到底动了多少"这个判断，一个静默算错的范数会直接变成一句错的结论，
而错的结论看上去和对的一样正常。所以每个数都在注释里写出手算过程。

**不需要 torch**（``paramspace`` 是纯 numpy），因此这些 case 在没有 torch 的
机器上也必须全绿。
"""

from __future__ import annotations

import numpy as np

from diag import paramspace as ps


# ---------------------------------------------------------------------------
# fixture：一个带 BN 的迷你模型
# ---------------------------------------------------------------------------
def _state():
    """6 个浮点张量 + 1 个 int 计数器，逐个覆盖四种 kind。

    键名排序后的顺序（``build_index`` 用的就是这个顺序）：

        bn1.bias            [0, 0]    bn_affine
        bn1.running_mean    [0, 0]    bn_buffer
        bn1.running_var     [1, 1]    bn_buffer
        bn1.weight          [1, 1]    bn_affine
        conv1.weight        [3, 4]    weight
        linear.bias         [1, -1]   bias
        bn1.num_batches_tracked  (int64) -> 被排除
    """
    return {
        "conv1.weight": np.array([[3.0, 4.0]]),
        "bn1.weight": np.array([1.0, 1.0]),
        "bn1.bias": np.array([0.0, 0.0]),
        "bn1.running_mean": np.array([0.0, 0.0]),
        "bn1.running_var": np.array([1.0, 1.0]),
        "bn1.num_batches_tracked": np.array(5, dtype=np.int64),
        "linear.bias": np.array([1.0, -1.0]),
    }


# ---------------------------------------------------------------------------
# 分类与索引
# ---------------------------------------------------------------------------
def test_bn_affine_is_detected_structurally_not_by_name():
    """判据是"父模块下有 running_mean"，不是名字里有没有 'bn'。"""
    keys = ["norm7.weight", "norm7.running_mean", "conv.weight"]
    assert ps.parameter_kind("norm7.weight", keys) == ps.KIND_BN_AFFINE
    assert ps.parameter_kind("norm7.running_mean", keys) == ps.KIND_BN_BUFFER
    # 名字里带 bn 但没有 running_mean 兄弟 -> 是普通 weight，不是 BN 仿射
    keys2 = ["bn_like.weight"]
    assert ps.parameter_kind("bn_like.weight", keys2) == ps.KIND_WEIGHT


def test_index_orders_by_key_and_excludes_integer_buffers():
    index = ps.build_index(_state())
    assert index.keys == ["bn1.bias", "bn1.running_mean", "bn1.running_var",
                          "bn1.weight", "conv1.weight", "linear.bias"]
    assert index.kinds == [ps.KIND_BN_AFFINE, ps.KIND_BN_BUFFER,
                           ps.KIND_BN_BUFFER, ps.KIND_BN_AFFINE,
                           ps.KIND_WEIGHT, ps.KIND_BIAS]
    # int64 的 num_batches_tracked 被排除，但**如实登记**在 excluded 里
    assert index.excluded == ["bn1.num_batches_tracked"]
    assert index.n_params == 12                     # 6 个张量 × 2 个元素
    assert index.offsets == [0, 2, 4, 6, 8, 10, 12]
    assert index.layers[:2] == ["bn1", "bn1"]
    assert index.blocks == ["bn1"] * 4 + ["conv1", "linear"]


def test_flatten_follows_index_order():
    vector = ps.flatten(_state())
    expected = [0.0, 0.0,      # bn1.bias
                0.0, 0.0,      # bn1.running_mean
                1.0, 1.0,      # bn1.running_var
                1.0, 1.0,      # bn1.weight
                3.0, 4.0,      # conv1.weight
                1.0, -1.0]     # linear.bias
    assert np.allclose(vector, expected)
    assert vector.dtype == np.float64


def test_trainable_mask_excludes_bn_buffers():
    index = ps.build_index(_state())
    mask = index.kind_mask(ps.TRAINABLE_KINDS)
    # 8 个可训练坐标 = bn1.bias(2) + bn1.weight(2) + conv1.weight(2)
    #                  + linear.bias(2)；running_mean / running_var 被排除
    assert int(mask.sum()) == 8
    assert not mask[2:6].any()          # running_mean / running_var 的位置
    assert int(index.kind_mask((ps.KIND_BN_BUFFER,)).sum()) == 4


# ---------------------------------------------------------------------------
# 位移
# ---------------------------------------------------------------------------
def test_displacement_and_relative_scale_are_hand_computable():
    state_a = _state()
    state_b = _state()
    state_b["conv1.weight"] = np.array([[3.0, 8.0]])       # 只动一个坐标 +4
    theta, delta, index = ps.displacement(state_a, state_b)

    trainable = index.kind_mask(ps.TRAINABLE_KINDS)
    assert np.isclose(ps.l2(delta[trainable]), 4.0)
    # ‖θ‖² = bn1.bias 0 + bn1.weight 2 + conv1.weight 25 + linear.bias 2 = 29
    assert np.isclose(ps.l2(theta[trainable]), np.sqrt(29.0))
    assert np.isclose(ps.relative_displacement(delta[trainable],
                                               theta[trainable]),
                      4.0 / np.sqrt(29.0))
    # BN buffer 没动
    assert np.isclose(ps.l2(delta[index.kind_mask((ps.KIND_BN_BUFFER,))]), 0.0)


def test_displacement_refuses_mismatched_key_sets():
    """键集合不一致 -> 报错，**不取交集**。静默取交集会把事故变成正常的表。"""
    state_a = _state()
    state_b = _state()
    del state_b["linear.bias"]
    try:
        ps.displacement(state_a, state_b)
    except ValueError as error:
        assert "linear.bias" in str(error)
    else:
        raise AssertionError("键集合不一致时应当报错")


# ---------------------------------------------------------------------------
# 向量度量
# ---------------------------------------------------------------------------
def test_cosine_is_nan_for_zero_vector_not_zero():
    assert np.isclose(ps.cosine([1.0, 2.0], [2.0, 4.0]), 1.0)
    assert np.isclose(ps.cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
    assert np.isclose(ps.cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)
    # 零向量与谁都不成角度：未定义 -> nan，而不是"正交"
    assert np.isnan(ps.cosine([0.0, 0.0], [1.0, 1.0]))


def test_sign_agreement_skips_zeros_and_reports_denominator():
    x = np.array([1.0, -1.0, 0.0, 2.0])
    y = np.array([1.0, 1.0, 5.0, -2.0])
    out = ps.sign_agreement(x, y)
    # 有效坐标 = 0, 1, 3（第 2 个 x 为 0 被跳过）；只有第 0 个同号
    assert out["n_compared"] == 3
    assert out["n_skipped"] == 1
    assert np.isclose(out["rate"], 1.0 / 3.0)
    assert np.isclose(out["conflict_rate"], 2.0 / 3.0)


def test_sign_agreement_is_nan_when_nothing_comparable():
    out = ps.sign_agreement(np.zeros(3), np.ones(3))
    assert out["n_compared"] == 0
    assert np.isnan(out["rate"])


def test_topk_energy_matches_hand_computation():
    delta = np.array([1.0, 2.0, 3.0, 4.0])          # 总能量 30
    rows = {row["fraction"]: row for row in ps.topk_energy(delta, (0.25, 0.5))}
    # k=0.25 -> ceil(1) = 1 个坐标（|Δ| 最大的 4）-> 16/30
    assert rows[0.25]["n_top"] == 1
    assert np.isclose(rows[0.25]["energy_share"], 16.0 / 30.0)
    assert np.isclose(rows[0.25]["concentration"], (16.0 / 30.0) / 0.25)
    # k=0.5 -> 2 个坐标（4 与 3）-> 25/30
    assert rows[0.5]["n_top"] == 2
    assert np.isclose(rows[0.5]["energy_share"], 25.0 / 30.0)


def test_group_energy_shares_sum_to_one():
    state_a = _state()
    state_b = _state()
    state_b["conv1.weight"] = np.array([[3.0, 8.0]])
    state_b["linear.bias"] = np.array([4.0, -1.0])          # +3
    theta, delta, index = ps.displacement(state_a, state_b)
    rows = {row["group"]: row for row in
            ps.group_energy(delta, index, by="block", base=theta)}
    # 能量：conv1 16，linear 9，bn1 0 -> 占比 16/25 与 9/25
    assert np.isclose(rows["conv1"]["energy_share"], 16.0 / 25.0)
    assert np.isclose(rows["linear"]["energy_share"], 9.0 / 25.0)
    assert np.isclose(rows["bn1"]["energy_share"], 0.0)
    assert np.isclose(sum(r["energy_share"] for r in rows.values()), 1.0)
    # linear.bias 的 ‖θ‖ = sqrt(2)，位移 3 -> 相对位移 3/sqrt(2)
    assert np.isclose(rows["linear"]["relative_displacement"],
                      3.0 / np.sqrt(2.0))


def test_group_energy_rejects_mismatched_index():
    """向量已被掩码筛过、索引却是全量 -> 长度对不上，必须报错而不是广播。"""
    state = _state()
    index = ps.build_index(state)
    try:
        ps.group_energy(np.zeros(5), index, by="layer")
    except ValueError as error:
        assert "长度" in str(error)
    else:
        raise AssertionError("长度不一致时应当报错")


# ---------------------------------------------------------------------------
# 秩与相关
# ---------------------------------------------------------------------------
def test_average_rank_averages_ties():
    ranks = ps.average_rank(np.array([10.0, 20.0, 20.0, 5.0]))
    # 5 -> 1；10 -> 2；两个 20 并列占 3 与 4 -> 各 3.5
    assert np.allclose(ranks, [2.0, 3.5, 3.5, 1.0])


def test_rank_quantile_of_uniform_subset_is_half():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    out = ps.rank_quantile(values, np.array([True, False, False, True]))
    # 分位 = (rank - 0.5)/n = 0.125 / 0.375 / 0.625 / 0.875，取首尾 -> 均值 0.5
    assert np.isclose(out["mean_quantile"], 0.5)
    assert out["n_selected"] == 2 and out["n_total"] == 4
    lowest = ps.rank_quantile(values, np.array([True, True, False, False]))
    assert np.isclose(lowest["mean_quantile"], 0.25)


def test_pearson_and_spearman_against_hand_values():
    assert np.isclose(ps.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
    assert np.isclose(ps.pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)
    # 常数序列的相关系数未定义 -> nan（不是 0）
    assert np.isnan(ps.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    # 秩 x = 1,2,3,4；秩 y = 1,3,2,4 -> 中心化后点积 4，两边范数各 sqrt(5)
    assert np.isclose(ps.spearman([1.0, 2.0, 3.0, 4.0],
                                  [10.0, 30.0, 20.0, 40.0]), 0.8)


def test_spearman_is_invariant_to_monotone_transform():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 5.0, 3.0, 4.0])
    assert np.isclose(ps.spearman(x, y), ps.spearman(np.exp(x), y))


# ---------------------------------------------------------------------------
# 分箱
# ---------------------------------------------------------------------------
def test_binned_curve_uniform_edges_and_last_bin_is_closed():
    rows = ps.binned_curve(np.array([0.0, 1.0, 2.0, 3.0]),
                           np.array([0.0, 10.0, 20.0, 30.0]),
                           n_bins=2, mode="uniform")
    assert [row["count"] for row in rows] == [2, 2]
    assert np.isclose(rows[0]["y_mean"], 5.0)
    assert np.isclose(rows[1]["y_mean"], 25.0)      # 最大值 3.0 落在末箱内
    assert np.isclose(rows[1]["x_high"], 3.0)


def test_binned_curve_quantile_mode_splits_evenly():
    rows = ps.binned_curve(np.array([1.0, 2.0, 3.0, 4.0]),
                           np.array([1.0, 1.0, 5.0, 5.0]),
                           n_bins=2, mode="quantile")
    assert [row["count"] for row in rows] == [2, 2]
    assert np.isclose(rows[0]["y_mean"], 1.0)
    assert np.isclose(rows[1]["y_mean"], 5.0)


def test_binned_curve_keeps_empty_bins():
    """空箱保留（count=0，统计量 nan）—— 合并会让 x 轴不再等距。"""
    rows = ps.binned_curve(np.array([0.0, 0.0, 10.0]), np.array([1.0, 1.0, 2.0]),
                           n_bins=5, mode="uniform")
    assert len(rows) == 5
    assert rows[0]["count"] == 2 and rows[-1]["count"] == 1
    assert all(row["count"] == 0 for row in rows[1:-1])
    assert np.isnan(rows[1]["y_mean"])


# ---------------------------------------------------------------------------
# 逐层表
# ---------------------------------------------------------------------------
def test_layer_table_separates_scaling_from_rewriting():
    """cos(θ, Δθ) 区分"整体缩放"与"方向改写" —— 逐层表的关键列。"""
    state_a = _state()
    state_b = _state()
    state_b["conv1.weight"] = np.array([[6.0, 8.0]])        # 纯缩放 ×2
    state_b["linear.bias"] = np.array([-1.0, -1.0])         # 方向被改写
    theta, delta, index = ps.displacement(state_a, state_b)
    rows = {row["group"]: row for row in
            ps.layer_table(theta, delta, index, by="layer")}
    assert np.isclose(rows["conv1"]["cos_theta_delta"], 1.0)   # Δ ∥ θ
    # linear: θ=(1,-1), Δ=(-2,0) -> cos = -2 / (sqrt(2)*2) = -1/sqrt(2)
    assert np.isclose(rows["linear"]["cos_theta_delta"], -1.0 / np.sqrt(2.0))
    assert rows["bn1"]["kind"] == "bn_affine+bn_buffer"        # 混合，不挑代表
