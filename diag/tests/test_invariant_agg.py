"""``diag.invariant_agg`` 的单元测试。

重点全部放在**算错了也不会报错**的地方：

- 伪梯度方向（陷阱 1）—— 搞反会让防御变成加速攻击，而 loss 曲线不会立刻异常；
- 掩码是否真的把分歧维度按住；
- trimmed mean 是否在**每个维度独立**排序（按向量整体排序也跑得通，但错）；
- FedBN 私有 key 与非浮点 key 是否绕开了 sign/trim；
- 是否写坏了传入的 state_dict（``agg_avg`` 的别名 bug，陷阱 2）。
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn

from diag.fedbn import is_fedbn_private_key, split_keys
from diag.invariant_agg import (RoundStats, and_mask, invariant_aggregate,
                                pseudo_gradients, sign_consistency,
                                trim_count, trimmed_mean,
                                use_invariant_aggregator)

from resnet import get_resnet
from server import BasicServer


def _state(values):
    """``{"linear.weight": tensor}`` 形态的极简 state_dict。"""
    return {"linear.weight": torch.tensor(values, dtype=torch.float32)}


# ---------------------------------------------------------------------------
# 伪梯度方向（陷阱 1）
# ---------------------------------------------------------------------------
def test_pseudo_gradient_is_old_minus_new():
    w_prev = _state([10.0, 10.0])
    clients = [_state([7.0, 12.0])]
    stacked = pseudo_gradients(w_prev, clients, ["linear.weight"])
    # g = w_prev - w_i：客户端把参数**降低**了 -> 伪梯度为正
    assert torch.equal(stacked["linear.weight"][0],
                       torch.tensor([3.0, -2.0]))


def test_reduces_to_fedavg_without_mask_or_trim():
    """τ<0 全放行、α=0 不裁剪时，必须精确等于 FedAvg。

    这一个断言同时钉死**两件**事：伪梯度方向（旧减新）与更新方向
    （w_t = w_prev − ḡ）。任何一个搞反，结果都会变成
    ``w_prev + (w_prev − mean)``，即朝远离客户端的方向走 —— 那正是
    "防御变成加速攻击"的形态。
    """
    w_prev = _state([0.0, 0.0, 0.0])
    clients = [_state([1.0, 2.0, 3.0]), _state([3.0, 4.0, 5.0]),
               _state([5.0, 6.0, 7.0])]
    update, _ = invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    expected = torch.stack([c["linear.weight"] for c in clients]).mean(dim=0)
    assert torch.allclose(update["linear.weight"], expected, atol=1e-6)


def test_wrong_direction_would_move_away_from_clients():
    """反向实现会把全局模型推到客户端均值的另一侧 —— 这里固定住正确的一侧。"""
    w_prev = _state([0.0])
    clients = [_state([2.0]), _state([4.0])]
    update, _ = invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    result = float(update["linear.weight"][0])
    assert result == 3.0                      # 客户端均值
    assert abs(result - 3.0) < abs(-3.0 - 3.0)   # 不是 w_prev - (mean - w_prev)


# ---------------------------------------------------------------------------
# 符号一致性与掩码
# ---------------------------------------------------------------------------
def test_sign_consistency_unanimous_and_split():
    stacked = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
    consistency, zeros = sign_consistency(stacked)
    assert torch.allclose(consistency, torch.tensor([1.0, 0.0]))
    assert int(zeros.sum()) == 0


def test_sign_consistency_counts_zeros_in_denominator_by_default():
    stacked = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    counted, zeros = sign_consistency(stacked, "count")
    assert torch.allclose(counted, torch.tensor([0.5]))    # |2| / 4
    assert int(zeros.sum()) == 2

    abstained, _ = sign_consistency(stacked, "abstain")
    assert torch.allclose(abstained, torch.tensor([1.0]))  # |2| / 2


def test_sign_consistency_of_all_zero_dimension_is_zero_not_one():
    """全零维度上没有任何客户端表达意见，不该被当成"完全一致"而放行。"""
    stacked = torch.zeros(4, 1)
    for policy in ("count", "abstain"):
        consistency, _ = sign_consistency(stacked, policy)
        assert float(consistency[0]) == 0.0


def test_and_mask_is_strictly_greater_than_tau():
    consistency = torch.tensor([0.2, 0.2001, 0.19])
    mask = and_mask(consistency, tau=0.2)
    assert torch.equal(mask, torch.tensor([0.0, 1.0, 0.0]))


def test_one_over_n_mode_scales_the_whole_update():
    """论文字面版本会把结果整体缩放 1/N —— 判为笔误，这里固定住其行为。"""
    consistency = torch.ones(3)
    binary = and_mask(consistency, tau=0.0, mode="binary", n_clients=10)
    literal = and_mask(consistency, tau=0.0, mode="one_over_n", n_clients=10)
    assert torch.allclose(literal, binary / 10.0)


def test_mask_freezes_dimensions_where_clients_disagree():
    """分歧维度必须停在 w_prev；一致维度必须动。"""
    w_prev = _state([100.0, 100.0])
    # 第 0 维四个客户端符号一致，第 1 维两两相反
    clients = [_state([90.0, 90.0]), _state([92.0, 110.0]),
               _state([94.0, 90.0]), _state([96.0, 110.0])]
    update, stats = invariant_aggregate(w_prev, clients, tau=0.5, trim_alpha=0.0)
    assert float(update["linear.weight"][1]) == 100.0      # 被掩掉，原地不动
    assert float(update["linear.weight"][0]) != 100.0
    assert abs(stats.mask_keep_ratio - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Trimmed mean
# ---------------------------------------------------------------------------
def test_trim_count_uses_ceiling():
    assert trim_count(10, 0.25) == 3       # ⌈2.5⌉
    assert trim_count(10, 0.1) == 1
    assert trim_count(10, 0.0) == 0


def test_trimmed_mean_removes_outliers_per_dimension():
    """必须**逐维度**排序。按向量整体排序也能跑，但结果是错的。"""
    stacked = torch.tensor([[100.0, 1.0],      # 第 0 维的上离群
                            [1.0, 2.0],
                            [2.0, 3.0],
                            [3.0, -100.0]])    # 第 1 维的下离群
    mean, effective = trimmed_mean(stacked, alpha=0.25)   # 每端去 1，剩 2
    assert effective == 2
    assert torch.allclose(mean, torch.tensor([2.5, 1.5]))


def test_trimmed_mean_effective_n_matches_the_spec_warning():
    """N=10、α=0.25 时只剩 4 个 —— 陷阱 7 要求如实报告。"""
    stacked = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    _, effective = trimmed_mean(stacked, alpha=0.25)
    assert effective == 4
    _, effective = trimmed_mean(stacked, alpha=0.1)
    assert effective == 8


def test_trimmed_mean_raises_when_everything_is_trimmed():
    stacked = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    try:
        trimmed_mean(stacked, alpha=0.5)      # 每端去 2，剩 0
    except ValueError as exc:
        assert "全部去光" in str(exc)
    else:
        raise AssertionError("trim 耗尽必须报错，不能静默退化")


# ---------------------------------------------------------------------------
# key 分类：FedBN 私有 / 非浮点
# ---------------------------------------------------------------------------
def test_split_keys_on_a_real_resnet():
    shared_float, shared_nonfloat, private = split_keys(
        get_resnet(size=10).state_dict())
    assert shared_float and private
    assert all(not is_fedbn_private_key(k) for k in shared_float)
    assert any(k.startswith("bn1.") for k in private)
    assert any("shortcut.1." in k for k in private)
    # num_batches_tracked 的 key 里含 "bn"，已归入私有 -> 共享侧不该有非浮点
    assert shared_nonfloat == []


def test_fedbn_private_keys_bypass_mask_and_trim():
    """私有 key 走简单平均：即使符号完全分歧也不该被掩成 w_prev。"""
    w_prev = {"linear.weight": torch.tensor([100.0]),
              "bn1.running_mean": torch.tensor([100.0])}
    clients = [{"linear.weight": torch.tensor([90.0]),
                "bn1.running_mean": torch.tensor([90.0])},
               {"linear.weight": torch.tensor([110.0]),
                "bn1.running_mean": torch.tensor([110.0])}]
    update, stats = invariant_aggregate(w_prev, clients, tau=0.9, trim_alpha=0.0)
    assert float(update["linear.weight"][0]) == 100.0        # 共享参数被掩掉
    assert float(update["bn1.running_mean"][0]) == 100.0     # 平均恰好也是 100
    assert stats.n_fedbn_private_keys == 1
    assert stats.n_shared_float_keys == 1


def test_nonfloat_shared_keys_fall_back_to_mean_and_are_reported():
    w_prev = {"linear.weight": torch.tensor([0.0]),
              "counter": torch.tensor([0], dtype=torch.int64)}
    clients = [{"linear.weight": torch.tensor([2.0]),
                "counter": torch.tensor([4], dtype=torch.int64)},
               {"linear.weight": torch.tensor([4.0]),
                "counter": torch.tensor([6], dtype=torch.int64)}]
    update, stats = invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    assert update["counter"].dtype == torch.int64      # dtype 保留
    assert int(update["counter"][0]) == 5
    assert stats.nonfloat_fallback_keys == ["counter"]


# ---------------------------------------------------------------------------
# 不写坏输入（陷阱 2：agg_avg 的别名 bug）
# ---------------------------------------------------------------------------
def test_does_not_mutate_incoming_state_dicts():
    w_prev = _state([0.0, 0.0])
    clients = [_state([1.0, 2.0]), _state([3.0, 4.0])]
    snapshots = [{k: v.clone() for k, v in state.items()} for state in clients]
    invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    for state, snapshot in zip(clients, snapshots):
        for key in state:
            assert torch.equal(state[key], snapshot[key]), key


def test_does_not_mutate_w_prev():
    w_prev = _state([5.0, 5.0])
    snapshot = {k: v.clone() for k, v in w_prev.items()}
    clients = [_state([1.0, 2.0]), _state([3.0, 4.0])]
    invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    assert torch.equal(w_prev["linear.weight"], snapshot["linear.weight"])


# ---------------------------------------------------------------------------
# 统计量
# ---------------------------------------------------------------------------
def test_zero_sign_ratio_is_reported():
    w_prev = _state([0.0, 0.0])
    # 第 1 维两个客户端都没动 -> 伪梯度恰为 0
    clients = [_state([1.0, 0.0]), _state([2.0, 0.0])]
    _, stats = invariant_aggregate(w_prev, clients, tau=-1.0, trim_alpha=0.0)
    assert abs(stats.zero_sign_ratio - 0.5) < 1e-9    # 2 / (2 维 × 2 客户端)


def test_round_stats_row_is_csv_friendly():
    row = RoundStats(nonfloat_fallback_keys=["a", "b"]).to_row()
    assert row["nonfloat_fallback_keys"] == "a;b"
    assert isinstance(row["mask_keep_ratio"], float)


# ---------------------------------------------------------------------------
# 接入真实的 BasicServer
# ---------------------------------------------------------------------------
def test_use_invariant_aggregator_preserves_fedbn_behaviour():
    """替换聚合器后，FedBN 仍然把私有 key pop 掉，全局模型的 BN 一位不变。"""
    from pfl import use_fedbn

    torch.manual_seed(0)
    server = BasicServer(get_resnet(size=10))
    use_fedbn(server)
    before = {k: v.clone() for k, v in server.global_model.state_dict().items()}

    clients = []
    for offset in (1, 2, 3):
        torch.manual_seed(offset)
        model = get_resnet(size=10)
        clients.append(model.state_dict())

    handle = use_invariant_aggregator(server, tau=-1.0, trim_alpha=0.0,
                                      verify_w_prev=False)
    try:
        server.agg_and_update(clients)
    finally:
        handle.detach()

    after = server.global_model.state_dict()
    for key in before:
        if is_fedbn_private_key(key):
            assert torch.equal(before[key], after[key]), f"BN key {key} 被改动了"
        else:
            assert not torch.equal(before[key], after[key]), f"共享 key {key} 没更新"
    assert len(handle.history) == 1
    assert handle.history[0].n_clients == 3


def test_detach_restores_the_original_aggregator():
    server = BasicServer(get_resnet(size=10))
    original = server.agg_and_update
    handle = use_invariant_aggregator(server, tau=0.2, trim_alpha=0.0,
                                      verify_w_prev=False)
    assert server.agg_and_update is not original
    handle.detach()
    assert server.agg_and_update == original


def test_verify_w_prev_catches_a_stale_global_model():
    """断言本身要有效：故意在轮次中途改动全局模型，聚合必须中止。"""
    from event_emitter import fl_event_emitter

    server = BasicServer(get_resnet(size=10))
    handle = use_invariant_aggregator(server, tau=-1.0, trim_alpha=0.0,
                                      verify_w_prev=True)
    try:
        fl_event_emitter.emit(event_name="on_round_begin", cur_round=1,
                              server=server, clients=[])
        with torch.no_grad():
            server.global_model.linear.weight[0, 0] += 1.0   # 中途被改
        try:
            server.agg_and_update([server.global_model.state_dict()])
        except RuntimeError as exc:
            assert "w_(t-1)" in str(exc)
        else:
            raise AssertionError("全局模型在轮次中途被改动时必须报错")
    finally:
        handle.detach()
