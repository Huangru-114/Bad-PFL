"""``diag.instrumentation`` 的单元测试（实验 I / J 共用的埋点层）。

盯的是并表时才会暴露、单看一张表毫无异常的错误：

- Δ = w−w_prev 与 g = w_prev−w 两种符号约定必须给出**完全相同**的指标；
- Median 的"中间两位"与 trimmed-mean 的"存活"必须共用同一个名次窗口原语；
- 0 恶意的轮次必须被**排除并计数**，而不是当成 AUC=0.5；
- 不做客户端级决策的防御不得被算出 TPR/FPR；
- 逐轮 AUC 取平均 ≠ 池化 AUC（前者是陷阱 2 明令禁止的）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from diag.instrumentation import (DEFENSES_WITH_CLIENT_DECISION,
                                  RoundRecord, RoundRecorder,
                                  compose_influence, influence_summary,
                                  load_round_records, median_influence,
                                  pooled_auc, rank_distribution,
                                  rank_window_fraction, round_signals,
                                  trim_survival)


def _state(values):
    return {"linear.weight": torch.tensor(values, dtype=torch.float32)}


def _round(is_malicious, l2, influence=None, round_index=1):
    n = len(is_malicious)
    return {
        "is_malicious": np.asarray(is_malicious, dtype=bool),
        "l2_to_median": np.asarray(l2, dtype=float),
        "influence": np.asarray(influence if influence is not None
                                else np.ones(n), dtype=float),
        "round_index": round_index,
    }


# ---------------------------------------------------------------------------
# 符号约定：Δ 与 g 必须等价
# ---------------------------------------------------------------------------
def test_sign_convention_does_not_change_any_signal():
    """实验 J 用 Δ=w−w_prev，实验 I 用 g=w_prev−w。两者必须给出同一组数字。

    把 w_prev 与客户端整体镜像（w' = 2*w_prev − w）就等价于把所有 g 取负，
    三个指标都必须逐位不变 —— 否则两个实验的表并不到一起。
    """
    w_prev = _state([0.0, 0.0, 0.0])
    clients = [_state([1.0, 2.0, 3.0]), _state([-1.0, 5.0, 0.0]),
               _state([4.0, -2.0, 1.0])]
    mirrored = [_state([-v for v in c["linear.weight"].tolist()])
                for c in clients]

    left = round_signals(w_prev, clients)
    right = round_signals(w_prev, mirrored)
    for key in ("update_norm", "l2_to_median", "cos_to_median"):
        assert np.allclose(left[key], right[key], atol=1e-9), key


def test_signals_ignore_fedbn_private_keys():
    """BN 坐标必须不进入距离 —— FedBN 从不聚合它们，而全局模型的 BN 停在初始化。"""
    w_prev = {"linear.weight": torch.zeros(2),
              "bn1.running_mean": torch.zeros(2)}
    same = [{"linear.weight": torch.tensor([1.0, 1.0]),
             "bn1.running_mean": torch.tensor([0.0, 0.0])},
            {"linear.weight": torch.tensor([1.0, 1.0]),
             "bn1.running_mean": torch.tensor([500.0, -500.0])}]
    signals = round_signals(w_prev, same)
    # 两个客户端的共享参数完全一致 -> 范数必须相同，BN 的巨大差异不得泄漏进来
    assert np.allclose(signals["update_norm"][0], signals["update_norm"][1])
    assert np.allclose(signals["l2_to_median"], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 名次窗口：Median 与 trim 共用同一个原语
# ---------------------------------------------------------------------------
def test_rank_window_fraction_counts_each_coordinate_once():
    stacked = torch.tensor([[3.0], [1.0], [2.0], [0.0]])
    # 升序名次：client 3(0.0)=0, client 1(1.0)=1, client 2(2.0)=2, client 0(3.0)=3
    fraction = rank_window_fraction(stacked, 1, 3)
    assert torch.allclose(fraction,
                          torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64))
    # 全窗口 -> 每个客户端都是 1.0；总和 = N
    assert torch.allclose(rank_window_fraction(stacked, 0, 4),
                          torch.ones(4, dtype=torch.float64))


def test_median_influence_picks_the_middle_two_for_even_n():
    stacked = torch.tensor([[10.0], [1.0], [2.0], [3.0]])
    # 升序：1(名次0), 2(名次1), 3(名次2), 10(名次3) -> 中间两位是 client 2 与 3
    influence = median_influence(stacked)
    assert torch.allclose(influence,
                          torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64))


def test_median_influence_sums_to_two_coordinates_worth():
    """N 偶数时每个坐标恰好有 2 个客户端进入中间两位 -> 各客户端比例之和 = 2。"""
    torch.manual_seed(0)
    stacked = torch.randn(10, 137)
    assert abs(float(median_influence(stacked).sum()) - 2.0) < 1e-9


def test_trim_survival_matches_the_aggregator_window():
    """trim_k=3、N=10 时每个坐标存活 4 个 -> 比例之和 = 4（陷阱 7 的那个 4）。"""
    torch.manual_seed(0)
    stacked = torch.randn(10, 137)
    assert abs(float(trim_survival(stacked, 3).sum()) - 4.0) < 1e-9
    assert abs(float(trim_survival(stacked, 1).sum()) - 8.0) < 1e-9


def test_trim_survival_of_an_outlier_client_is_zero():
    stacked = torch.tensor([[100.0, 100.0], [1.0, 1.0], [2.0, 2.0],
                            [3.0, 3.0], [4.0, 4.0], [-100.0, -100.0]])
    survival = trim_survival(stacked, trim_k=1)
    assert float(survival[0]) == 0.0     # 恒为最大 -> 每个坐标都被裁
    assert float(survival[5]) == 0.0     # 恒为最小
    assert float(survival[2]) == 1.0


# ---------------------------------------------------------------------------
# 影响力的组装
# ---------------------------------------------------------------------------
def test_compose_influence_covers_every_defense_in_the_spec_table():
    assert np.allclose(compose_influence("fedavg", n_clients=3), 1.0)
    assert np.allclose(
        compose_influence("median", n_clients=2,
                          median_influence_values=np.array([0.3, 0.7])),
        [0.3, 0.7])
    assert np.allclose(
        compose_influence("invariant", n_clients=2,
                          trim_survival_rate=np.array([0.8, 0.4]),
                          mask_keep_ratio=0.5),
        [0.4, 0.2])
    for defense in DEFENSES_WITH_CLIENT_DECISION:
        assert np.allclose(
            compose_influence(defense, n_clients=3,
                              selected=np.array([True, False, True])),
            [1.0, 0.0, 1.0])


def test_compose_influence_refuses_to_guess_missing_parts():
    for kwargs in ({"defense": "median"},
                   {"defense": "invariant", "trim_survival_rate": np.ones(2)},
                   {"defense": "flame"}):
        try:
            compose_influence(n_clients=2, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} 缺分量时必须报错，不能凑一个数出来")


def test_compose_influence_rejects_unknown_defense():
    try:
        compose_influence("krum_but_typo", n_clients=2)
    except ValueError as exc:
        assert "未知的防御" in str(exc)
    else:
        raise AssertionError("未知防御必须报错")


# ---------------------------------------------------------------------------
# 名次分布与池化 AUC
# ---------------------------------------------------------------------------
def test_rank_distribution_excludes_and_counts_rounds_without_malicious():
    records = [_round([False] * 10, np.arange(10.0)),
               _round([True] + [False] * 9, [9.0] + [1.0] * 9),
               _round([False] * 10, np.arange(10.0))]
    result = rank_distribution(records, "l2_to_median")
    assert result["n_rounds_no_malicious"] == 2
    assert result["n_rounds_used"] == 1
    assert result["n_malicious_observations"] == 1
    # 恶意客户端的 l2 最大 -> 降序名次 1
    assert result["ranks"].tolist() == [1]
    assert result["pooled_auc"] == 1.0


def test_pooled_auc_is_not_the_mean_of_per_round_aucs():
    """两轮恶意数不同：池化 AUC 按 pair 加权，与逐轮 AUC 取平均不同。

    陷阱 2 禁止报逐轮 AUC 的平均值，这个测试固定住两者确实不等。
    """
    # 第 1 轮：1 恶意（最大）-> 轮内 AUC = 1.0，pair 数 1×3=3
    # 第 2 轮：3 恶意（全部最小）-> 轮内 AUC = 0.0，pair 数 3×1=3
    records = [_round([True, False, False, False], [9.0, 1.0, 2.0, 3.0]),
               _round([True, True, True, False], [1.0, 2.0, 3.0, 9.0])]
    result = rank_distribution(records, "l2_to_median")
    assert abs(result["pooled_auc"] - 0.5) < 1e-9
    assert result["n_malicious_observations"] == 4      # 1 + 3
    # 逐轮 AUC 取平均在这个构造上恰好也是 0.5，所以再看名次分布：
    assert sorted(result["ranks"].tolist()) == [1, 2, 3, 4]


def test_pooled_auc_handles_ties_as_half_credit():
    records = [_round([True, False], [5.0, 5.0])]
    assert abs(pooled_auc(records, "l2_to_median") - 0.5) < 1e-9


def test_rank_distribution_skips_rounds_that_are_all_malicious():
    records = [_round([True, True], [1.0, 2.0])]
    result = rank_distribution(records, "l2_to_median")
    assert result["n_rounds_all_malicious"] == 1
    assert result["n_rounds_used"] == 0
    assert np.isnan(result["pooled_auc"])


def test_rank_histogram_length_matches_client_count():
    records = [_round([True] + [False] * 9, [9.0] + [1.0] * 9)]
    result = rank_distribution(records, "l2_to_median")
    assert len(result["histogram"]) == 10
    assert result["histogram"][0] == 1


# ---------------------------------------------------------------------------
# 影响力汇总
# ---------------------------------------------------------------------------
def test_influence_summary_reports_the_ratio_and_the_counts():
    """均值池化的是 (轮次, 客户端) 观测，不是"逐轮均值再平均"。

    这一点必须固定住：按轮平均会让"只有 1 个恶意客户端的轮次"与"有 3 个的
    轮次"权重相同，而恶意数服从超几何分布，权重本就该不同。
    """
    records = [_round([True, False, False], [1, 2, 3], influence=[0.2, 0.8, 0.8]),
               _round([False, False, False], [1, 2, 3], influence=[0.9, 0.9, 0.9])]
    summary = influence_summary(records)
    benign_mean = (0.8 + 0.8 + 0.9 + 0.9 + 0.9) / 5      # = 0.86
    assert abs(summary["influence_malicious_mean"] - 0.2) < 1e-9
    assert abs(summary["influence_benign_mean"] - benign_mean) < 1e-9
    assert abs(summary["influence_ratio"] - 0.2 / benign_mean) < 1e-9
    assert summary["n_rounds_no_malicious"] == 1
    assert summary["n_malicious_observations"] == 1
    assert summary["n_benign_observations"] == 5


def test_influence_ratio_near_one_means_the_defense_did_nothing():
    records = [_round([True, False], [1, 2], influence=[0.5, 0.5])]
    assert abs(influence_summary(records)["influence_ratio"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 落盘与读回
# ---------------------------------------------------------------------------
def test_recorder_round_trip():
    record = RoundRecord(
        round_index=7, client_ids=np.arange(3),
        is_malicious=np.array([True, False, False]),
        update_norm=np.array([1.0, 2.0, 3.0]),
        l2_to_median=np.array([0.5, 0.6, 0.7]),
        cos_to_median=np.array([0.9, 0.8, 0.7]),
        influence=np.array([0.1, 0.2, 0.3]),
        trim_survival_rate=np.array([0.4, 0.5, 0.6]),
        mask_keep_ratio=0.42, zero_sign_ratio=0.01, effective_trim_n=4,
        n_malicious_in_round=1, n_selected=3, defense="invariant",
        extra={"note": "smoke"})
    with tempfile.TemporaryDirectory() as tmp:
        recorder = RoundRecorder(tmp, "run_a")
        path = recorder.write(record)
        assert path.name == "round_0007.npz"
        loaded = load_round_records(tmp, "run_a")
        assert len(loaded) == 1
        assert loaded[0]["extra"]["note"] == "smoke"
        assert np.allclose(loaded[0]["influence"], [0.1, 0.2, 0.3])
        assert int(loaded[0]["effective_trim_n"]) == 4


def test_recorder_orders_rounds_numerically():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = RoundRecorder(tmp, "run_b")
        for index in (10, 2, 100):
            recorder.write(RoundRecord(
                round_index=index, client_ids=np.arange(2),
                is_malicious=np.array([True, False]),
                update_norm=np.zeros(2), l2_to_median=np.zeros(2),
                cos_to_median=np.zeros(2), influence=np.zeros(2)))
        rounds = [int(r["round_index"]) for r in load_round_records(tmp, "run_b")]
        assert rounds == [2, 10, 100]


# ---------------------------------------------------------------------------
# 与聚合器的一致性
# ---------------------------------------------------------------------------
def test_trim_survival_agrees_with_the_aggregator_trim():
    """埋点算的存活率必须与 ``invariant_agg.trimmed_mean`` 实际裁掉的是同一批。"""
    from diag.invariant_agg import trim_count, trimmed_mean

    torch.manual_seed(0)
    stacked = torch.randn(10, 1)
    k = trim_count(10, 0.25)
    _, effective = trimmed_mean(stacked, 0.25)
    survival = trim_survival(stacked, k)
    assert int(survival.sum().round()) == effective        # 4
    # 最大与最小的三个客户端必须存活率为 0
    order = torch.argsort(stacked.flatten())
    for index in order[:k].tolist() + order[-k:].tolist():
        assert float(survival[index]) == 0.0
