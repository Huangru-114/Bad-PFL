"""``diag.defenses`` 与 ``diag.exp_ij`` 的单元测试。

盯的是并表与解读时才会暴露的错误：

- FedAvg 必须与仓库原本的 ``agg_avg`` 数值等价（否则"无防御"这一组不同源）；
- 消融的三个变体必须是**同一个聚合器的参数设置**，不是三份实现；
- Gram 矩阵导出的距离/余弦必须与直接展平计算一致（流式实现最容易在这里错）；
- FLAME 在 N 很小时 HDBSCAN 会把全部点判为噪声，回退必须被记录；
- ``median`` / ``invariant`` 绝不能被算出 TPR/FPR。
"""

from __future__ import annotations

import numpy as np
import torch

from diag.defenses import (ABLATION_VARIANTS, DEFENSES, Flame, InvariantAggregator,
                           build_defense, gram_matrix, use_defense)
from diag.exp_ij import ablation_verdict, detection_table
from diag.fedbn import split_keys

from resnet import get_resnet
from server import BasicServer, agg_avg


def _state(values, key="linear.weight"):
    return {key: torch.tensor(values, dtype=torch.float32)}


def _clients(rows):
    return [_state(row) for row in rows]


# ---------------------------------------------------------------------------
# FedAvg 必须与原仓库等价
# ---------------------------------------------------------------------------
def test_fedavg_matches_repo_agg_avg():
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # agg_avg 会原地改写 state_dicts[0]，所以喂它一份拷贝
    reference = agg_avg([{k: v.clone() for k, v in c.items()} for c in clients])

    update, outcome = build_defense("fedavg")(w_prev, clients)
    assert torch.allclose(update["linear.weight"], reference["linear.weight"])
    assert np.allclose(outcome.influence, 1.0)
    assert outcome.selected is None          # 不做客户端级决策


def test_defenses_do_not_mutate_incoming_state_dicts():
    """``agg_avg`` 的别名 bug：任何原地写都会污染第一个客户端上传的 dict。"""
    w_prev = _state([0.0, 0.0])
    for name in DEFENSES:
        clients = _clients([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        snapshots = [{k: v.clone() for k, v in c.items()} for c in clients]
        build_defense(name, tau=0.0, trim_alpha=0.25)(w_prev, clients)
        for state, snapshot in zip(clients, snapshots):
            for key in state:
                assert torch.equal(state[key], snapshot[key]), f"{name}: {key}"


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------
def test_median_takes_the_coordinatewise_median():
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 100.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    update, outcome = build_defense("median")(w_prev, clients)
    # g = w_prev - w = [-1,-100], [-2,-2], [-3,-3], [-4,-4]
    # 逐坐标中位数（torch 取偶数时的下中位）= [-3, -4]；w_t = 0 - (-3, -4)
    assert torch.allclose(update["linear.weight"], torch.tensor([3.0, 4.0]))
    assert outcome.selected is None
    # 极端离群的客户端 0 在第 1 维不该进中间两位
    assert outcome.influence[0] < outcome.influence[1]


# ---------------------------------------------------------------------------
# 消融变体 = 参数设置
# ---------------------------------------------------------------------------
def test_ablation_variants_are_parameterisations_of_one_aggregator():
    for name in ABLATION_VARIANTS:
        rule = build_defense(name, tau=0.2, trim_alpha=0.25)
        assert isinstance(rule, InvariantAggregator), name
        assert rule.name == name
    assert build_defense("invariant_mask_only").trim_alpha == 0.0
    assert build_defense("invariant_trim_only").tau == -1.0
    # 组合保留调用方给的参数
    combined = build_defense("invariant", tau=0.3, trim_alpha=0.1)
    assert (combined.tau, combined.trim_alpha) == (0.3, 0.1)


def test_mask_only_variant_does_not_trim():
    """trim_alpha=0 -> 离群客户端仍然参与平均，存活率恒为 1。"""
    w_prev = _state([0.0])
    clients = _clients([[100.0], [1.0], [2.0], [3.0]])
    _, outcome = build_defense("invariant_mask_only", tau=-1.0)(w_prev, clients)
    assert np.allclose(outcome.trim_survival_rate, 1.0)
    assert outcome.effective_trim_n == 4


def test_trim_only_variant_keeps_every_dimension():
    """tau=-1 -> 掩码全放行，保留率必须是 1.0。"""
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, -1.0], [2.0, 2.0], [3.0, -3.0], [4.0, 4.0]])
    _, outcome = build_defense("invariant_trim_only", trim_alpha=0.25)(
        w_prev, clients)
    assert abs(outcome.mask_keep_ratio - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Gram 矩阵：流式实现必须与直接展平一致
# ---------------------------------------------------------------------------
def test_gram_matrix_matches_the_flattened_computation():
    torch.manual_seed(0)
    w_prev = {"a.weight": torch.randn(4, 3), "b.weight": torch.randn(5)}
    clients = [{"a.weight": torch.randn(4, 3), "b.weight": torch.randn(5)}
               for _ in range(4)]
    keys, _, _ = split_keys(clients[0])

    gram = gram_matrix(w_prev, clients, keys)
    flat = torch.stack([
        torch.cat([(w_prev[k] - c[k]).flatten() for k in keys]).double()
        for c in clients])
    assert torch.allclose(gram, flat @ flat.T, atol=1e-9)


# ---------------------------------------------------------------------------
# Multi-Krum
# ---------------------------------------------------------------------------
def test_multi_krum_excludes_the_outlier():
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 1.0], [1.1, 1.1], [0.9, 0.9], [1.05, 0.95],
                        [50.0, -50.0]])      # 最后一个是明显的离群点
    _, outcome = build_defense("multi_krum", n_byzantine=1)(w_prev, clients)
    assert outcome.selected is not None       # 做客户端级决策
    assert not bool(outcome.selected[4])
    assert outcome.influence[4] == 0.0
    assert outcome.influence[:4].sum() > 0


def test_multi_krum_influence_is_binary():
    w_prev = _state([0.0])
    clients = _clients([[float(i)] for i in range(6)])
    _, outcome = build_defense("multi_krum", n_byzantine=1)(w_prev, clients)
    assert set(np.unique(outcome.influence)) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# FLAME
# ---------------------------------------------------------------------------
def test_flame_clips_by_the_median_norm_and_records_both_norms():
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [80.0, 0.0]])
    _, outcome = build_defense("flame", seed=0)(w_prev, clients)
    extra = outcome.extra
    assert extra["clipped"][3] is True                    # 大范数被裁
    assert extra["norm_after_clip"][3] < extra["norm_before_clip"][3]
    assert abs(extra["norm_after_clip"][3] - extra["clip_bound"]) < 1e-6


def test_flame_records_the_hdbscan_fallback_on_small_n():
    """N=4 时 min_cluster_size=3，HDBSCAN 很可能全判噪声 —— 回退必须被记录。

    这里不断言回退一定触发（取决于 sklearn 版本的实现细节），只断言
    这三个字段一定存在且自洽：不回退时必须有一个非空的最大簇。
    """
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [-1.0, 5.0]])
    _, outcome = build_defense("flame", seed=0)(w_prev, clients)
    extra = outcome.extra
    for key in ("flame_n_clusters", "flame_n_noise", "flame_max_cluster_size",
                "flame_fallback_triggered"):
        assert key in extra, key
    if extra["flame_fallback_triggered"]:
        assert bool(outcome.selected.all())               # 回退 = 全部接受
    else:
        assert extra["flame_max_cluster_size"] >= 1


def test_flame_noise_is_reproducible_with_a_seed():
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [1.0, 0.05]])
    first, _ = Flame(noise_lambda=0.1, seed=7)(w_prev, clients)
    second, _ = Flame(noise_lambda=0.1, seed=7)(w_prev, clients)
    assert torch.allclose(first["linear.weight"], second["linear.weight"])


def test_flame_noise_never_touches_the_global_rng():
    """加噪若走全局 RNG，会推进随机流并悄悄改变后续的客户端采样与 PGD 起点。

    这种污染**没有任何症状**，只会让 FLAME 组与其他组不再同源。
    """
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [1.0, 0.05]])
    rule = Flame(noise_lambda=0.5)          # 不给 seed
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    rule(w_prev, clients)
    assert torch.equal(before, torch.get_rng_state())


def test_flame_noise_varies_across_rounds():
    """同一个 Flame 实例连续两轮的噪声必须不同，否则等于每轮加同一份扰动。"""
    w_prev = _state([0.0, 0.0])
    clients = _clients([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1], [1.0, 0.05]])
    rule = Flame(noise_lambda=0.5, seed=3)
    first, _ = rule(w_prev, clients)
    second, _ = rule(w_prev, clients)
    assert not torch.allclose(first["linear.weight"], second["linear.weight"])


# ---------------------------------------------------------------------------
# FedBN 与非浮点 key
# ---------------------------------------------------------------------------
def test_every_defense_leaves_fedbn_keys_in_the_update_dict():
    """私有 key 必须留在字典里，好让 ``fedbn_update`` 照常 pop。"""
    w_prev = {"linear.weight": torch.zeros(2),
              "bn1.running_mean": torch.zeros(2),
              "bn1.num_batches_tracked": torch.tensor(0)}
    clients = [{"linear.weight": torch.tensor([float(i), float(i)]),
                "bn1.running_mean": torch.tensor([float(i), 0.0]),
                "bn1.num_batches_tracked": torch.tensor(i)}
               for i in range(1, 5)]
    for name in DEFENSES:
        update, _ = build_defense(name, tau=0.0, trim_alpha=0.25)(w_prev, clients)
        assert "bn1.running_mean" in update, name
        assert update["bn1.num_batches_tracked"].dtype == torch.int64, name


def test_use_defense_restores_the_original_aggregator():
    server = BasicServer(get_resnet(size=10))
    original = server.agg_and_update
    restore = use_defense(server, build_defense("median"))
    assert server.agg_and_update is not original
    restore()
    assert server.agg_and_update == original


# ---------------------------------------------------------------------------
# exp_ij：不得为无决策的防御编造 TPR/FPR
# ---------------------------------------------------------------------------
def _record(defense, selected=None, extra=None):
    return {
        "round_index": 1, "defense": defense,
        "is_malicious": np.array([True, False, False, False]),
        "influence": np.array([0.1, 0.9, 0.9, 0.9]),
        "l2_to_median": np.array([9.0, 1.0, 2.0, 3.0]),
        "cos_to_median": np.array([0.1, 0.9, 0.8, 0.7]),
        "mask_keep_ratio": 0.5, "zero_sign_ratio": 0.0, "effective_trim_n": 2,
        "extra": {**({"selected": selected} if selected is not None else {}),
                  **(extra or {})},
    }


def test_detection_table_leaves_tpr_empty_for_defenses_without_decisions():
    for defense in ("median", "invariant", "fedavg"):
        frame = detection_table([_record(defense)])
        for column in ("tpr", "fpr", "precision"):
            assert frame[column].isna().all(), f"{defense}/{column}"


def test_detection_table_computes_tpr_for_flame_and_multi_krum():
    for defense in ("flame", "multi_krum"):
        # 恶意客户端被排除、良性全部保留 -> TPR=1, FPR=0
        frame = detection_table([_record(defense,
                                         selected=[False, True, True, True])])
        assert float(frame["tpr"].iloc[0]) == 1.0
        assert float(frame["fpr"].iloc[0]) == 0.0
        assert float(frame["precision"].iloc[0]) == 1.0


def test_detection_table_records_malicious_rank():
    frame = detection_table([_record("median")])
    assert float(frame["malicious_rank_l2_mean"].iloc[0]) == 1.0   # l2 最大


def test_ablation_verdict_says_undetermined_when_variants_are_missing():
    import pandas as pd

    frame = pd.DataFrame([{"defense": "invariant", "round": 10,
                           "asr_personalized_targeted": 0.1}])
    verdict = ablation_verdict(frame)
    assert "未能确定" in verdict["verdict"]
    assert "invariant_mask_only" in verdict["missing"]


def test_preserve_rng_state_restores_all_three_streams():
    """周期评估必须不推进任何随机流。

    这条曾经**真的坏过**：``_maybe_evaluate`` 里 ``get_resnet()`` 新建模型会
    消耗全局 torch RNG，导致恶意客户端的投毒掩码与生成器训练被整体推移，
    而三个良性客户端毫无变化 —— 那种局部差异极易被误读成"防御生效了"。
    """
    import random as py_random

    from diag.track import preserve_rng_state
    from resnet import get_resnet

    torch.manual_seed(1234)
    np.random.seed(1234)
    py_random.seed(1234)
    torch_before = torch.get_rng_state()
    numpy_before = np.random.get_state()[1].copy()
    python_before = py_random.getstate()

    with preserve_rng_state():
        get_resnet(size=10)               # 权重初始化会吃掉大量 torch RNG
        torch.randn(1000)
        np.random.rand(100)
        py_random.random()

    assert torch.equal(torch_before, torch.get_rng_state())
    assert np.array_equal(numpy_before, np.random.get_state()[1])
    assert python_before == py_random.getstate()


def test_ablation_verdict_fails_when_a_single_component_is_not_worse():
    import pandas as pd

    rows = []
    for name, asr in (("invariant", 0.9), ("invariant_mask_only", 0.2),
                      ("invariant_trim_only", 0.3)):
        rows.append({"defense": name, "round": 10,
                     "asr_personalized_targeted": asr})
    verdict = ablation_verdict(pd.DataFrame(rows))
    assert verdict["passed"] is False
    assert "未通过" in verdict["verdict"]
    assert "边缘案例攻击" in verdict["caveat"]
