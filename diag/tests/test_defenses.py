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


def _malicious_getter(n: int, malicious=(0,)):
    """给 ``oracle_exclude`` 用的恶意掩码；遍历 ``DEFENSES`` 的测试都要带上它。"""
    mask = np.zeros(n, dtype=bool)
    mask[list(malicious)] = True
    return lambda: mask


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
        build_defense(name, tau=0.0, trim_alpha=0.25,
                      malicious_getter=_malicious_getter(4))(w_prev, clients)
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
        update, _ = build_defense(name, tau=0.0, trim_alpha=0.25,
                                  malicious_getter=_malicious_getter(4)
                                  )(w_prev, clients)
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


def test_all_defenses_run_on_cuda_when_available():
    """设备一致性回归。

    第一版实现在集群上崩了：``Expected all tensors to be on the same device``
    —— 所有 float64 累加器都建在 CPU 上，而数据在 GPU 上。本容器只有 CPU，
    这个测试在这里是空转；**它的价值在于集群上会真跑**。

    没有 GPU 时用 meta 设备也无法替代：``meta + cpu`` 不报错，
    且 ``bincount`` 在 meta 上未实现。
    """
    if not torch.cuda.is_available():
        print("      (跳过：本机无 CUDA —— 这个测试要在集群上才有意义)")
        return

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    w_prev = {k: v.to(device) for k, v in get_resnet(size=10).state_dict().items()}
    clients = []
    for offset in range(6):
        torch.manual_seed(offset + 1)
        clients.append({k: v.to(device)
                        for k, v in get_resnet(size=10).state_dict().items()})

    from diag.instrumentation import round_signals

    for name in list(DEFENSES) + list(ABLATION_VARIANTS):
        update, outcome = build_defense(name, tau=0.2, trim_alpha=0.25)(
            w_prev, clients)
        assert update["linear.weight"].device.type == "cuda", name
        assert np.isfinite(outcome.influence).all(), name
    signals = round_signals(w_prev, clients, trim_k=1)
    for key, values in signals.items():
        assert np.isfinite(values).all(), key


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


def test_oracle_exclude_drops_exactly_the_malicious_updates():
    """静默恶意：被剔除者影响力为 0，聚合结果等于只对良性求平均。"""
    from diag.defenses import OracleExclude

    w_prev = _state([0.0])
    clients = _clients([[2.0], [4.0], [100.0]])      # 第 3 个是恶意
    mask = np.array([False, False, True])
    update, outcome = OracleExclude(malicious_getter=lambda: mask)(w_prev, clients)

    assert float(update["linear.weight"][0]) == 3.0   # (2+4)/2，恶意的 100 未参与
    assert np.allclose(outcome.influence, [1.0, 1.0, 0.0])
    assert outcome.selected.tolist() == [True, True, False]
    assert outcome.extra["n_excluded_oracle"] == 1
    assert outcome.extra["inner_defense"] == "fedavg"


def test_oracle_exclude_yields_perfect_tpr_and_zero_fpr():
    """上界组的自检：完美决策必须让整套 TPR/FPR 机制给出 1.0 / 0.0。"""
    from diag.exp_ij import detection_table

    record = _record("oracle_exclude", selected=[False, True, True, True])
    frame = detection_table([record])
    assert float(frame["tpr"].iloc[0]) == 1.0
    assert float(frame["fpr"].iloc[0]) == 0.0
    assert float(frame["youden_j"].iloc[0]) == 1.0


def test_oracle_exclude_refuses_a_mask_of_the_wrong_length():
    """掩码顺序/时机对不上是静默错误，必须报错而不是凑合。"""
    from diag.defenses import OracleExclude

    w_prev = _state([0.0])
    clients = _clients([[1.0], [2.0], [3.0]])
    rule = OracleExclude(malicious_getter=lambda: np.array([False, True]))
    try:
        rule(w_prev, clients)
    except ValueError as exc:
        assert "长度" in str(exc)
    else:
        raise AssertionError("掩码长度不符必须报错")


def test_oracle_exclude_aborts_when_every_client_is_malicious():
    """全恶意时没有合法聚合结果；静默地把攻击者聚合进去是最坏结局。"""
    from diag.defenses import OracleExclude

    w_prev = _state([0.0])
    clients = _clients([[1.0], [2.0]])
    rule = OracleExclude(malicious_getter=lambda: np.array([True, True]))
    try:
        rule(w_prev, clients)
    except RuntimeError as exc:
        assert "全是恶意" in str(exc)
    else:
        raise AssertionError("全恶意必须中止")


def test_oracle_exclude_requires_a_getter():
    from diag.defenses import OracleExclude

    try:
        OracleExclude()(_state([0.0]), _clients([[1.0]]))
    except ValueError as exc:
        assert "malicious_getter" in str(exc)
    else:
        raise AssertionError("缺 malicious_getter 必须报错")


def test_oracle_exclude_can_wrap_another_rule():
    """剔除后可以再叠一个聚合规则；默认是 fedavg，但不写死。"""
    rule = build_defense("oracle_exclude", inner="median",
                         malicious_getter=lambda: np.array([False] * 4 + [True]))
    w_prev = _state([0.0])
    clients = _clients([[1.0], [2.0], [3.0], [4.0], [900.0]])
    update, outcome = rule(w_prev, clients)
    assert outcome.extra["inner_defense"] == "median"
    assert float(outcome.influence[4]) == 0.0
    # 剩下 4 个的 g = [-1,-2,-3,-4]；torch.median 对偶数取**下**中位 -> -3，
    # 于是 w_t = 0 − (−3) = 3。恶意的 900 完全没有参与。
    assert float(update["linear.weight"][0]) == 3.0


def test_decision_diagnosis_flags_a_tpr_that_is_only_an_exclusion_artefact():
    """每轮固定排除 4/10 且与恶意无关 -> TPR≈0.4 但 J≈0，必须被点出来。"""
    import pandas as pd

    from diag.exp_ij import decision_diagnosis

    rng = np.random.RandomState(0)
    rows = []
    for r in range(300):
        excluded = np.zeros(10, dtype=bool)
        excluded[rng.choice(10, 4, replace=False)] = True   # 与恶意无关
        malicious = np.zeros(10, dtype=bool)
        malicious[rng.choice(10, 1)] = True
        tpr = float((excluded & malicious).sum() / malicious.sum())
        fpr = float((excluded & ~malicious).sum() / (~malicious).sum())
        rows.append({"round": r, "defense": "flame", "n_selected": 10,
                     "n_malicious_in_round": 1, "tpr": tpr, "fpr": fpr,
                     "youden_j": tpr - fpr, "n_excluded": 4})
    result = decision_diagnosis(pd.DataFrame(rows))
    assert abs(result["youden_j"]) < 0.05
    assert abs(result["chance_tpr_fpr"] - 0.4) < 1e-9
    assert any("基本无关" in n for n in result["notes"])


def test_decision_diagnosis_stays_quiet_when_detection_is_real():
    import pandas as pd

    from diag.exp_ij import decision_diagnosis

    rows = [{"round": r, "defense": "multi_krum", "n_selected": 10,
             "n_malicious_in_round": 1, "tpr": 1.0, "fpr": 0.111,
             "youden_j": 0.889, "n_excluded": 2} for r in range(100)]
    result = decision_diagnosis(pd.DataFrame(rows))
    assert result["youden_j"] > 0.8
    assert not any("基本无关" in n for n in result["notes"])


def test_decision_diagnosis_refuses_defenses_without_client_decisions():
    import pandas as pd

    from diag.exp_ij import decision_diagnosis

    for defense in ("median", "invariant", "fedavg"):
        rows = [{"round": 1, "defense": defense, "n_selected": 10,
                 "n_malicious_in_round": 1}]
        result = decision_diagnosis(pd.DataFrame(rows))
        assert "不做客户端级二元决策" in result["verdict"], defense


def test_chance_mask_keep_ratio_matches_the_binomial_tail():
    from diag.invariant_agg import chance_mask_keep_ratio

    # N=10、τ=0.2 -> |Σ| ≥ 4 -> 1 − (C(10,4)+C(10,5)+C(10,6))/2^10
    assert abs(chance_mask_keep_ratio(10, 0.2) - (1 - 672 / 1024)) < 1e-12
    # τ=0.2 与 τ=0.3 是**同一个掩码**：Σ 与 N 同奇偶，|Σ|=3 不可能出现
    assert chance_mask_keep_ratio(10, 0.2) == chance_mask_keep_ratio(10, 0.3)
    # τ 越大保留越少
    ratios = [chance_mask_keep_ratio(10, t) for t in (0.1, 0.3, 0.5, 0.7)]
    assert ratios == sorted(ratios, reverse=True)


def test_mask_diagnosis_flags_chance_level_and_no_response():
    """两条判据都必须在"掩码没起作用"的构造上触发。"""
    import pandas as pd

    from diag.exp_ij import mask_diagnosis

    rng = np.random.RandomState(0)
    rows = []
    for r in range(200):
        n_mal = 1 if r % 2 else 0
        rows.append({"round": r, "tau": 0.2, "n_selected": 10,
                     "n_malicious_in_round": n_mal,
                     # 贴着随机基线 0.34375，且与是否有恶意无关
                     "mask_keep_ratio": 0.3438 + rng.normal(0, 0.002)})
    result = mask_diagnosis(pd.DataFrame(rows))
    assert result["equivalent_threshold"] == 4        # 不是 3
    assert len(result["notes"]) == 2
    assert "随机符号基线" in result["notes"][0]
    assert "没有反应" in result["notes"][1]


def test_mask_diagnosis_stays_quiet_when_the_mask_actually_works():
    import pandas as pd

    rows = [{"round": r, "tau": 0.2, "n_selected": 10,
             "n_malicious_in_round": r % 2,
             # 远高于随机基线，且攻击者在场时明显更低
             "mask_keep_ratio": 0.90 if r % 2 == 0 else 0.60}
            for r in range(200)]
    from diag.exp_ij import mask_diagnosis

    assert mask_diagnosis(pd.DataFrame(rows))["notes"] == []


def _ablation_frame(values):
    import pandas as pd

    return pd.DataFrame([{"defense": name, "round": 10,
                          "asr_personalized_targeted": asr}
                         for name, asr in values.items()])


def test_ablation_verdict_fails_when_a_single_component_is_not_worse():
    verdict = ablation_verdict(_ablation_frame({
        "invariant": 0.9, "invariant_mask_only": 0.2,
        "invariant_trim_only": 0.3}))
    assert verdict["passed"] is False
    assert "未通过" in verdict["verdict"]
    assert "边缘案例攻击" in verdict["caveat"]


def test_ablation_verdict_is_undetermined_without_a_fedavg_baseline():
    """排序满足但没有无防御对照 —— 不能算通过。

    全部变体都停在高位时，"单组件 > 组合"这个排序同样可以由"防御完全无效"
    产生，而那正是这个检查点要排除的另一种可能。
    """
    verdict = ablation_verdict(_ablation_frame({
        "invariant": 0.91, "invariant_mask_only": 0.99,
        "invariant_trim_only": 0.95}))
    assert verdict["passed"] is None
    assert "未能确定" in verdict["verdict"]
    assert "fedavg" in verdict["verdict"]


def test_ablation_verdict_is_undetermined_when_the_combination_suppresses_nothing():
    """有 fedavg 对照，但组合几乎没把 ASR 压下来 -> 这次检查没有区分力。"""
    verdict = ablation_verdict(_ablation_frame({
        "fedavg": 0.93, "invariant": 0.91,
        "invariant_mask_only": 0.99, "invariant_trim_only": 0.95}))
    assert verdict["passed"] is None
    assert "没有区分力" in verdict["verdict"]


def test_ablation_verdict_passes_only_when_the_combination_actually_works():
    verdict = ablation_verdict(_ablation_frame({
        "fedavg": 0.95, "invariant": 0.05,
        "invariant_mask_only": 0.65, "invariant_trim_only": 0.51}))
    assert verdict["passed"] is True
    assert "通过" in verdict["verdict"]
