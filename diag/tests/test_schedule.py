"""``diag.schedule`` 的单元测试。

调度错一轮，1B 的"停攻后衰减"曲线就整体平移，而图看上去完全正常 ——
所以每种调度的边界轮次都逐个钉死。
"""

from __future__ import annotations

import numpy as np

from diag.schedule import SCHEDULES, AttackSchedule, gate_attack


def _on(schedule, total=20, mta=None):
    return [t for t in range(total) if schedule.active(t, mta)]


# ---------------------------------------------------------------------------
# 各调度的边界
# ---------------------------------------------------------------------------
def test_continuous_is_always_on():
    assert _on(AttackSchedule("continuous"), total=5) == [0, 1, 2, 3, 4]


def test_burst_is_half_open_on_the_right():
    schedule = AttackSchedule("burst", start=3, length=4)
    assert _on(schedule, total=12) == [3, 4, 5, 6]


def test_early_stops_before_end():
    assert _on(AttackSchedule("early", end=3), total=6) == [0, 1, 2]


def test_late_starts_at_start():
    assert _on(AttackSchedule("late", start=4), total=7) == [4, 5, 6]


def test_intermittent_duty_cycle():
    schedule = AttackSchedule("intermittent", period=5, duty=2)
    assert _on(schedule, total=12) == [0, 1, 5, 6, 10, 11]


def test_intermittent_respects_start_offset():
    schedule = AttackSchedule("intermittent", start=3, period=4, duty=1)
    assert _on(schedule, total=12) == [3, 7, 11]


# ---------------------------------------------------------------------------
# after_mta：锁存与滞后
# ---------------------------------------------------------------------------
def test_after_mta_latches_once_triggered():
    schedule = AttackSchedule("after_mta", mta_threshold=0.6)
    assert schedule.active(0, 0.30) is False
    assert schedule.active(1, 0.61) is True          # 触发
    assert schedule.trigger_round == 1
    # MTA 掉回阈值以下也**不能**关掉，否则就成了第七种调度
    assert schedule.active(2, 0.10) is True
    assert schedule.active(3, None) is True


def test_after_mta_ignores_missing_mta():
    """没评估过的轮次 mta=None，不能当成 0 也不能当成触发。"""
    schedule = AttackSchedule("after_mta", mta_threshold=0.6)
    assert schedule.active(0, None) is False
    assert schedule.trigger_round is None


def test_after_mta_ignores_nan():
    schedule = AttackSchedule("after_mta", mta_threshold=0.6)
    assert schedule.active(0, float("nan")) is False


# ---------------------------------------------------------------------------
# 参数校验：宁可报错也不要静默地变成另一种调度
# ---------------------------------------------------------------------------
def _raises(fn) -> str:
    try:
        fn()
    except ValueError as error:
        return str(error)
    raise AssertionError("应当报错但没有")


def test_unknown_kind_lists_the_options():
    message = _raises(lambda: AttackSchedule("whenever"))
    assert "continuous" in message


def test_duty_at_least_period_is_rejected_not_silently_continuous():
    message = _raises(lambda: AttackSchedule("intermittent", period=4, duty=4))
    assert "continuous" in message


def test_missing_required_parameters_are_rejected():
    assert "attack-length" in _raises(lambda: AttackSchedule("burst"))
    assert "attack-end" in _raises(lambda: AttackSchedule("early"))
    assert "attack-start" in _raises(lambda: AttackSchedule("late"))
    assert "mta-threshold" in _raises(lambda: AttackSchedule("after_mta"))
    assert "mta-threshold" in _raises(
        lambda: AttackSchedule("after_mta", mta_threshold=1.5))


def test_every_named_schedule_is_constructible():
    """SCHEDULES 里列出来的每一种都要能构造 —— 漏一种 CLI 就会崩在集群上。"""
    built = {
        "continuous": dict(),
        "burst": dict(start=1, length=2),
        "intermittent": dict(period=4, duty=1),
        "after_mta": dict(mta_threshold=0.5),
        "early": dict(end=3),
        "late": dict(start=3),
    }
    for kind in SCHEDULES:
        schedule = AttackSchedule(kind, **built[kind])
        assert schedule.describe()
        assert isinstance(schedule.active(5, 0.9), bool)


def test_expected_attack_rounds_is_none_only_for_after_mta():
    assert AttackSchedule("burst", start=2, length=3
                          ).expected_attack_rounds(20) == 3
    assert AttackSchedule("intermittent", period=5, duty=2
                          ).expected_attack_rounds(20) == 8
    assert AttackSchedule("after_mta", mta_threshold=0.5
                          ).expected_attack_rounds(20) is None


# ---------------------------------------------------------------------------
# 门控：投毒与生成器 hook 必须一起关
# ---------------------------------------------------------------------------
class _FakeClient:
    def __init__(self, malicious=True):
        self.diag_is_malicious = malicious
        self.poison_calls = 0
        self.hook_calls = 0
        self.registered_funcs = {"before_local_training": [self._train_gen]}

        def poison(data, label):
            self.poison_calls += 1
            return data * 0 + 99, label * 0 + 7

        self.poison_func = poison

    def _train_gen(self, client):
        self.hook_calls += 1


def test_gate_turns_off_both_poisoning_and_the_generator():
    client = _FakeClient()
    state = {"round": 0}
    schedule = AttackSchedule("early", end=2)
    restore = gate_attack([client], schedule,
                          lambda: state["round"], lambda: None)

    # 第 0 轮：开
    data, label = client.poison_func(np.array([1]), np.array([1]))
    client.registered_funcs["before_local_training"][0](client)
    assert data[0] == 99 and client.poison_calls == 1 and client.hook_calls == 1

    # 第 5 轮：关 —— 数据原样返回，生成器也不训练
    state["round"] = 5
    data, label = client.poison_func(np.array([1]), np.array([2]))
    client.registered_funcs["before_local_training"][0](client)
    assert data[0] == 1 and label[0] == 2
    assert client.poison_calls == 1 and client.hook_calls == 1
    restore()


def test_continuous_installs_no_wrapper_at_all():
    """默认设定必须与本模块出现之前逐位一致 —— 不包装就不会有任何差异。"""
    client = _FakeClient()
    original_poison = client.poison_func
    original_hooks = client.registered_funcs["before_local_training"]
    restore = gate_attack([client], AttackSchedule("continuous"),
                          lambda: 0, lambda: None)
    assert client.poison_func is original_poison
    assert client.registered_funcs["before_local_training"] is original_hooks
    restore()


def test_benign_clients_are_never_touched():
    benign = _FakeClient(malicious=False)
    original = benign.poison_func
    restore = gate_attack([benign], AttackSchedule("early", end=2),
                          lambda: 99, lambda: None)
    assert benign.poison_func is original
    restore()


def test_restore_puts_everything_back():
    client = _FakeClient()
    original_poison = client.poison_func
    original_hooks = client.registered_funcs["before_local_training"]
    restore = gate_attack([client], AttackSchedule("early", end=2),
                          lambda: 0, lambda: None)
    assert client.poison_func is not original_poison
    restore()
    assert client.poison_func is original_poison
    assert client.registered_funcs["before_local_training"] == original_hooks


def test_gate_reads_mta_through_the_getter_not_a_snapshot():
    """after_mta 必须**每轮**读一次当下的 MTA，不能在安装时把值抓死。"""
    client = _FakeClient()
    state = {"round": 0, "mta": 0.1}
    schedule = AttackSchedule("after_mta", mta_threshold=0.5)
    restore = gate_attack([client], schedule,
                          lambda: state["round"], lambda: state["mta"])

    client.poison_func(np.array([1]), np.array([1]))
    assert client.poison_calls == 0                 # 还没到阈值
    state["round"], state["mta"] = 1, 0.7
    client.poison_func(np.array([1]), np.array([1]))
    assert client.poison_calls == 1                 # 下一轮触发
    restore()


def test_schedule_is_evaluated_once_per_round_not_once_per_call():
    """一轮内多次取数据必须得到同一个判定。

    否则 after_mta 会在**一轮中途**翻转：同一轮里前几个 batch 干净、
    后几个 batch 投毒，而落盘的 attack_active_this_round 只能记一个值。
    """
    schedule = AttackSchedule("after_mta", mta_threshold=0.5)
    assert schedule.active_for_round(3, 0.1) is False
    # 同一轮内 MTA 变了也不改判定
    assert schedule.active_for_round(3, 0.9) is False
    # 换一轮才重新判定
    assert schedule.active_for_round(4, 0.9) is True


def test_recorded_flag_matches_what_actually_happened():
    """落盘的 attack_active 与实际投毒必须是同一次判定的结果。"""
    client = _FakeClient()
    state = {"round": 7, "mta": 0.9}
    schedule = AttackSchedule("after_mta", mta_threshold=0.5)
    restore = gate_attack([client], schedule,
                          lambda: state["round"], lambda: state["mta"])
    client.poison_func(np.array([1]), np.array([1]))
    assert client.poison_calls == 1
    # run_fl 在轮首用同一个 API 取要落盘的标志
    assert schedule.active_for_round(7, state["mta"]) is True
    restore()
