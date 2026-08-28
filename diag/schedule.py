"""攻击调度：决定**哪些轮次**恶意客户端真的投毒（实验 1B）。

六种调度回答的是不同的问题：

==============  ====================================  ==========================
调度             行为                                   问的问题
==============  ====================================  ==========================
``continuous``  全程投毒（默认，= 原论文设定）            基线
``burst``       ``[start, start+length)`` 投毒          短促高强度够不够
``intermittent``周期 ``period`` 里前 ``duty`` 轮投毒     间歇投毒会不会被稀释
``after_mta``   MTA 首次达到阈值后才开始，**并锁存**      后门是否需要主任务先成熟
``early``       ``[0, end)`` 投毒，之后停手              后门形成后能否自持
``late``        ``[start, ∞)`` 投毒                     晚期进场是否更省
==============  ====================================  ==========================

# 三件必须说清楚的事

1. **不同调度之间的 RNG 流必然发散。** 一旦某一轮的训练数据不同，之后所有的
   模型、采样、PGD 起点都不同。这是原理上无法对齐的，不是实现缺陷 ——
   所以调度之间的比较**必须靠多 seed**，不能靠单次 run 的逐位可比。
   本模块因此**不**为"关闭"的轮次空转消耗 RNG：那样既费一次 PGD 前后向，
   又只能对齐到第一次分歧为止，没有收益。

2. **``after_mta`` 用的是最近一次评估的 MTA，不是当轮的真值。** MTA 只在
   ``--eval-every`` 的轮次上算得出来，所以触发点最多滞后 ``eval_every`` 轮。
   触发后**锁存**（``_latched``）—— 不锁存的话 MTA 在阈值附近抖动会让攻击
   忽开忽关，那是第七种调度，不是设计意图。触发轮次记进 ``trigger_round``,
   分析时必须把它当协变量而不是常数。

3. **"关闭"是真的关闭。** 生成器训练 hook 与 ``poison_func`` 一起门控。
   只关投毒、留着生成器继续训练，等于给攻击者一个"停手但仍在侦察"的能力,
   那不是 ``early`` 想问的问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

__all__ = ["SCHEDULES", "AttackSchedule", "gate_attack"]

SCHEDULES = ("continuous", "burst", "intermittent", "after_mta", "early",
             "late")


@dataclass
class AttackSchedule:
    """哪一轮投毒。``active()`` 是唯一的判定入口。"""

    kind: str = "continuous"
    start: int = 0
    end: int = 0
    length: int = 0
    period: int = 0
    duty: int = 0
    mta_threshold: float = float("nan")

    # 运行时状态
    trigger_round: Optional[int] = None
    _latched: bool = False
    _cached_round: Optional[int] = None
    _cached: bool = True

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in SCHEDULES:
            raise ValueError(f"未知调度 '{self.kind}'；可选 {list(SCHEDULES)}")
        if self.kind == "burst" and self.length <= 0:
            raise ValueError("burst 需要 --attack-length > 0")
        if self.kind == "intermittent":
            if self.period <= 0 or self.duty <= 0:
                raise ValueError("intermittent 需要 --attack-period > 0 且 "
                                 "--attack-duty > 0")
            if self.duty >= self.period:
                raise ValueError(
                    f"--attack-duty={self.duty} >= --attack-period="
                    f"{self.period}，那就是 continuous，别用 intermittent 绕")
        if self.kind == "after_mta":
            threshold = self.mta_threshold
            if not (threshold == threshold) or not 0.0 < threshold < 1.0:
                raise ValueError("after_mta 需要 --attack-mta-threshold ∈ (0,1)")
        if self.kind == "early" and self.end <= 0:
            raise ValueError("early 需要 --attack-end > 0")
        if self.kind == "late" and self.start <= 0:
            raise ValueError("late 需要 --attack-start > 0")

    def active(self, round_index: int, mta: Optional[float] = None) -> bool:
        """第 ``round_index`` 轮是否投毒。``mta`` 是**最近一次评估**的主任务精度。"""
        round_index = int(round_index)
        if self.kind == "continuous":
            return True
        if self.kind == "burst":
            return self.start <= round_index < self.start + self.length
        if self.kind == "intermittent":
            return (round_index - self.start) % self.period < self.duty \
                if round_index >= self.start else False
        if self.kind == "early":
            return round_index < self.end
        if self.kind == "late":
            return round_index >= self.start
        if self.kind == "after_mta":
            if self._latched:
                return True
            if mta is not None and mta == mta and mta >= self.mta_threshold:
                self._latched = True
                self.trigger_round = round_index
                return True
            return False
        raise AssertionError(self.kind)      # __post_init__ 已经挡住了

    def active_for_round(self, round_index: int,
                         mta: Optional[float] = None) -> bool:
        """一轮**只判定一次**，之后本轮内复用。

        ``after_mta`` 会在首次达标时锁存，所以"什么时候问"会改变答案：
        轮首问（MTA 还是上一轮的）和轮尾问（MTA 已更新）可能不同。不缓存的话，
        本轮实际有没有投毒、与落盘记录的 ``attack_active_this_round``
        就可能差一轮 —— 而这一轮正好是 1B 最关心的那一轮。
        """
        round_index = int(round_index)
        if self._cached_round != round_index:
            self._cached_round = round_index
            self._cached = self.active(round_index, mta)
        return self._cached

    def describe(self) -> str:
        if self.kind == "continuous":
            return "continuous（全程）"
        if self.kind == "burst":
            return f"burst（轮次 [{self.start}, {self.start + self.length})）"
        if self.kind == "intermittent":
            return (f"intermittent（从第 {self.start} 轮起，每 {self.period} 轮"
                    f"投毒前 {self.duty} 轮）")
        if self.kind == "after_mta":
            return f"after_mta（MTA 首次 >= {self.mta_threshold:.3f} 后锁存）"
        if self.kind == "early":
            return f"early（轮次 [0, {self.end})）"
        return f"late（轮次 [{self.start}, ∞)）"

    def expected_attack_rounds(self, total_round: int) -> Optional[int]:
        """总共会投毒多少轮。``after_mta`` 取决于训练过程，返回 ``None``。"""
        if self.kind == "after_mta":
            return None
        return sum(1 for t in range(total_round) if self.active(t))


def gate_attack(clients: Sequence[Any], schedule: AttackSchedule,
                round_getter: Callable[[], int],
                mta_getter: Callable[[], Optional[float]],
                generator_schedule: Optional[AttackSchedule] = None
                ) -> Callable[[], None]:
    """按调度门控投毒与生成器训练，返回还原函数。

    两处都要门控：

    - ``client.poison_func``（``PoisonClient.fetch_data`` 调用它，``client.py:124``）
    - ``registered_funcs["before_local_training"]`` 里的 ``trigger_gen_trainer``
      （``fba.py:63`` 注册）

    ``continuous`` 直接返回空操作 —— 不包装就不会引入任何差异，
    这样"默认设定"与本模块出现之前逐位一致。

    ``generator_schedule``：给**生成器训练**一个**独立于投毒**的门控窗口。
    缺省（None）时生成器与投毒同窗（= 停攻即冻结 δ，实验 1B 的 A/B 线）。
    传入独立调度可实现"投毒只在攻击窗口、但生成器在衰减段继续在线"（C 线，
    上界对照）—— 例如 ``late(start=attack_start)`` 让 δ 在 [start, ∞) 一直更新。
    """
    if schedule.kind == "continuous" and generator_schedule is None:
        return lambda: None

    restores: List[Callable[[], None]] = []

    def _is_on() -> bool:
        return schedule.active_for_round(round_getter(), mta_getter())

    def _gen_on() -> bool:
        if generator_schedule is None:
            return _is_on()
        return generator_schedule.active_for_round(round_getter(), mta_getter())

    for client in clients:
        if not getattr(client, "diag_is_malicious", False):
            continue

        original_poison = getattr(client, "poison_func", None)
        if original_poison is not None:
            def _gated(data, label, _original=original_poison):
                # 关闭时原样返回干净数据 —— 与 BasicClient.fetch_data 等价
                return (data, label) if not _is_on() else _original(data, label)

            client.poison_func = _gated
            restores.append(
                lambda c=client, f=original_poison: setattr(c, "poison_func", f))

        stage = "before_local_training"
        hooks = client.registered_funcs.get(stage)
        if hooks:
            original_hooks = list(hooks)

            def _gated_hook(target_client, _hooks=tuple(original_hooks)):
                if not _gen_on():        # 生成器用独立门控（缺省与投毒同窗）
                    return
                for hook in _hooks:
                    hook(target_client)

            client.registered_funcs[stage] = [_gated_hook]
            restores.append(
                lambda c=client, s=stage, h=original_hooks:
                c.registered_funcs.__setitem__(s, h))

    def restore() -> None:
        for undo in restores:
            undo()

    return restore
