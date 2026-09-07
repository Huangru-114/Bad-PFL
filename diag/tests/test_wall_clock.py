"""墙钟仪表（Stage B 标定）—— `track.py` 是否真的记下了每轮的耗时。

**为什么需要**：标定要回答「降轮数还是提 local_steps 更划算」，答案完全取决于
「训练 vs 评估」的墙钟比例。而 implantation CSV 此前**一个时间列都没有**，
这个问题在 Bad-PFL 侧根本答不了。tf-dpfl 侧的对应件是 `[Timing]` 行。

**本模块的验证范围（如实说）**：这里是 AST / 源码级的接线检查，
不是端到端计时。本机没有 torch，`TrainingTracker` 连 import 都做不到，
真正的数值要在集群上跑一个 run 之后看 CSV 的两列。
这条测试挡的是「列被改没了 / 计时被搬到错误的位置 / 用了会被 NTP 拽走的钟」。

纯 stdlib，本地秒级。
"""

from __future__ import annotations

import ast
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent / "track.py"
SRC = TRACK.read_text(encoding="utf-8")


def _func_src(name: str) -> str:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(SRC, node)
            assert seg, f"取不到 {name} 的源码"
            return seg
    raise AssertionError(f"track.py 里找不到函数 {name}")


def test_implantation_row_carries_both_wall_clock_columns():
    """两列缺一不可：只有总耗时的话，「评估贵」与「训练贵」分不开，
    而这两种情况下 Stage B 的建议正好相反。"""
    row = _func_src("_record_evaluation") if "_record_evaluation" in SRC else SRC
    assert '"train_wall_s"' in row
    assert '"eval_wall_s"' in row


def _clock_calls(func_name: str) -> set:
    """函数里实际调用到的 `time.*`（**按 AST，不按文本**）。

    第一版是文本匹配，结果被同一个函数里「不要用 time.time()」这句**注释**
    绊倒了 —— 注释和字符串里的调用不是调用。
    """
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {n.func.attr for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "time"}
    raise AssertionError(f"track.py 里找不到函数 {func_name}")


def test_timing_uses_monotonic_clock():
    """`time.time()` 会被 NTP 校时拽走，量出负的耗时。计时一律用单调钟。"""
    calls = _clock_calls("attach")
    assert "perf_counter" in calls, f"attach 没用单调钟；实际调用 {calls}"
    assert "time" not in calls, f"attach 里调了 time.time()；实际调用 {calls}"


def test_train_time_is_stamped_before_evaluation_starts():
    """train_wall_s 必须在 `_maybe_evaluate` **之前**定下来。

    否则评估耗时会被算进训练里 —— 而评估在 eval_every 轮才发生，
    于是「训练时间」会随 eval_every 变化，是个假指标。
    """
    attach = _func_src("attach")
    i_train = attach.index("self._t_train_s")
    i_eval = attach.index("self._maybe_evaluate")
    assert i_train < i_eval, "train_wall_s 在评估之后才算 —— 会把评估算进训练"


def test_eval_time_is_backfilled_onto_the_row_of_this_round():
    """评估耗时只能事后知道，而植入行在 `_evaluate_now` 里就写了 → 必须回填。

    回填要认轮次：若上一轮的行还在末尾（本轮没评估），把本轮的耗时写上去
    就是张冠李戴。
    """
    attach = _func_src("attach")
    assert 'implantation_rows[-1]["eval_wall_s"]' in attach
    assert 'implantation_rows[-1].get("round") == self.cur_round' in attach, \
        "回填没有校验轮次 —— 可能写到上一轮的行上"


def test_unmeasured_time_is_nan_not_zero():
    """没量到就留 nan。0.0 会被读成「不花时间」，而这正是标定要测的量
    （铁律 #5：无定义的指标留空，绝不用 0 填充后当数值参与统计）。"""
    assert '"train_wall_s": (float("nan") if self._t_train_s is None' in SRC
    assert '"eval_wall_s": float("nan")' in SRC


def test_round_begin_stamps_the_clock():
    attach = _func_src("attach")
    assert "self._t_round_begin = time.perf_counter()" in attach
    i_begin = attach.index("_on_round_begin")
    i_end = attach.index("_on_round_end")
    i_stamp = attach.index("self._t_round_begin = time.perf_counter()")
    assert i_begin < i_stamp < i_end, "打点不在 _on_round_begin 里"
