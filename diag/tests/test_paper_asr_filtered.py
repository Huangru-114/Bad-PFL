"""逐轮 filtered ASR（``diag/track.py::_paper_asr``）。

**为什么要这一列**：本仓库的主 ASR ``asr_paper_*`` 一直是 **unfiltered**
（分母 = 全部样本，逐行复刻 ``main.py:131``），而 tf-dpfl 那边
（``attack/backdoor_eval.py:41``）是 **filtered**（排除真实标签已是目标类的样本）。
两库的数字此前不在同一个分母上，报告里却并排讨论；报告 §2 写的定义还是
「排除目标类」，与它实际引用的列相反。

CIFAR-10 均匀分布下目标类约占 1/10，所以两个口径差约 10 个百分点 ——
足以改变「哪个剂量格更强」这类结论。

补这一列是**零额外机时**：原始标签在同一次前向里本来就在手边。
此前只有 ``recompute_asr_final`` 能出 filtered，而它只能算最终轮
（逐轮的 per-client 模型没有存）。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TRACK = Path(__file__).resolve().parent.parent / "track.py"

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None                                      # type: ignore[assignment]
    HAVE_TORCH = False


def _require_torch():
    if not HAVE_TORCH:
        raise unittest.SkipTest("需要 torch（本机未安装）")


def _tree():
    return ast.parse(TRACK.read_text(encoding="utf-8"))


def _columns():
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "IMPLANTATION_COLUMNS"):
            return [e.value for e in node.value.elts]
    raise AssertionError("track.py 里找不到 IMPLANTATION_COLUMNS")


# ══════════════════════════════════════════════════════════════════════════
# 不需要 torch
# ══════════════════════════════════════════════════════════════════════════

FILTERED_COLS = ("asr_paper_filtered_benign",
                 "asr_paper_filtered_malicious",
                 "asr_paper_filtered_all")


def test_filtered_columns_are_declared():
    cols = _columns()
    for c in FILTERED_COLS:
        assert c in cols, f"{c} 没有登记进 IMPLANTATION_COLUMNS"


def test_unfiltered_columns_are_untouched():
    """
    新增列不能改动既有列的语义 —— 此前所有 exp1 结果都建立在
    ``asr_paper_*``（unfiltered）之上，改名或改义会让它们不可比。
    """
    cols = _columns()
    for c in ("asr_paper_benign", "asr_paper_malicious", "asr_paper_all"):
        assert c in cols, f"既有列 {c} 不见了"


def test_every_declared_column_is_actually_written():
    """
    登记了列但没往 row 里写 = CSV 里一整列空白，而且**不会报错**。
    这是本仓库对「漏登记的测试文件」同款的态度：静默的空比缺失更糟。
    """
    cols = _columns()
    written = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    written.add(k.value)
    missing = [c for c in cols if c not in written]
    assert not missing, f"这些列登记了但从没被写进任何 row：{missing}"


def test_paper_asr_returns_both_conventions():
    """``_paper_asr`` 必须返回 (unfiltered, filtered) 两个值。"""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "_paper_asr":
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            assert returns, "_paper_asr 没有 return"
            for r in returns:
                assert isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2, \
                    f"_paper_asr 的 return 不是二元组：{ast.unparse(r)}"
            return
    raise AssertionError("track.py 里找不到 _paper_asr")


def test_filtered_denominator_excludes_the_target_class():
    """filtered 的分母必须由 ``labels != target`` 决定，而不是别的什么。"""
    src = TRACK.read_text(encoding="utf-8")
    assert "eligible = (labels != target)" in src, \
        "找不到 filtered 的资格判定 `labels != target`"


# ══════════════════════════════════════════════════════════════════════════
# 需要 torch：真跑一遍，手算出正确答案对拍
# ══════════════════════════════════════════════════════════════════════════

class _FakeLoader:
    def __init__(self, images, labels):
        self._b = [(images, labels)]

    def __iter__(self):
        return iter(self._b)


class _FixedPred(torch.nn.Module if HAVE_TORCH else object):
    """把预测写死，这样正确答案可以手算出来。"""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._lin = torch.nn.Linear(1, 1)      # 让 .training / .eval() 有意义

    def forward(self, x):
        n_cls = int(self._preds.max()) + 2
        out = torch.zeros(len(self._preds), n_cls)
        out[torch.arange(len(self._preds)), self._preds] = 1.0
        return out


def _tracker(target=0):
    from diag.track import TrainingTracker
    obj = TrainingTracker.__new__(TrainingTracker)
    obj.paper_eval_func = lambda images, labels: (images, labels)
    obj.target_class = target
    obj.device = torch.device("cpu")
    return obj, TrainingTracker


def test_filtered_and_unfiltered_match_hand_computed_values():
    """
    手工构造：8 个样本，真实标签里 2 个已经是目标类 0。
    预测里有 5 个是目标类，其中 2 个落在「真实标签本来就是 0」的样本上。

      unfiltered = 5/8 = 0.625            ← 把 2 个本来就是 0 的算成了"攻击成功"
      filtered   = (5-2)/(8-2) = 3/6 = 0.5

    差 0.125 —— 这就是口径不一致能造成的量级。
    """
    _require_torch()
    obj, cls = _tracker(target=0)
    labels = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    preds  = torch.tensor([0, 0, 0, 0, 0, 4, 5, 6])
    images = torch.zeros(8, 1)
    unf, flt = cls._paper_asr(obj, _FixedPred(preds), _FakeLoader(images, labels))
    assert abs(unf - 5 / 8) < 1e-9, f"unfiltered={unf}，应为 0.625"
    assert abs(flt - 3 / 6) < 1e-9, f"filtered={flt}，应为 0.5"


def test_unfiltered_is_bit_identical_to_the_old_behaviour():
    """
    新增 filtered 不能改动 unfiltered 的数值 —— 此前所有 exp1 结果都建立在它之上。
    没有目标类样本时两者必须相等（分母相同）。
    """
    _require_torch()
    obj, cls = _tracker(target=0)
    labels = torch.tensor([1, 2, 3, 4])          # 一个目标类样本都没有
    preds  = torch.tensor([0, 0, 3, 4])
    unf, flt = cls._paper_asr(obj, _FixedPred(preds), _FakeLoader(torch.zeros(4, 1), labels))
    assert abs(unf - 0.5) < 1e-9
    assert abs(flt - unf) < 1e-9, "没有目标类样本时两个口径必须相等"


def test_all_samples_are_target_class_gives_nan_not_zero():
    """
    全是目标类样本时 filtered 的分母为 0 → **无定义**，必须是 nan 不能是 0。
    0 会被读成「攻击完全失败」，而那是个强结论。
    """
    _require_torch()
    obj, cls = _tracker(target=0)
    labels = torch.tensor([0, 0, 0])
    preds  = torch.tensor([0, 0, 0])
    unf, flt = cls._paper_asr(obj, _FixedPred(preds), _FakeLoader(torch.zeros(3, 1), labels))
    assert abs(unf - 1.0) < 1e-9
    assert flt != flt, f"filtered 应为 nan，实际是 {flt}"


def test_model_training_mode_is_restored():
    """评估不能把模型留在 eval() —— 那会静默改变后续训练的 BN 行为。"""
    _require_torch()
    obj, cls = _tracker(target=0)
    model = _FixedPred(torch.tensor([0, 1]))
    model.train()
    cls._paper_asr(obj, model, _FakeLoader(torch.zeros(2, 1), torch.tensor([1, 2])))
    assert model.training, "_paper_asr 之后模型没有恢复 train() 状态"
