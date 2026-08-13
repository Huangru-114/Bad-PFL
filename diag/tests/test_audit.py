"""``diag.audit`` 的单元测试。

重点是那些**失效时会静默给出错误结论**的地方：
- 泄漏对照的 monkey-patch 是否真的移除了梯度（若没生效，两臂看起来一样，
  会被误读成"泄漏不是原因"）；
- 数据集重建的一致性守卫是否真的能抓到内容不匹配；
- Markdown 转义（判定规则里含 `|`，不转义会让整张表错位）。
"""

from __future__ import annotations

import json
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from diag.audit.common import (THRESHOLDS, Verdict, assert_partition_matches,
                               discover_runs, fmt, group_compare, md_kv,
                               md_table, valid_class_count)
from diag.audit.leak_experiment import leak_free_poison_client


# ---------------------------------------------------------------------------
# 泄漏对照的 monkey-patch
# ---------------------------------------------------------------------------
class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 4)

    def forward(self, x):
        return self.linear(x.flatten(1))


def _make_poison_client():
    """构造一个最小 PoisonClient，其 poison_func 会像真攻击一样触发 backward。"""
    import fba
    from client import PoisonClient
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(0)
    images = torch.rand(8, 3, 4, 4, generator=generator)
    labels = torch.randint(0, 4, (8,), generator=generator)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    model = _TinyModel()
    client = PoisonClient(model, loader, loader, nn.CrossEntropyLoss(),
                          partial(torch.optim.SGD, lr=0.1), poison_func=None)
    client.local_model.device = torch.device("cpu")

    def poison_func(data, label):
        # 复刻 fba.our_poison_func 的关键副作用：pgd_attack 内部的 backward
        # 会把梯度累加进 client.local_model 的参数。
        return fba.pgd_attack(client.local_model, data, label), label

    client.poison_func = poison_func
    return client


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float((param.grad ** 2).sum().item())
    return float(np.sqrt(total))


def test_unpatched_poison_client_leaks_gradients():
    """确认泄漏确实存在 —— 否则整个对照实验是在测一个不存在的东西。"""
    client = _make_poison_client()
    client.local_model.zero_grad(set_to_none=True)
    client.fetch_data()
    assert _grad_norm(client.local_model) > 0.0, (
        "预期 pgd_attack 的 backward 会把梯度累加进模型参数（fba.py:16）")


def test_leak_free_context_removes_gradients():
    """Arm B 的 monkey-patch 必须真的把那份梯度清掉。"""
    client = _make_poison_client()
    client.local_model.zero_grad(set_to_none=True)
    with leak_free_poison_client():
        client.fetch_data()
    assert _grad_norm(client.local_model) == 0.0, (
        "Arm B 应当在取数后清零梯度；若未生效，两臂结果会一样，"
        "从而被误读成'泄漏不是原因'")


def test_leak_free_context_restores_original_method():
    """退出 with 块后必须还原，否则会污染后续的 Arm A。"""
    import client as client_module

    original = client_module.PoisonClient.fetch_data
    with leak_free_poison_client():
        assert client_module.PoisonClient.fetch_data is not original
    assert client_module.PoisonClient.fetch_data is original


def test_leak_free_context_restores_on_exception():
    import client as client_module

    original = client_module.PoisonClient.fetch_data
    try:
        with leak_free_poison_client():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert client_module.PoisonClient.fetch_data is original


# ---------------------------------------------------------------------------
# 数据集重建守卫
# ---------------------------------------------------------------------------
class _FakeBundle:
    def __init__(self, targets, num_classes=4):
        from torch.utils.data import TensorDataset
        self.train_dataset = TensorDataset(
            torch.zeros(len(targets), 1), torch.tensor(targets))
        self.train_dataset.targets = list(targets)
        self.num_classes = num_classes


def _meta_with(indices, histogram):
    return {"clients": [{"client_id": 0, "train_indices": indices,
                         "label_histogram": histogram}]}


def test_assert_partition_matches_accepts_consistent_dataset():
    targets = [0, 1, 2, 3] * 4
    bundle = _FakeBundle(targets)
    indices = [0, 1, 2, 3]
    assert_partition_matches(bundle, _meta_with(indices, [1, 1, 1, 1]))


def test_assert_partition_matches_detects_content_mismatch():
    """尺寸相同但内容不同 —— 这是最危险的静默错误。"""
    bundle = _FakeBundle([0, 0, 0, 0, 1, 1, 1, 1])
    try:
        assert_partition_matches(bundle, _meta_with([0, 1, 2, 3], [1, 1, 1, 1]))
    except RuntimeError as exc:
        assert "直方图对不上" in str(exc)
    else:
        raise AssertionError("内容不一致时必须抛异常，不能带病继续")


def test_assert_partition_matches_detects_out_of_range():
    bundle = _FakeBundle([0, 1, 2, 3])
    try:
        assert_partition_matches(bundle, _meta_with([0, 1, 99], [1, 1, 1, 0]))
    except RuntimeError as exc:
        assert "越界" in str(exc) or "train_indices" in str(exc)
    else:
        raise AssertionError("索引越界时必须抛异常")


# ---------------------------------------------------------------------------
# 统计与格式化
# ---------------------------------------------------------------------------
def test_group_compare_detects_full_separation():
    values = [1.0, 1.1, 1.2, 5.0, 5.1]
    labels = [False, False, False, True, True]
    stats = group_compare(values, labels, "x")
    assert stats["separated"] is True
    assert stats["separation_gap"] > 0
    assert stats["auc"] == 1.0


def test_group_compare_detects_overlap():
    values = [1.0, 3.0, 2.0, 4.0]
    labels = [False, False, True, True]
    stats = group_compare(values, labels, "x")
    assert stats["separated"] is False
    assert stats["separation_gap"] < 0


def test_group_compare_is_nan_safe():
    stats = group_compare([1.0, float("nan"), 5.0], [False, True, True], "x")
    assert stats["n_benign"] == 1 and stats["n_malicious"] == 1
    stats_empty = group_compare([1.0, 2.0], [False, False], "x")
    assert np.isnan(stats_empty["auc"])


def test_fmt_escapes_pipes_and_newlines():
    """判定规则里常含 `|rho| > 0.5`，不转义会切断 Markdown 表格。"""
    assert fmt("|rho| > 0.5") == r"\|rho\| > 0.5"
    assert "<br>" in fmt("a\nb")


def test_md_table_stays_rectangular_with_pipes():
    """含竖线的单元格必须被转义，否则表格列数不一致、整张表错位。

    注意要数**未转义**的竖线 —— `\\|` 里仍然有一个 `|` 字符，
    直接 count("|") 会把转义后的也算进去。
    """
    import re

    frame = pd.DataFrame([{"规则": "|x| > 1", "值": 2.0}])
    lines = md_table(frame).splitlines()
    counts = {len(re.findall(r"(?<!\\)\|", line)) for line in lines}
    assert len(counts) == 1, f"各行的未转义竖线数不一致: {counts}"


def test_md_table_empty_frame():
    assert "无数据" in md_table(pd.DataFrame())


def test_valid_class_count():
    assert valid_class_count([10, 4, 5, 0], min_class_count=5) == 2


def test_verdict_as_row_has_all_columns():
    row = Verdict("项", "结论", "规则", "实测", "处理").as_row()
    assert set(row) == {"检查项", "判定", "规则", "实测", "后续处理"}


def test_thresholds_are_documented_numbers():
    for key, value in THRESHOLDS.items():
        assert isinstance(value, (int, float)), f"{key} 必须是数值阈值"


def test_discover_runs_parses_and_ignores_junk():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "attack_a0.05_s2").mkdir()
        (root / "clean_a1.0_s0").mkdir()
        (root / "not_a_run").mkdir()
        frame = discover_runs(root)
    assert len(frame) == 2
    assert set(frame["mode"]) == {"attack", "clean"}
    assert 0.05 in set(frame["alpha"])


def test_discover_runs_missing_root_returns_empty():
    assert len(discover_runs(Path("/definitely/not/here"))) == 0


def test_md_table_escapes_header_names():
    """列名里也可能有竖线（如 ``||mu-mu_g||^2``），表头必须一起转义。"""
    import re

    frame = pd.DataFrame([{"AUC(||mu||^2)": 1.0, "x": 2.0}])
    lines = md_table(frame).splitlines()
    counts = {len(re.findall(r"(?<!\\)\|", line)) for line in lines}
    assert len(counts) == 1, f"表头未转义导致列数不一致: {counts}"


def test_md_table_preserves_column_dtypes():
    """按列取值而非 iterrows()：否则 int/bool 会被上转成 float。"""
    frame = pd.DataFrame([{"seed": 0, "alpha": 0.5, "flag": True},
                          {"seed": 12, "alpha": 0.05, "flag": False}])
    rendered = md_table(frame)
    assert "| 0 |" in rendered and "| 12 |" in rendered, (
        f"seed 应显示为整数，实际渲染:\n{rendered}")
    assert "是" in rendered and "否" in rendered, "bool 应显示为 是/否"
