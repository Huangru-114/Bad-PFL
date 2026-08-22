"""``diag.represent`` 的单元测试。

主要盯两件事：

- **配对距离必须真的配对**。形状不一致时如果按最短长度截断，drift 会变成一个
  看起来正常但毫无意义的数；必须报错。
- **``trigger_pull`` 的符号**。正数 = 触发器把表征拉向目标类原型。符号搞反会让
  整个结论倒过来，而图上完全看不出来。
"""

from __future__ import annotations

import numpy as np
import torch

from diag.represent import (prototype_distances, representation_row,
                            representation_table)


class _FakeProbe:
    """固定顺序的探针集：直接吐出预设的 (images, labels)。"""

    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def loader(self, batch_size=256):
        return [(self.images, self.labels)]


class _LinearModel(torch.nn.Module):
    """倒数第二层 = 输入本身，方便手算特征。"""

    def __init__(self, dim=4, num_classes=3, shift=0.0):
        super().__init__()
        self.shift = shift
        self.linear = torch.nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(x + self.shift)


def _probe(n=18, dim=4, num_classes=3):
    # 每类至少 5 个 —— class_prototypes 的 min_class_count 默认就是 5,
    # 少于它原型是 nan，触发器那几列会全是 nan 而不是被测到
    torch.manual_seed(0)
    images = torch.randn(n, dim)
    labels = torch.arange(n) % num_classes
    return _FakeProbe(images, labels)


# ---------------------------------------------------------------------------
# 原型距离
# ---------------------------------------------------------------------------
def test_prototype_distance_is_zero_for_a_single_collapsed_class():
    feats = torch.zeros(18, 4)
    labels = torch.arange(18) % 3
    out = prototype_distances(feats, labels, target_class=0, num_classes=3)
    assert out["target_proto_valid"] == 1.0
    assert np.isclose(out["distance_to_target_proto"], 0.0)


def test_invalid_target_prototype_gives_nan_not_zero():
    """目标类样本太少时原型无效 —— 距离必须是 nan，0 会被读成'完全重合'。"""
    feats = torch.randn(12, 4)
    labels = torch.tensor([1] * 6 + [2] * 6)      # 类 0 一个样本都没有
    out = prototype_distances(feats, labels, target_class=0, num_classes=3)
    assert out["target_proto_valid"] == 0.0
    assert not np.isfinite(out["distance_to_target_proto"])
    # 尺度信息仍然要给
    assert np.isfinite(out["feature_norm_mean"])


# ---------------------------------------------------------------------------
# 配对距离
# ---------------------------------------------------------------------------
def test_drift_against_itself_is_zero():
    probe = _probe()
    model = _LinearModel()
    row = representation_row(model, probe, torch.device("cpu"),
                             target_class=0, num_classes=3)
    feats = row["_feats"]
    again = representation_row(model, probe, torch.device("cpu"),
                               target_class=0, num_classes=3,
                               previous_feats=feats)
    assert np.isclose(again["feature_drift"], 0.0, atol=1e-6)


def test_first_model_has_nan_drift_not_zero():
    """第一个快照没有'前一个'，drift 是 nan。0 会被读成'完全没变'。"""
    row = representation_row(_LinearModel(), _probe(), torch.device("cpu"),
                             target_class=0, num_classes=3)
    assert not np.isfinite(row["feature_drift"])
    assert not np.isfinite(row["clean_repr_distance"])


def test_mismatched_feature_shapes_raise_instead_of_truncating():
    probe = _probe()
    row = representation_row(_LinearModel(), probe, torch.device("cpu"),
                             target_class=0, num_classes=3)
    wrong = row["_feats"][:5]                  # 少了几行
    raised = None
    try:
        representation_row(_LinearModel(), probe, torch.device("cpu"),
                           target_class=0, num_classes=3,
                           previous_feats=wrong)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "配对距离无意义" in raised


# ---------------------------------------------------------------------------
# trigger_pull 的符号
# ---------------------------------------------------------------------------
def test_trigger_pull_is_positive_when_the_trigger_moves_toward_the_target():
    """把样本直接搬到目标类原型上 -> 距离降到 0 -> pull = 原距离 > 0。"""
    probe = _probe(n=18)
    model = _LinearModel()
    base = representation_row(model, probe, torch.device("cpu"),
                              target_class=0, num_classes=3)
    feats = base["_feats"]
    labels = probe.labels
    target_proto = feats[labels == 0].mean(dim=0)

    def to_target(images, targets, _proto=target_proto):
        # 本模型的特征 = 输入，所以把输入设成原型即可
        return _proto[None, :].expand(images.shape[0], -1).clone()

    row = representation_row(model, probe, torch.device("cpu"),
                             target_class=0, num_classes=3,
                             trigger_transform=to_target)
    assert np.isclose(row["trigger_repr_distance"], 0.0, atol=1e-5)
    assert row["trigger_pull"] > 0
    assert np.isclose(row["trigger_pull"], row["target_class_repr_distance"],
                      atol=1e-5)


def test_trigger_pull_is_negative_when_the_trigger_moves_away():
    probe = _probe(n=18)
    model = _LinearModel()
    base = representation_row(model, probe, torch.device("cpu"),
                              target_class=0, num_classes=3)
    target_proto = base["_feats"][probe.labels == 0].mean(dim=0)

    def away(images, targets, _proto=target_proto):
        return images + 50.0 * (images - _proto[None, :])

    row = representation_row(model, probe, torch.device("cpu"),
                             target_class=0, num_classes=3,
                             trigger_transform=away)
    assert row["trigger_pull"] < 0


def test_no_trigger_transform_leaves_trigger_columns_nan():
    row = representation_row(_LinearModel(), _probe(), torch.device("cpu"),
                             target_class=0, num_classes=3)
    assert not np.isfinite(row["trigger_repr_distance"])
    assert not np.isfinite(row["trigger_pull"])


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------
def test_table_computes_drift_against_the_previous_model_in_order():
    probe = _probe()
    models = [_LinearModel(shift=s) for s in (0.0, 0.0, 1.0)]
    table = representation_table(
        models, probe, torch.device("cpu"),
        labels=[{"round": r} for r in (10, 20, 30)],
        target_class=0, num_classes=3)
    assert table["round"].tolist() == [10, 20, 30]
    assert not np.isfinite(table["feature_drift"].iloc[0])
    # 前两个模型完全相同 -> drift = 0；第三个不同 -> drift > 0
    assert np.isclose(table["feature_drift"].iloc[1], 0.0, atol=1e-6)
    assert table["feature_drift"].iloc[2] > 0.0


def test_table_rejects_mismatched_labels():
    raised = None
    try:
        representation_table([_LinearModel()], _probe(), torch.device("cpu"),
                             labels=[{"round": 1}, {"round": 2}],
                             target_class=0, num_classes=3)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "长度不一致" in raised
