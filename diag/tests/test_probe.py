"""``diag.probe.build_probe_set`` 的单元测试：不相交、规模、确定性、标签纯度。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from diag.probe import assert_disjoint, build_probe_set, get_targets

NUM_CLASSES = 10
TARGET_CLASS = 0

PROBE_CFG = {"probe": {"seed": 12345, "n_ref": 30, "n_other_per_class": 8,
                       "n_query": 40, "n_target": 20}}


class FakeDataset(Dataset):
    """每类 ``per_class`` 个样本的假数据集。"""

    def __init__(self, per_class: int = 100, num_classes: int = NUM_CLASSES):
        self.targets = [cls for cls in range(num_classes) for _ in range(per_class)]
        self.data = torch.arange(len(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def test_get_targets_returns_int_array():
    targets = get_targets(FakeDataset(per_class=5))
    assert isinstance(targets, np.ndarray)
    assert targets.dtype == np.int64
    assert len(targets) == 50


def test_subsets_are_pairwise_disjoint():
    probe = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    # 不相交是硬要求：重叠会让 kNN 匹配到底图相同的样本，overlap 静默虚高。
    assert_disjoint(probe.ref, probe.other, probe.query, probe.target)

    all_indices = sum((probe.indices[name]
                       for name in ("ref", "other", "query", "target")), [])
    assert len(all_indices) == len(set(all_indices))


def test_subset_sizes_match_config():
    probe = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    cfg = PROBE_CFG["probe"]
    assert len(probe.ref) == cfg["n_ref"]
    assert len(probe.target) == cfg["n_target"]
    assert len(probe.query) == cfg["n_query"]
    assert len(probe.other) == cfg["n_other_per_class"] * (NUM_CLASSES - 1)


def test_same_seed_gives_identical_indices():
    """探针集必须在所有 run 之间完全一致，否则比较不是 apples-to-apples。"""
    first = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    second = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    for name in ("ref", "other", "query", "target"):
        assert first.indices[name] == second.indices[name]


def test_different_seed_gives_different_indices():
    other_cfg = {"probe": dict(PROBE_CFG["probe"], seed=999)}
    first = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    second = build_probe_set(FakeDataset(), TARGET_CLASS, other_cfg)
    assert first.indices["ref"] != second.indices["ref"]


def test_build_probe_set_does_not_disturb_global_numpy_rng():
    """探针抽样必须用独立 RNG，否则会扰乱数据划分的随机流。"""
    np.random.seed(0)
    expected = np.random.rand(3).tolist()

    np.random.seed(0)
    build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    actual = np.random.rand(3).tolist()

    assert expected == actual


def test_ref_and_target_are_all_target_class():
    dataset = FakeDataset()
    probe = build_probe_set(dataset, TARGET_CLASS, PROBE_CFG)
    targets = get_targets(dataset)
    assert (targets[probe.indices["ref"]] == TARGET_CLASS).all()
    assert (targets[probe.indices["target"]] == TARGET_CLASS).all()


def test_query_and_other_contain_no_target_class():
    dataset = FakeDataset()
    probe = build_probe_set(dataset, TARGET_CLASS, PROBE_CFG)
    targets = get_targets(dataset)
    assert (targets[probe.indices["query"]] != TARGET_CLASS).all()
    assert (targets[probe.indices["other"]] != TARGET_CLASS).all()


def test_other_is_class_balanced():
    dataset = FakeDataset()
    probe = build_probe_set(dataset, TARGET_CLASS, PROBE_CFG)
    targets = get_targets(dataset)
    counts = np.bincount(targets[probe.indices["other"]], minlength=NUM_CLASSES)
    per_class = PROBE_CFG["probe"]["n_other_per_class"]
    for cls in range(NUM_CLASSES):
        assert counts[cls] == (0 if cls == TARGET_CLASS else per_class)


def test_insufficient_samples_raises_clearly():
    tiny = FakeDataset(per_class=5)
    try:
        build_probe_set(tiny, TARGET_CLASS, PROBE_CFG)
    except ValueError as exc:
        assert "不足" in str(exc)
    else:
        raise AssertionError("样本不足时必须抛 ValueError，而不是静默截断")


def test_missing_target_class_raises():
    dataset = FakeDataset()
    try:
        build_probe_set(dataset, 999, PROBE_CFG)
    except ValueError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("目标类不存在时必须抛 ValueError")


def test_assert_disjoint_detects_overlap():
    from torch.utils.data import Subset

    dataset = FakeDataset()
    a = Subset(dataset, [1, 2, 3])
    b = Subset(dataset, [3, 4, 5])
    try:
        assert_disjoint(a, b)
    except ValueError as exc:
        assert "重叠" in str(exc)
    else:
        raise AssertionError("重叠时必须抛 ValueError")


def test_loaders_cover_every_sample():
    probe = build_probe_set(FakeDataset(), TARGET_CLASS, PROBE_CFG)
    loaders = probe.loaders(batch_size=7)   # 故意选不整除的 batch size
    for name in ("ref", "other", "query", "target"):
        total = sum(batch[0].shape[0] for batch in loaders[name])
        assert total == len(getattr(probe, name)), (
            f"{name} 的 loader 丢样本了（drop_last 泄漏？）")
