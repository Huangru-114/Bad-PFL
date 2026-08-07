"""``diag.features`` 的单元测试：形状、nan 语义、kNN 的极端行为。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from diag.features import (class_prototypes, extract_penultimate,
                           find_last_linear, fisher_dispersion, knn_overlap,
                           margin_matrix)


class TinyNet(nn.Module):
    """两层小网络：``backbone`` 输出即倒数第二层特征。"""

    def __init__(self, in_dim: int = 12, feat_dim: int = 8, num_classes: int = 4):
        super().__init__()
        self.backbone = nn.Linear(in_dim, feat_dim)
        self.linear = nn.Linear(feat_dim, num_classes)   # 最后一个 Linear = 分类头

    def forward(self, x):
        return self.linear(torch.relu(self.backbone(x.flatten(1))))


def _tiny_loader(n: int = 40, in_dim: int = 12, num_classes: int = 4,
                 batch_size: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, in_dim, generator=generator)
    y = torch.randint(0, num_classes, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_find_last_linear_picks_classifier_head():
    model = TinyNet()
    assert find_last_linear(model) is model.linear


def test_extract_penultimate_shapes_and_meta():
    model = TinyNet()
    out = extract_penultimate(model, _tiny_loader(), torch.device("cpu"),
                              source="probe")
    assert out.feats.shape == (40, 8)
    assert out.labels.shape == (40,)
    assert out.logits.shape == (40, 4)
    assert out.meta["source"] == "probe"
    assert out.meta["feat_dim"] == 8
    assert out.meta["n_samples"] == 40


def test_extract_penultimate_forces_eval_and_restores_mode():
    """特征提取必须在 eval 下进行，否则 BN 统计量污染结果；且不能有副作用。"""
    model = TinyNet()
    model.train()
    seen = {}

    original_forward = model.linear.forward

    def spy(x):
        seen["training"] = model.training
        return original_forward(x)

    model.linear.forward = spy
    extract_penultimate(model, _tiny_loader(), torch.device("cpu"), source="probe")
    assert seen["training"] is False, "前向时模型必须处于 eval 模式"
    assert model.training is True, "提取结束后必须恢复调用前的 train/eval 状态"


def test_class_prototypes_shapes_and_empty_class_is_nan():
    feats = torch.randn(30, 6)
    labels = torch.cat([torch.zeros(20, dtype=torch.long),
                        torch.ones(10, dtype=torch.long)])
    stats = class_prototypes(feats, labels, num_classes=4, min_class_count=5)

    assert stats["proto"].shape == (4, 6)
    assert stats["count"].shape == (4,)
    assert stats["scatter"].shape == (4,)
    assert stats["count"][0] == 20 and stats["count"][1] == 10
    # 类 2、3 一个样本都没有 -> proto / scatter 必须是 nan，不能是 0
    assert torch.isnan(stats["proto"][2]).all()
    assert torch.isnan(stats["proto"][3]).all()
    assert torch.isnan(stats["scatter"][2])
    assert not torch.isnan(stats["proto"][0]).any()


def test_class_prototypes_respects_min_class_count():
    feats = torch.randn(12, 5)
    labels = torch.cat([torch.zeros(10, dtype=torch.long),
                        torch.ones(2, dtype=torch.long)])
    stats = class_prototypes(feats, labels, num_classes=2, min_class_count=5)
    assert not torch.isnan(stats["proto"][0]).any()
    assert torch.isnan(stats["proto"][1]).all(), "样本数低于阈值的类必须填 nan"
    assert stats["count"][1] == 2, "count 应保留真实计数，不受阈值影响"


def test_fisher_dispersion_nan_propagation():
    proto = torch.tensor([[0.0, 0.0], [3.0, 4.0], [float("nan"), float("nan")]])
    scatter = torch.tensor([2.0, 8.0, float("nan")])
    count = torch.tensor([10.0, 10.0, 0.0])
    dispersion = fisher_dispersion(proto, scatter, count)

    assert dispersion.shape == (3,)
    assert torch.isnan(dispersion[2]), "无效类必须返回 nan"
    assert torch.isfinite(dispersion[0]) and torch.isfinite(dispersion[1])
    assert (dispersion[0] > 0) and (dispersion[1] > 0)


def test_fisher_dispersion_degenerate_denominator_is_nan():
    """所有原型重合 -> 分母为 0 -> 必须是 nan 而不是 inf。"""
    proto = torch.zeros(3, 4)
    scatter = torch.tensor([1.0, 1.0, 1.0])
    count = torch.tensor([5.0, 5.0, 5.0])
    dispersion = fisher_dispersion(proto, scatter, count)
    assert torch.isnan(dispersion).all()
    assert not torch.isinf(dispersion).any()


def test_margin_matrix_diagonal_is_nan_and_shape():
    num_classes, dim = 5, 7
    generator = torch.Generator().manual_seed(1)
    proto = torch.randn(num_classes, dim, generator=generator)
    W = torch.randn(num_classes, dim, generator=generator)
    b = torch.randn(num_classes, generator=generator)

    margins = margin_matrix(proto, W, b)
    assert margins.shape == (num_classes, num_classes)
    assert torch.isnan(torch.diagonal(margins)).all(), "对角线必须是 nan"
    off_diagonal = ~torch.eye(num_classes, dtype=torch.bool)
    assert torch.isfinite(margins[off_diagonal]).all()


def test_margin_matrix_nan_row_for_invalid_prototype():
    proto = torch.randn(3, 4)
    proto[1] = float("nan")
    W = torch.randn(3, 4)
    b = torch.zeros(3)
    margins = margin_matrix(proto, W, b)
    assert torch.isnan(margins[1]).all(), "原型无效的类整行必须是 nan"
    assert torch.isfinite(margins[0, 2])


def test_margin_matrix_sign_matches_geometry():
    """原型明显偏向类 0 时，M[0, 1] 应为正。"""
    proto = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.zeros(2)
    margins = margin_matrix(proto, W, b)
    assert margins[0, 1] > 0
    assert margins[1, 0] > 0


def test_knn_overlap_perfect_and_zero_overlap():
    generator = torch.Generator().manual_seed(2)
    # ref 与 query 来自同一个紧簇；other 远在天边 -> overlap 应接近 1
    ref = torch.randn(100, 4, generator=generator) * 0.1
    query_inside = torch.randn(50, 4, generator=generator) * 0.1
    other_far = torch.randn(100, 4, generator=generator) * 0.1 + 100.0

    inside = knn_overlap(query_inside, ref, other_far, k=10)
    assert inside > 0.95, f"完全重合时 overlap 应接近 1.0，实际 {inside}"

    # query 挪到 other 那边 -> overlap 应接近 0
    query_outside = torch.randn(50, 4, generator=generator) * 0.1 + 100.0
    outside = knn_overlap(query_outside, ref, other_far, k=10)
    assert outside < 0.05, f"完全分离时 overlap 应接近 0.0，实际 {outside}"


def test_knn_overlap_empty_ref_returns_nan():
    query = torch.randn(10, 3)
    empty = torch.zeros(0, 3)
    other = torch.randn(10, 3)
    assert math.isnan(knn_overlap(query, empty, other, k=5))


def test_knn_overlap_k_larger_than_bank_degrades_gracefully():
    """k 超过参照库大小时退化为使用全部参照点，不应崩溃。"""
    query = torch.randn(5, 3)
    ref = torch.randn(3, 3)
    other = torch.randn(2, 3)
    value = knn_overlap(query, ref, other, k=50)
    assert 0.0 <= value <= 1.0
    # 库共 5 个点、其中 3 个属于 ref -> 全取时比例恒为 0.6
    assert abs(value - 0.6) < 1e-6
