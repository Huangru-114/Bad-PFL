"""nan 安全性测试。

本诊断中 nan 是**有意义的信号**（"该客户端没有这个类的样本"），
不是缺失值。所有统计函数必须：跳过 nan、不崩溃、不返回被 nan 污染的结果、
不把退化情形悄悄变成 0 或 inf。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from diag.features import (class_prototypes, fisher_dispersion, knn_overlap,
                           margin_matrix)
from diag.metrics import observable_score
from diag.nanstats import (nan_cv, nanmad, nanmean, nanmedian, nanstd,
                           nan_to_num_count)

NAN = float("nan")


def test_nanmean_skips_nan():
    x = torch.tensor([1.0, NAN, 3.0])
    assert abs(float(nanmean(x)) - 2.0) < 1e-6


def test_nanstd_matches_numpy_on_clean_input():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = float(np.std(x.numpy(), ddof=1))
    assert abs(float(nanstd(x)) - expected) < 1e-5


def test_nanstd_skips_nan():
    x = torch.tensor([1.0, NAN, 2.0, 3.0, 4.0])
    expected = float(np.nanstd(np.array([1.0, 2.0, 3.0, 4.0]), ddof=1))
    assert abs(float(nanstd(x)) - expected) < 1e-5


def test_nanstd_insufficient_samples_returns_nan():
    assert math.isnan(float(nanstd(torch.tensor([5.0]))))
    assert math.isnan(float(nanstd(torch.tensor([NAN, NAN]))))


def test_nanstd_along_dim_with_ragged_nan():
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [1.0, NAN, 3.0],
                      [NAN, NAN, NAN]])
    out = nanstd(x, dim=1)
    assert out.shape == (3,)
    assert torch.isfinite(out[0]) and torch.isfinite(out[1])
    assert math.isnan(float(out[2])), "整行 nan 必须返回 nan"


def test_nanmedian_and_nanmad_skip_nan():
    x = torch.tensor([1.0, NAN, 3.0, 5.0])
    assert float(nanmedian(x)) == 3.0
    assert float(nanmad(x)) == 2.0


def test_nanmad_zero_for_constant_input():
    """常数输入的 MAD 是 0 —— 下游必须自己防除零，这里只确认数值正确。"""
    assert float(nanmad(torch.tensor([2.0, 2.0, 2.0]))) == 0.0


def test_nan_cv_zero_mean_returns_nan_not_inf():
    x = torch.tensor([-1.0, 1.0])
    value = float(nan_cv(x))
    assert math.isnan(value), "均值为 0 时 CV 必须是 nan 而不是 inf"


def test_nan_cv_normal_case():
    x = torch.tensor([1.0, 2.0, 3.0, NAN])
    expected = float(np.nanstd(np.array([1.0, 2.0, 3.0]), ddof=1) / 2.0)
    assert abs(float(nan_cv(x)) - expected) < 1e-5


def test_nan_to_num_count():
    x = torch.tensor([[1.0, NAN], [NAN, NAN]])
    counts = nan_to_num_count(x, dim=1)
    assert counts.tolist() == [1.0, 0.0]


def test_no_function_returns_inf_on_degenerate_input():
    """退化情形一律返回 nan，绝不返回 inf —— inf 会污染下游聚合。"""
    degenerate = torch.tensor([0.0, 0.0, 0.0])
    for value in (nan_cv(degenerate), nanstd(torch.tensor([NAN]))):
        assert not math.isinf(float(value))


def test_class_prototypes_all_nan_labels_bucket():
    feats = torch.randn(6, 4)
    labels = torch.zeros(6, dtype=torch.long)
    stats = class_prototypes(feats, labels, num_classes=3, min_class_count=2)
    assert torch.isnan(stats["proto"][1]).all()
    assert torch.isnan(stats["proto"][2]).all()
    # 后续统计必须能在含 nan 的原型上工作而不崩溃
    dispersion = fisher_dispersion(stats["proto"], stats["scatter"], stats["count"])
    assert dispersion.shape == (3,)


def test_margin_matrix_all_nan_prototypes_does_not_crash():
    proto = torch.full((4, 5), NAN)
    W = torch.randn(4, 5)
    b = torch.zeros(4)
    margins = margin_matrix(proto, W, b)
    assert margins.shape == (4, 4)
    assert torch.isnan(margins).all()


def test_observable_score_with_partially_nan_matrices():
    """部分 nan 不应传染到本可计算的客户端。"""
    generator = torch.Generator().manual_seed(0)
    matrices = []
    for _ in range(5):
        matrix = 1.0 + 0.1 * torch.randn(4, 4, generator=generator)
        matrix.fill_diagonal_(NAN)
        matrices.append(matrix)
    matrices[2][1, :] = NAN          # 一个客户端的某个类原型无效

    scores = observable_score(matrices)
    assert torch.isfinite(scores[0]), "其他客户端不应被污染"
    assert torch.isfinite(scores[2]), "部分 nan 的客户端仍应有有效得分"


def test_observable_score_constant_margins_gives_nan_not_inf():
    """所有客户端完全相同 -> MAD = 0 -> z 无定义 -> 必须 nan 而不是 inf。"""
    matrix = torch.ones(4, 4)
    matrix.fill_diagonal_(NAN)
    scores = observable_score([matrix.clone() for _ in range(5)])
    assert not torch.isinf(scores).any()
    assert torch.isnan(scores).all()


def test_knn_overlap_nan_on_empty_inputs():
    assert math.isnan(knn_overlap(torch.randn(4, 3), torch.zeros(0, 3),
                                  torch.randn(4, 3), k=2))
    assert math.isnan(knn_overlap(torch.zeros(0, 3), torch.randn(4, 3),
                                  torch.randn(4, 3), k=2))


def test_analysis_helpers_are_nan_safe():
    from diag.analysis import auc_score, pearson_spearman

    scores = [1.0, NAN, 3.0, 4.0]
    labels = [False, True, True, True]
    value = auc_score(scores, labels)
    assert 0.0 <= value <= 1.0, "nan 分数应被剔除而不是让 AUC 变成 nan"

    # 单一类别 -> AUC 无定义
    assert math.isnan(auc_score([1.0, 2.0], [True, True]))
    # 点数不足 -> 相关系数无定义
    assert math.isnan(pearson_spearman([1.0, 2.0], [1.0, 2.0])[0])
