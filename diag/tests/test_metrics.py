"""``diag.metrics`` 的单元测试。

重点是 ``observable_score``：在人工构造的"某一列被压低"的边距矩阵上必须给出
高分，在纯随机矩阵上必须给出低分。
"""

from __future__ import annotations

import math

import torch

from diag.metrics import baseline_signals, observable_score


def _baseline_margins(num_clients: int = 8, num_classes: int = 5,
                      seed: int = 0) -> list:
    """构造一批"正常"的边距矩阵：基线 1.0 + 小噪声，对角线 nan。"""
    generator = torch.Generator().manual_seed(seed)
    matrices = []
    for _ in range(num_clients):
        matrix = 1.0 + 0.1 * torch.randn(num_classes, num_classes, generator=generator)
        matrix.fill_diagonal_(float("nan"))
        matrices.append(matrix)
    return matrices


def test_observable_score_shape_and_dtype():
    matrices = _baseline_margins()
    scores = observable_score(matrices, lambda_std=1.0)
    assert scores.shape == (len(matrices),)
    assert scores.dtype.is_floating_point


def test_observable_score_flags_depressed_column():
    """客户端 0 把所有类到类 2 的边距整体压低 -> 其 A 应显著高于其他客户端。"""
    matrices = _baseline_margins(num_clients=8, num_classes=5, seed=0)
    target = 2
    for c in range(5):
        if c != target:
            matrices[0][c, target] -= 3.0

    scores = observable_score(matrices, lambda_std=1.0)
    others = torch.cat([scores[1:2], scores[2:]])
    assert scores[0] > others.max() + 1.0, (
        f"被压低目标列的客户端应显著突出: scores={scores.tolist()}")


def test_observable_score_low_on_random_matrices():
    """纯随机矩阵上不应出现突出的高分。"""
    matrices = _baseline_margins(num_clients=8, num_classes=5, seed=7)
    scores = observable_score(matrices, lambda_std=1.0)
    finite = scores[torch.isfinite(scores)]
    assert len(finite) == 8
    assert float(finite.max()) < 5.0, f"随机矩阵不应给出高分: {finite.tolist()}"


def test_observable_score_uniform_depression_beats_noisy_depression():
    """同样压低，压得越均匀（std 越小）得分越高 —— 这是 lambda_std 项的作用。"""
    target = 2
    uniform = _baseline_margins(num_clients=8, num_classes=5, seed=1)
    noisy = [m.clone() for m in uniform]

    generator = torch.Generator().manual_seed(99)
    for c in range(5):
        if c == target:
            continue
        uniform[0][c, target] -= 3.0
        noisy[0][c, target] -= 3.0 + 3.0 * torch.randn(1, generator=generator).item()

    score_uniform = observable_score(uniform, lambda_std=1.0)[0]
    score_noisy = observable_score(noisy, lambda_std=1.0)[0]
    assert score_uniform > score_noisy


def test_observable_score_empty_input():
    assert observable_score([]).shape == (0,)


def test_observable_score_all_nan_client_is_nan():
    matrices = _baseline_margins(num_clients=4, num_classes=4)
    matrices[1] = torch.full((4, 4), float("nan"))
    scores = observable_score(matrices)
    assert math.isnan(float(scores[1]))
    assert torch.isfinite(scores[0])


def test_observable_score_signature_excludes_privileged_inputs():
    """信息隔离约束：签名里不允许出现 delta / clean_model / target_class。

    这条约束是方法有效性的前提，必须在代码层面可检查。
    """
    import inspect

    parameters = set(inspect.signature(observable_score).parameters)
    forbidden = {"delta", "delta_fn", "clean_model", "target_class", "generator",
                 "labels", "is_malicious"}
    assert not (parameters & forbidden), (
        f"observable_score 不得接收特权信息，但发现: {parameters & forbidden}")


def test_baseline_signals_columns_and_ordering():
    generator = torch.Generator().manual_seed(5)
    updates = [torch.randn(20, generator=generator) for _ in range(6)]
    # 客户端 0 是明显的离群点
    updates[0] = updates[0] + 50.0

    frame = baseline_signals(updates)
    assert list(frame.columns) == ["client_id", "l2_to_median",
                                   "sign_consistency", "consensus_strength"]
    assert len(frame) == 6
    assert frame.loc[0, "l2_to_median"] == frame["l2_to_median"].max(), (
        "离群客户端应有最大的 L2 距离")
    assert (frame["sign_consistency"].abs() <= 1.0 + 1e-6).all()


def test_baseline_signals_empty_input():
    frame = baseline_signals([])
    assert len(frame) == 0
    assert "l2_to_median" in frame.columns
