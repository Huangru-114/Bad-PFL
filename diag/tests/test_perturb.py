"""``diag.perturb`` 的单元测试：形状、像素范围、L∞ 预算、缺组件时的异常。"""

from __future__ import annotations

import torch
import torch.nn as nn

from diag.perturb import (MODES, apply_perturbation, delta_from_generator, linf,
                          make_delta_fn, make_xi_fn)

EPS = 4.0 / 255.0


class TinyGenerator(nn.Module):
    """模拟 ``generator.Autoencoder``：同形状输出 + Tanh 收尾（故有界）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return torch.tanh(self.conv(x) * 5.0)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def _fixture():
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(6, 3, 8, 8, generator=generator)
    labels = torch.randint(0, 4, (6,), generator=generator)
    gen = TinyGenerator()
    model = TinyClassifier()
    delta_fn = make_delta_fn(gen, eps=EPS)
    xi_fn = make_xi_fn(model, eps=EPS, num_iter=1, seed=123)
    return images, labels, gen, model, delta_fn, xi_fn


def test_all_modes_preserve_shape_and_pixel_range():
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    rng = torch.Generator().manual_seed(7)
    for mode in MODES:
        out = apply_perturbation(images, mode, delta_fn=delta_fn, xi_fn=xi_fn,
                                 eps=EPS, labels=labels, rng=rng)
        assert out.shape == images.shape, f"模式 {mode} 改变了形状"
        assert float(out.min()) >= 0.0 - 1e-6, f"模式 {mode} 像素下溢"
        assert float(out.max()) <= 1.0 + 1e-6, f"模式 {mode} 像素上溢"


def test_clean_mode_is_identity():
    images, labels, *_ = _fixture()
    out = apply_perturbation(images, "clean")
    assert torch.allclose(out, images)
    assert out is not images, "应返回副本而不是原张量"


def test_random_noise_linf_equals_eps():
    """负对照必须与 δ 共用同一个 L∞ 预算，否则比较不公平。"""
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    # 用中间灰度图避免 clamp 削掉扰动，从而能精确测到 L∞
    images = torch.full_like(images, 0.5)
    rng = torch.Generator().manual_seed(3)
    out = apply_perturbation(images, "random_noise", eps=EPS, rng=rng)
    assert abs(linf(out, images) - EPS) < 1e-6


def test_delta_linf_within_budget():
    """δ 由 Tanh 收尾，逐元素满足 |δ| <= eps（可能小于，因为 tanh 未必饱和）。"""
    images, labels, gen, *_ = _fixture()
    delta = delta_from_generator(gen, images, eps=EPS)
    assert delta.shape == images.shape
    assert float(delta.abs().max()) <= EPS + 1e-6


def test_xi_linf_within_budget():
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    images = torch.full_like(images, 0.5)
    out = apply_perturbation(images, "xi_only", xi_fn=xi_fn, labels=labels)
    assert linf(out, images) <= EPS + 1e-6


def test_delta_plus_xi_budget_is_sum():
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    images = torch.full_like(images, 0.5)
    out = apply_perturbation(images, "delta_plus_xi", delta_fn=delta_fn,
                             xi_fn=xi_fn, labels=labels)
    assert linf(out, images) <= 2 * EPS + 1e-6


def test_xi_fn_is_reproducible_with_fixed_seed():
    """固定 seed 后 ξ 必须可复现 —— 原实现的随机起点是不固定的。"""
    images, labels, gen, model, *_ = _fixture()
    first = make_xi_fn(model, eps=EPS, seed=42)(images, labels)
    second = make_xi_fn(model, eps=EPS, seed=42)(images, labels)
    assert torch.allclose(first, second), "同 seed 的 ξ 必须完全一致"


def test_xi_fn_does_not_leave_gradients_on_model():
    """``pgd_attack`` 内部的 backward 会把梯度累加进模型参数，必须被清掉。"""
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    xi_fn(images, labels)
    for param in model.parameters():
        assert param.grad is None, "诊断路径不应在模型上留下梯度副作用"


def test_xi_fn_restores_model_training_mode():
    images, labels, gen, model, delta_fn, xi_fn = _fixture()
    model.train()
    xi_fn(images, labels)
    assert model.training is True


def test_missing_components_raise_clearly():
    images, labels, gen, model, delta_fn, xi_fn = _fixture()

    for mode in ("delta_only", "delta_plus_xi"):
        try:
            apply_perturbation(images, mode, xi_fn=xi_fn, labels=labels)
        except ValueError as exc:
            assert "delta" in str(exc)
        else:
            raise AssertionError(f"模式 {mode} 缺少 delta 时应抛 ValueError")

    for mode in ("xi_only", "delta_plus_xi"):
        try:
            apply_perturbation(images, mode, delta_fn=delta_fn, labels=labels)
        except ValueError as exc:
            assert "xi_fn" in str(exc)
        else:
            raise AssertionError(f"模式 {mode} 缺少 xi_fn 时应抛 ValueError")

    # 需要 ξ 却没给标签
    try:
        apply_perturbation(images, "xi_only", xi_fn=xi_fn)
    except ValueError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("缺少 labels 时应抛 ValueError")

    # random_noise 没给 eps
    try:
        apply_perturbation(images, "random_noise")
    except ValueError as exc:
        assert "eps" in str(exc)
    else:
        raise AssertionError("random_noise 缺少 eps 时应抛 ValueError")


def test_unknown_mode_raises():
    images, *_ = _fixture()
    try:
        apply_perturbation(images, "not_a_mode")
    except ValueError as exc:
        assert "not_a_mode" in str(exc)
    else:
        raise AssertionError("未知模式应抛 ValueError")


def test_delta_shape_mismatch_raises():
    images, labels, *_ = _fixture()
    bad_delta = torch.zeros(6, 3, 4, 4)
    try:
        apply_perturbation(images, "delta_only", delta=bad_delta)
    except ValueError as exc:
        assert "形状" in str(exc)
    else:
        raise AssertionError("delta 形状不匹配时应抛 ValueError")


# ---------------------------------------------------------------------------
# 末端 clamp 的忠实性（2026-09 审计）
# ---------------------------------------------------------------------------
def test_clamp_flag_reproduces_the_original_unclamped_composition():
    """`fba.our_poison_func`（fba.py:55）加完 δ 之后**不** clamp。

    默认 clamp=True 会削掉贴边像素上的那部分 δ，使重建的触发器弱于真实攻击；
    clamp=False 才是原实现的口径。两条路径必须真的不同，否则这个开关是假的。
    """
    import torch
    from diag.perturb import apply_perturbation

    images = torch.zeros(1, 3, 4, 4)          # 全 0：加负向 δ 必然越界
    delta = torch.full_like(images, -4.0 / 255.0)
    clamped = apply_perturbation(images, "delta_only", delta=delta)
    raw = apply_perturbation(images, "delta_only", delta=delta, clamp=False)
    assert torch.allclose(clamped, torch.zeros_like(images))   # 全被削掉
    assert torch.allclose(raw, delta)                          # 原样保留
    assert not torch.allclose(clamped, raw)


def test_clip_diagnostics_quantifies_what_the_clamp_removed():
    import torch
    from diag.perturb import apply_perturbation

    images = torch.zeros(1, 3, 2, 2)
    delta = torch.full_like(images, -0.1)
    diagnostics = {}
    apply_perturbation(images, "delta_only", delta=delta,
                       diagnostics=diagnostics)
    # 12 个像素全部越界 -> 100% 被改动，扰动被 100% 削掉
    assert np.isclose(diagnostics["clipped_fraction"], 1.0)
    assert np.isclose(diagnostics["clipped_l1_ratio"], 1.0)
    assert np.isclose(diagnostics["linf_before_clip"], 0.1)
    assert np.isclose(diagnostics["linf_after_clip"], 0.0)


def test_clip_diagnostics_is_all_zero_when_nothing_is_clipped():
    import torch
    from diag.perturb import apply_perturbation

    images = torch.full((1, 3, 2, 2), 0.5)    # 远离边界
    delta = torch.full_like(images, 0.1)
    diagnostics = {}
    out = apply_perturbation(images, "delta_only", delta=delta,
                             diagnostics=diagnostics)
    assert np.isclose(diagnostics["clipped_fraction"], 0.0)
    assert np.isclose(diagnostics["clipped_l1_ratio"], 0.0)
    assert torch.allclose(out, images + delta)
