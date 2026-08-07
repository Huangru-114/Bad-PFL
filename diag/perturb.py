"""扰动施加：clean / xi_only / delta_only / delta_plus_xi / random_noise。

# δ 与 ξ 在原仓库中的对应关系（fba.py:46-58）

    poison_data = pgd_attack(client.local_model, poison_data, label)  # 行53 -> x + ξ
    gen_trigger = trigger_gen(data) / 255. * 4.                       # 行54 -> δ(x)
    poison_data = mask * (poison_data + gen_trigger) + ~mask * data   # 行55 -> x + ξ + δ

| 组件 | 表达式                          | L∞ 预算 | 性质 |
|------|--------------------------------|---------|------|
| ξ    | ``pgd_attack(model, x, y) - x``| 4/255   | 无目标 PGD，破坏真实类特征 |
| δ    | ``generator(x) * (4/255)``     | 4/255   | 生成器输出，Tanh 收尾故有界 |

两者**完全可以独立取出**。

# 三个必须记住的性质

1. **δ 是输入条件化的**，不是一个固定张量：``δ = generator(x)``，每张图片不同。
   "同一个 δ 施加到不同客户端"的正确说法是"同一生成器 + 同一批探针图片
   → 同一份 δ，施加到不同客户端的模型上"。
2. **ξ 是模型相关的**：原实现用**该客户端自己的 local_model** 现算
   （fba.py:53）。本模块保持这个语义 —— ``xi_fn`` 显式绑定一个模型，
   调用方决定绑哪个。
3. **ξ 是随机的**：``pgd_attack`` 以 ``uniform_(-eps, eps)`` 随机起点开始
   （fba.py:8），原实现未固定随机流。本模块通过保存/恢复全局 RNG 状态
   使其可复现，**不改动 pgd_attack 本身的数学**。

# 与原实现的两处刻意偏离（均记录在 PATCHES.md）

- 生成 ξ 时模型置于 ``.eval()``。原投毒路径中模型处于 ``.train()``
  （client.py:35 先于 :36），BatchNorm 会用 batch 统计量并更新 running stats。
  诊断必须用 eval，否则特征被污染（这是 spec 明确的陷阱）。
- 调用 ``pgd_attack`` 后清零模型参数梯度。原实现中 ``loss.backward()``
  会把梯度累加进模型参数（fba.py:16），而 ``local_update`` 的
  ``zero_grad()`` 发生在取数**之前**（client.py:34-36）。诊断只做读操作，
  必须避免这个副作用。
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional

import torch
import torch.nn as nn

import fba  # 原仓库模块；直接复用 pgd_attack，不重写，避免实现漂移

__all__ = ["MODES", "MODES_REQUIRING_DELTA", "MODES_REQUIRING_XI",
           "delta_from_generator", "make_xi_fn", "make_delta_fn",
           "apply_perturbation", "linf"]

MODES = ("clean", "xi_only", "delta_only", "delta_plus_xi", "random_noise")
MODES_REQUIRING_DELTA = ("delta_only", "delta_plus_xi")
MODES_REQUIRING_XI = ("xi_only", "delta_plus_xi")

_DEFAULT_EPS = 4.0 / 255.0


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    """两张图之间的 L∞ 距离（标量）。用于预算校验与单元测试。"""
    return float((a - b).abs().max().item())


@contextlib.contextmanager
def _temporary_torch_seed(seed: Optional[int]):
    """在保存/恢复全局 torch RNG 状态的前提下临时设定种子。

    这样既让 ``pgd_attack`` 的随机起点可复现，又不会扰乱调用方所在的随机流。
    ``seed is None`` 时不做任何事（沿用原实现的不确定行为）。
    """
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


def delta_from_generator(generator: nn.Module, images: torch.Tensor,
                         eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """δ(x) = ``generator(x) * eps``。

    与 fba.py:54 的 ``trigger_gen(data) / 255. * 4.`` **数值等价**
    （``/255.*4.`` 就是 ``*(4/255)``）。生成器以 ``Tanh`` 收尾
    （generator.py:34），故 ``|δ| <= eps`` 逐元素成立。

    返回的是**扰动量**本身（不是加过扰动的图），也不做 clamp ——
    clamp 由 ``apply_perturbation`` 在合成之后统一执行。
    """
    was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            return generator(images) * float(eps)
    finally:
        if was_training:
            generator.train()


def make_delta_fn(generator: nn.Module, eps: float = _DEFAULT_EPS
                  ) -> Callable[[torch.Tensor], torch.Tensor]:
    """把生成器包装成 ``images -> δ`` 的闭包。"""

    def delta_fn(images: torch.Tensor) -> torch.Tensor:
        return delta_from_generator(generator, images, eps=eps)

    return delta_fn


def make_xi_fn(model: nn.Module, eps: float = _DEFAULT_EPS,
               alpha: Optional[float] = None, num_iter: int = 1,
               seed: Optional[int] = None, use_eval_mode: bool = True
               ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """构造 ``(images, labels) -> images + ξ`` 的闭包，内部调用原
    ``fba.pgd_attack``。

    Parameters
    ----------
    model:
        生成 ξ 所依据的模型。**原实现用的是该客户端自己的 local_model**
        （fba.py:53），跨客户端测量时调用方需自行决定绑定哪个模型，
        并在结果中如实记录。
    seed:
        随机起点的基准种子。每次调用使用 ``seed + call_index``，
        因此同一个 ``xi_fn`` 在同一批次序列上完全可复现。
        传 ``None`` 则保持原实现的不确定行为。

    Notes
    -----
    - ξ 的计算需要**真实标签**（无目标 PGD 对真实标签做梯度上升）。
    - 返回值是 ``x + ξ``，已被 ``pgd_attack`` 内部 clamp 到 [0, 1]。
    - 调用后会清零 ``model`` 的参数梯度（见模块 docstring）。
    """
    alpha = float(eps if alpha is None else alpha)
    counter = {"n": 0}

    def xi_fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        was_training = model.training
        if use_eval_mode:
            model.eval()
        call_seed = None if seed is None else int(seed) + counter["n"]
        counter["n"] += 1
        try:
            with _temporary_torch_seed(call_seed):
                # 原函数，未做任何数学上的改动。
                adv = fba.pgd_attack(model, images, labels.long(),
                                     epsilon=float(eps), alpha=alpha,
                                     num_iter=int(num_iter))
        finally:
            # pgd_attack 内部的 backward 会把梯度累加进模型参数，必须清掉。
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()
        return adv.detach()

    return xi_fn


def apply_perturbation(images: torch.Tensor, mode: str,
                       delta: Optional[torch.Tensor] = None,
                       xi_fn: Optional[Callable] = None,
                       eps: Optional[float] = None,
                       generator: Optional[nn.Module] = None,
                       labels: Optional[torch.Tensor] = None,
                       delta_fn: Optional[Callable] = None,
                       rng: Optional[torch.Generator] = None,
                       pixel_min: float = 0.0,
                       pixel_max: float = 1.0) -> torch.Tensor:
    """按 ``mode`` 施加扰动，返回与输入同形状的图像张量。

    Parameters
    ----------
    delta:
        预先算好的 δ **扰动量**（与 images 同形状）。与 ``generator`` /
        ``delta_fn`` 三选一。
    xi_fn:
        ``(images, labels) -> images + ξ``。由 ``make_xi_fn`` 构造。
    labels:
        真实标签。**凡是需要 ξ 的模式都必须提供**（无目标 PGD 需要真实标签）。
        这是相对于原始 spec 签名的一处附加参数。
    eps:
        L∞ 预算。``random_noise`` 与 δ 必须共用同一个预算 —— 负对照的公平性
        要求，否则"随机噪声不像目标类"可能只是因为它更弱。

    Notes
    -----
    - ``random_noise`` 使用 ``eps * Rademacher``（逐元素 ±eps），
      其 L∞ **恰好等于** eps。δ 的实际 L∞ 是 ``eps * max|tanh|``，
      在 tanh 未饱和时**可能小于** eps —— 两者是"同预算"，不是"同实测幅度"。
    - ``delta_plus_xi`` 中的 δ 由**干净输入** x 计算，而非由 ``x + ξ`` 计算，
      与 fba.py:54 使用 ``data`` 而非 ``poison_data`` 的行为一致。
    - 所有模式最终 clamp 到 ``[pixel_min, pixel_max]``。
    - 缺少必需组件时抛 ``ValueError``，**不做静默降级**。
    """
    if mode not in MODES:
        raise ValueError(f"未知的扰动模式 '{mode}'；可选: {list(MODES)}")

    if mode == "clean":
        return images.clone()

    if mode in MODES_REQUIRING_XI:
        if xi_fn is None:
            raise ValueError(f"模式 '{mode}' 需要 xi_fn，但未提供")
        if labels is None:
            raise ValueError(
                f"模式 '{mode}' 需要真实标签 labels（无目标 PGD 对真实标签"
                f"做梯度上升），但未提供")

    if mode in MODES_REQUIRING_DELTA:
        if delta is None and generator is None and delta_fn is None:
            raise ValueError(
                f"模式 '{mode}' 需要 delta / delta_fn / generator 三者之一，"
                f"但都未提供")

    if mode == "random_noise" and eps is None:
        raise ValueError(
            "模式 'random_noise' 需要显式的 eps —— 负对照必须与 δ 共用"
            "同一个 L∞ 预算")

    # --- δ ---
    delta_tensor: Optional[torch.Tensor] = None
    if mode in MODES_REQUIRING_DELTA:
        if delta is not None:
            delta_tensor = delta
        elif delta_fn is not None:
            delta_tensor = delta_fn(images)
        else:
            delta_tensor = delta_from_generator(
                generator, images, eps=_DEFAULT_EPS if eps is None else eps)
        if delta_tensor.shape != images.shape:
            raise ValueError(
                f"delta 形状 {tuple(delta_tensor.shape)} 与输入 "
                f"{tuple(images.shape)} 不一致")
        delta_tensor = delta_tensor.to(images.device, images.dtype)

    # --- 合成 ---
    if mode == "delta_only":
        out = images + delta_tensor
    elif mode == "xi_only":
        out = xi_fn(images, labels)
    elif mode == "delta_plus_xi":
        # δ 由干净输入算（对应 fba.py:54 的 `trigger_gen(data)`）。
        out = xi_fn(images, labels) + delta_tensor
    elif mode == "random_noise":
        signs = torch.randint(0, 2, images.shape, generator=rng,
                              device=images.device, dtype=images.dtype) * 2 - 1
        out = images + float(eps) * signs
    else:  # pragma: no cover - 上面已穷举
        raise ValueError(f"未处理的模式 '{mode}'")

    return out.clamp(float(pixel_min), float(pixel_max))
