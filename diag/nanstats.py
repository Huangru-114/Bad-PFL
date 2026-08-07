"""nan-safe 统计工具。

为什么需要这个模块：本诊断中 nan 是**有意义的信号**（"该客户端没有这个类的样本"、
"该类原型无效"），绝不能用 0 填充。PyTorch 提供了 ``torch.nanmean`` 与
``torch.nanmedian``，但**没有** ``torch.nanstd``（已在 torch 2.13 上确认），
所以这里自行实现，并统一所有 nan 语义。

统一约定（所有函数遵守）：
- 输入中的 nan 一律被**跳过**，不参与计算。
- 若某个归约维度上的有效元素数不足（mean/median 需 >=1，unbiased std 需 >=2），
  结果为 **nan**，而不是 0，也不抛异常。
- 所有函数都不会把 nan "传染"给本可计算的结果。
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "nan_to_num_count",
    "nanmean",
    "nanstd",
    "nanmedian",
    "nanmad",
    "nan_cv",
]


def _nan(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(float("nan"), dtype=dtype, device=device)


def nan_to_num_count(x: torch.Tensor, dim: Optional[int] = None,
                     keepdim: bool = False) -> torch.Tensor:
    """沿 ``dim`` 统计**非 nan** 元素个数（返回浮点张量，便于后续除法）。"""
    mask = ~torch.isnan(x)
    if dim is None:
        return mask.sum().to(x.dtype)
    return mask.sum(dim=dim, keepdim=keepdim).to(x.dtype)


def nanmean(x: torch.Tensor, dim: Optional[int] = None,
            keepdim: bool = False) -> torch.Tensor:
    """nan-safe 均值。全 nan 的切片返回 nan（torch.nanmean 的原生行为）。"""
    if dim is None:
        return torch.nanmean(x)
    return torch.nanmean(x, dim=dim, keepdim=keepdim)


def nanstd(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False,
           unbiased: bool = True) -> torch.Tensor:
    """nan-safe 标准差（``torch.nanstd`` 不存在，故自行实现）。

    nan 行为：跳过 nan；有效元素数 < 2（unbiased）或 < 1（biased）时返回 nan。
    """
    if dim is None:
        flat = x.reshape(-1)
        return nanstd(flat, dim=0, keepdim=False, unbiased=unbiased)

    mask = ~torch.isnan(x)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    xz = torch.where(mask, x, zero)

    n = mask.sum(dim=dim, keepdim=True).to(x.dtype)
    # n == 0 时 mean 会是 0/0；先用安全分母算，最后统一盖成 nan。
    safe_n = torch.where(n > 0, n, torch.ones_like(n))
    mean = xz.sum(dim=dim, keepdim=True) / safe_n

    dev2 = torch.where(mask, (x - mean) ** 2, zero)
    denom = n - 1.0 if unbiased else n
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    var = dev2.sum(dim=dim, keepdim=True) / safe_denom
    var = torch.where(denom > 0, var, _nan(x.dtype, x.device))

    out = torch.sqrt(var.clamp_min(0))
    if not keepdim:
        out = out.squeeze(dim)
    return out


def nanmedian(x: torch.Tensor, dim: Optional[int] = None,
              keepdim: bool = False) -> torch.Tensor:
    """nan-safe 中位数。

    注意 ``torch.nanmedian`` 取的是**下中位数**（偶数个元素时取较小的那个），
    与 ``numpy.nanmedian`` 的插值行为不同。本诊断中只用它做稳健中心估计，
    这个差异无实质影响，但跨语言比对结果时要记得。
    全 nan 的切片返回 nan。
    """
    if dim is None:
        return torch.nanmedian(x)
    values, _ = torch.nanmedian(x, dim=dim, keepdim=keepdim)
    return values


def nanmad(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False,
           scale: float = 1.0) -> torch.Tensor:
    """nan-safe 中位绝对偏差 MAD = median(|x - median(x)|)。

    ``scale`` 默认 1.0（原始 MAD）。若要与正态分布的 sigma 可比，传 1.4826。
    本项目的 ``observable_score`` 使用原始 MAD，因为它只做跨客户端的相对比较。
    nan 行为：跳过 nan；全 nan 返回 nan。
    """
    if dim is None:
        flat = x.reshape(-1)
        return nanmad(flat, dim=0, keepdim=False, scale=scale)

    med = nanmedian(x, dim=dim, keepdim=True)
    dev = torch.abs(x - med)
    out = nanmedian(dev, dim=dim, keepdim=keepdim)
    return out * scale


def nan_cv(x: torch.Tensor, dim: Optional[int] = None,
           keepdim: bool = False, unbiased: bool = True) -> torch.Tensor:
    """nan-safe 变异系数 CV = std / mean。

    用均值归一化是为了避免"均值大的组方差自然大"这一伪影。
    nan 行为：均值为 0（或 |mean| < 1e-12）时返回 nan，而不是 inf。
    """
    mu = nanmean(x, dim=dim, keepdim=keepdim)
    sd = nanstd(x, dim=dim, keepdim=keepdim, unbiased=unbiased)
    denom = torch.abs(mu)
    out = sd / denom
    return torch.where(denom > 1e-12, out, _nan(out.dtype, out.device))
