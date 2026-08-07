"""特征提取与决策几何量。

所有跨客户端比较在本诊断中都是**标量指标**的比较（overlap 率、准确率、边距、
散度），不涉及直接比较不同模型的特征向量 —— 后者需要先做表征对齐，本项目
不实现，也不要在下游代码里引入。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, NamedTuple, Optional

import torch
import torch.nn as nn

from .nanstats import nanmean

__all__ = ["Penultimate", "find_last_linear", "extract_penultimate",
           "class_prototypes", "fisher_dispersion", "margin_matrix",
           "knn_overlap"]

FeatureSource = Literal["probe", "local"]


class Penultimate(NamedTuple):
    """``extract_penultimate`` 的返回值，可按 4-tuple 解包。"""

    feats: torch.Tensor      # [N, d] 倒数第二层特征
    labels: torch.Tensor     # [N]    真实标签
    logits: torch.Tensor     # [N, C] 分类头输出（顺带取回，避免二次前向）
    meta: Dict[str, Any]     # 记录 source / n_samples / feat_dim 等


def find_last_linear(model: nn.Module) -> nn.Linear:
    """返回模型中**最后一个** ``nn.Linear``，即分类头。

    对 ``resnet.ResNet`` 而言就是 ``model.linear``（resnet.py:87）。
    用遍历而非硬编码属性名，是为了与架构解耦（mobilenet/densenet 同样适用）。
    """
    last: Optional[nn.Linear] = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise ValueError("模型中找不到 nn.Linear，无法定位分类头")
    return last


def extract_penultimate(
    model: nn.Module,
    loader,
    device,
    *,
    source: FeatureSource,
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Penultimate:
    """用 forward pre-hook 提取倒数第二层特征，**不修改模型定义**。

    Parameters
    ----------
    source:
        必须显式声明特征来自共享探针集(``"probe"``)还是客户端本地数据
        (``"local"``)。这个参数只用于记录与校验 —— 目的是让"用错数据集"
        这类错误在代码层面显形，而不是等到结果出来才发现。
        除 ``observable_score`` 的本地类原型外，一律应为 ``"probe"``。
    transform:
        可选的 ``(images, labels) -> images`` 扰动函数，在**梯度启用**的区域
        内执行（``xi_fn`` 内部的 PGD 需要反传），随后才在 ``no_grad`` 下前向。

    Notes
    -----
    - 内部强制 ``model.eval()``，否则 BatchNorm 会用 batch 统计量并更新
      running stats，污染所有下游指标。函数返回前恢复模型原本的 train/eval 状态。
    - 前向在 ``torch.no_grad()`` 下执行；``transform`` **不在** no_grad 内。
    - 本函数不产生 nan；nan 只可能来自下游的类原型/几何量计算。
    """
    head = find_last_linear(model)
    captured: list[torch.Tensor] = []

    def pre_hook(_module, args):
        captured.append(args[0].detach())

    was_training = model.training
    model.eval()
    handle = head.register_forward_pre_hook(pre_hook)

    feats_all, labels_all, logits_all = [], [], []
    try:
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)

            if transform is not None:
                # 梯度启用区域：xi_fn 内部的 PGD 需要对输入求导。
                images = transform(images, labels)

            captured.clear()
            with torch.no_grad():
                logits = model(images)

            if not captured:
                raise RuntimeError(
                    "forward pre-hook 没有捕获到分类头输入；"
                    "模型的 forward 可能没有经过最后一个 nn.Linear")

            feats_all.append(captured[-1].flatten(1).float().cpu())
            logits_all.append(logits.detach().float().cpu())
            labels_all.append(labels.detach().cpu())
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not feats_all:
        raise ValueError("loader 为空，无法提取特征")

    feats = torch.cat(feats_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    logits = torch.cat(logits_all, dim=0)
    meta = {"source": source, "n_samples": int(feats.shape[0]),
            "feat_dim": int(feats.shape[1]),
            "transform": transform is not None}
    return Penultimate(feats=feats, labels=labels, logits=logits, meta=meta)


def class_prototypes(feats: torch.Tensor, labels: torch.Tensor,
                     num_classes: int,
                     min_class_count: int = 5) -> Dict[str, torch.Tensor]:
    """类原型、类计数、类内散度。

    Returns
    -------
    dict with keys:
        ``proto``   [C, d] 类均值；无效类填 nan
        ``count``   [C]    该类的**实际**样本数（始终是真实计数，不填 nan）
        ``scatter`` [C]    类内散度 tr(Sigma_c)；无效类填 nan

    Notes
    -----
    - ``tr(Sigma_c)`` 用 Bessel 修正（除以 ``count-1``）。``count < 2`` 时散度
      无定义，填 nan。
    - ``count[c] < min_class_count`` 的类，``proto`` 与 ``scatter`` 均填 nan。
      **下游所有统计必须用 nan-safe 版本**（见 ``diag.nanstats``），
      禁止用 0 填充 —— "这个客户端没有这个类"是信号，不是缺失值。
    """
    feats = feats.float()
    dim = feats.shape[1]
    proto = torch.full((num_classes, dim), float("nan"))
    scatter = torch.full((num_classes,), float("nan"))
    count = torch.zeros(num_classes)

    for cls in range(num_classes):
        mask = labels == cls
        n = int(mask.sum().item())
        count[cls] = float(n)
        if n < max(1, int(min_class_count)):
            continue
        block = feats[mask]
        mu = block.mean(dim=0)
        proto[cls] = mu
        if n >= 2:
            scatter[cls] = ((block - mu) ** 2).sum() / (n - 1)

    return {"proto": proto, "count": count, "scatter": scatter}


def fisher_dispersion(proto: torch.Tensor, scatter: torch.Tensor,
                      count: torch.Tensor) -> torch.Tensor:
    """Fisher 型归一化类内散度 ``J_c = tr(Sigma_c) / ||mu_c - mu_global||^2``。

    ``mu_global`` 是该模型上所有**有效**类原型按类计数加权的均值。

    归一化不是可选的：不同客户端的特征尺度差异很大（FedBN 下各自的 BN 参数
    不同），直接比较 ``tr(Sigma_c)`` 没有意义。

    nan 行为：``proto[c]`` 或 ``scatter[c]`` 为 nan 的类返回 nan；
    分母 ``||mu_c - mu_global||^2`` 退化为 0 时返回 nan（而不是 inf）。
    """
    num_classes = proto.shape[0]
    valid = ~torch.isnan(proto).any(dim=1)
    out = torch.full((num_classes,), float("nan"))
    if int(valid.sum().item()) == 0:
        return out

    weights = count[valid].clamp_min(0.0)
    if float(weights.sum().item()) <= 0:
        weights = torch.ones_like(weights)
    mu_global = (proto[valid] * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

    for cls in range(num_classes):
        if not bool(valid[cls]) or torch.isnan(scatter[cls]):
            continue
        denom = float(((proto[cls] - mu_global) ** 2).sum().item())
        if denom <= 1e-12:
            continue
        out[cls] = scatter[cls] / denom
    return out


def margin_matrix(proto: torch.Tensor, W: torch.Tensor,
                  b: torch.Tensor) -> torch.Tensor:
    """类间边距矩阵。

    ``M[c, t] = (<W[c] - W[t], proto[c]> + b[c] - b[t]) / ||W[c] - W[t]||``

    直观含义：类 ``c`` 的原型到 ``c`` 与 ``t`` 之间决策边界的**带符号距离**。
    ``M[c, t]`` 越小，说明类 ``c`` 的样本越容易被推向类 ``t``。

    nan 行为：
    - 对角线恒为 nan；
    - ``proto[c]`` 含 nan 的整行填 nan；
    - ``||W[c] - W[t]|| == 0``（两个类的权重向量完全相同）时填 nan。
    """
    num_classes = proto.shape[0]
    W = W.float()
    b = b.float()
    proto = proto.float()
    out = torch.full((num_classes, num_classes), float("nan"))

    for c in range(num_classes):
        if torch.isnan(proto[c]).any():
            continue
        for t in range(num_classes):
            if c == t:
                continue
            diff = W[c] - W[t]
            norm = float(diff.norm().item())
            if norm <= 1e-12:
                continue
            out[c, t] = (torch.dot(diff, proto[c]) + b[c] - b[t]) / norm
    return out


def knn_overlap(feats_query: torch.Tensor, feats_ref: torch.Tensor,
                feats_other: torch.Tensor, k: int = 20,
                chunk_size: int = 256) -> float:
    """量化"query 分布落在 ref 分布内部的程度"。

    对 ``feats_query`` 中每个点，在 ``feats_ref ∪ feats_other`` 中找 k 近邻
    （欧氏距离），返回近邻中属于 ``feats_ref`` 的平均比例。

    这是 t-SNE 目测的定量替代品。**所有结论必须来自本指标，
    t-SNE 只能用于展示。**

    ⚠️ 三组特征必须来自**同一模型的同一次前向配置**（同样的 eval 模式、
    同样的设备、同样的预处理）。混用不同模型提取的特征会让结果毫无意义，
    本函数无法检测这种误用 —— 调用方负责。

    nan 行为：``feats_ref`` 为空时返回 ``nan``（不是 0）。
    ``feats_query`` 为空时同样返回 ``nan``。
    若参照库总数小于 ``k``，则退化为使用全部参照点（等价于 ``k = 库大小``）。
    """
    if feats_ref is None or feats_ref.numel() == 0 or feats_ref.shape[0] == 0:
        return float("nan")
    if feats_query is None or feats_query.shape[0] == 0:
        return float("nan")

    ref = feats_ref.float()
    if feats_other is None or feats_other.numel() == 0:
        other = ref.new_zeros((0, ref.shape[1]))
    else:
        other = feats_other.float()
    bank = torch.cat([ref, other], dim=0)
    is_ref = torch.zeros(bank.shape[0], dtype=torch.bool)
    is_ref[: ref.shape[0]] = True

    effective_k = int(min(max(1, int(k)), bank.shape[0]))
    query = feats_query.float()

    totals = []
    for start in range(0, query.shape[0], int(chunk_size)):
        block = query[start:start + int(chunk_size)]
        dist = torch.cdist(block, bank)
        neighbor_idx = dist.topk(effective_k, dim=1, largest=False).indices
        totals.append(is_ref[neighbor_idx].float().mean(dim=1))

    per_point = torch.cat(totals, dim=0)
    return float(nanmean(per_point).item())
