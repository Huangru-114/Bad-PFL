"""四个诊断实验的指标计算。

本模块是**纯测量层**：输入已加载的模型 / 已构造的 loader，输出标量或
DataFrame 行。不训练、不画图、不落盘。

# 数据来源规则（重要）

| 测量                              | 数据来源            | 理由 |
|-----------------------------------|---------------------|------|
| 实验 A ``naturalness``            | **共享探针集**      | 受控比较：固定数据，只变模型 |
| 实验 B ``recovery_score``         | **共享探针集**      | 同上 |
| 实验 D ``dispersion_scores``      | **共享探针集**      | 同上 |
| 实验 C ``excess_response`` (E_k)  | **共享探针集**      | 这是 ground truth，要求精确 |
| 实验 C ``observable_score`` (A^k) | ⚠️ **客户端本地训练数据** | 忠实模拟防御方视角 |

最后一行是唯一的例外，也是刻意为之：``observable_score`` 模拟的是真实部署中
防御方能拿到的东西 —— 客户端在自己的本地数据上算原型然后上传。若用探针集算，
等于假设防御方持有一份干净的全局数据集，那是作弊。

其直接后果是：**本地数据少的客户端，其原型确实是噪声大的。** 这不是要"修掉"
的工程问题，而是防御方法本身必须面对的真实困难，应当被如实测量和报告。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .features import (class_prototypes, extract_penultimate, fisher_dispersion,
                       knn_overlap, margin_matrix)
from .nanstats import nanmean, nanmad, nanmedian, nanstd
from .perturb import apply_perturbation

__all__ = ["dispersion_scores", "naturalness", "recovery_score",
           "excess_response", "observable_score", "baseline_signals",
           "NATURALNESS_GROUPS"]

NATURALNESS_GROUPS = ("real", "delta_only", "delta_plus_xi", "random_noise")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _make_transform(mode: str, delta_fn, xi_fn, eps, rng_seed,
                    pixel_min: float, pixel_max: float):
    """把一个扰动模式包装成 ``extract_penultimate`` 需要的 transform 闭包。"""
    if mode == "clean":
        return None

    counter = {"n": 0}

    def transform(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        rng = None
        if rng_seed is not None:
            rng = torch.Generator(device=images.device)
            rng.manual_seed(int(rng_seed) + counter["n"])
            counter["n"] += 1
        return apply_perturbation(
            images, mode, xi_fn=xi_fn, delta_fn=delta_fn, eps=eps,
            labels=labels, rng=rng, pixel_min=pixel_min, pixel_max=pixel_max)

    return transform


def _fraction_predicted(logits: torch.Tensor, target_class: int) -> float:
    """预测为 ``target_class`` 的样本比例。空输入返回 nan。"""
    if logits.numel() == 0:
        return float("nan")
    return float((logits.argmax(dim=1) == int(target_class)).float().mean().item())


# ---------------------------------------------------------------------------
# 实验 D：源类表征混乱
# ---------------------------------------------------------------------------
def dispersion_scores(model: nn.Module, loader, benign_baseline: Optional[torch.Tensor],
                      *, device, num_classes: int, min_class_count: int = 5,
                      source: str = "probe") -> pd.DataFrame:
    """每个类的 Fisher 型归一化类内散度及其相对基线的对数比。

    ``v_c = log(J_c / benign_baseline[c])``，其中 ``benign_baseline`` 是良性
    客户端 ``J_c`` 的**中位数**（由调用方跨客户端算好后传入）。

    nan 行为：``J_c`` 无效（类样本不足 / 分母退化）时 ``J`` 与 ``v`` 均为 nan；
    ``benign_baseline`` 为 None 或该类基线 <= 0 时 ``v`` 为 nan。
    ``benign_baseline`` 为 None 是**第一遍**调用的正常用法 —— 先收集所有客户端
    的 ``J``，再算基线，再回填 ``v``。
    """
    out = extract_penultimate(model, loader, device, source=source)
    stats = class_prototypes(out.feats, out.labels, num_classes,
                             min_class_count=min_class_count)
    dispersion = fisher_dispersion(stats["proto"], stats["scatter"], stats["count"])

    rows = []
    for cls in range(num_classes):
        j_value = float(dispersion[cls].item())
        v_value = float("nan")
        if benign_baseline is not None and np.isfinite(j_value):
            base = float(benign_baseline[cls])
            if np.isfinite(base) and base > 0 and j_value > 0:
                v_value = float(np.log(j_value / base))
        rows.append({"class_id": cls,
                     "n_class_samples": int(stats["count"][cls].item()),
                     "J": j_value, "v": v_value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 实验 A：δ 的跨客户端自然度
# ---------------------------------------------------------------------------
def naturalness(model: nn.Module, probe_loaders: Dict[str, Any],
                delta_fn: Optional[Callable], xi_fn: Optional[Callable],
                target_class: int, *, device, k: int = 20,
                eps: float = 4.0 / 255.0, noise_seed: Optional[int] = 888,
                pixel_min: float = 0.0, pixel_max: float = 1.0,
                groups: Sequence[str] = NATURALNESS_GROUPS) -> Dict[str, Dict[str, float]]:
    """在**干净模型**上测量三类输入的"目标类归属度"。

    | 组             | 输入                        | 含义 |
    |----------------|-----------------------------|------|
    | ``real``       | 真实目标类图片(probe.target)| 正对照：真正的因果特征 |
    | ``delta_only`` | 非目标类图片 + δ            | 待测 |
    | ``delta_plus_xi`` | 非目标类图片 + δ + ξ     | 待测（与攻击时的完整触发器一致） |
    | ``random_noise``| 非目标类图片 + 同预算随机噪声| 负对照：无语义 |

    每组返回::

        overlap  = knn_overlap(该组特征, probe.ref 特征, probe.other 特征)
        nat_rate = 该组输入被该模型判为 target_class 的比例

    ⚠️ 必须传入**干净 run** 的模型。用攻击 run 的模型测会混入后门本身的效应，
    逻辑上是循环论证。本函数无法检测这种误用，调用方负责。

    ⚠️ ``ref`` / ``other`` 与各查询组的特征都由**同一个 model、同一次 eval 配置**
    提取，满足 ``knn_overlap`` 的前提。

    nan 行为：某组的 loader 为空或 ref 为空时，该组的 overlap 为 nan。
    """
    # 参照库：ref（真实目标类）与 other（其余类），都不加扰动。
    ref_out = extract_penultimate(model, probe_loaders["ref"], device, source="probe")
    other_out = extract_penultimate(model, probe_loaders["other"], device, source="probe")

    group_spec = {
        "real": ("target", "clean"),
        "delta_only": ("query", "delta_only"),
        "delta_plus_xi": ("query", "delta_plus_xi"),
        "random_noise": ("query", "random_noise"),
        "xi_only": ("query", "xi_only"),
    }

    results: Dict[str, Dict[str, float]] = {}
    for group in groups:
        if group not in group_spec:
            raise ValueError(f"未知的组 '{group}'；可选: {list(group_spec)}")
        loader_key, mode = group_spec[group]

        if mode in ("delta_only", "delta_plus_xi") and delta_fn is None:
            raise ValueError(f"组 '{group}' 需要 delta_fn，但未提供")
        if mode in ("xi_only", "delta_plus_xi") and xi_fn is None:
            raise ValueError(f"组 '{group}' 需要 xi_fn，但未提供")

        transform = _make_transform(mode, delta_fn, xi_fn, eps, noise_seed,
                                    pixel_min, pixel_max)
        out = extract_penultimate(model, probe_loaders[loader_key], device,
                                  source="probe", transform=transform)
        results[group] = {
            "overlap": knn_overlap(out.feats, ref_out.feats, other_out.feats, k=k),
            "nat_rate": _fraction_predicted(out.logits, target_class),
            "n_samples": int(out.feats.shape[0]),
        }
    return results


# ---------------------------------------------------------------------------
# 实验 B：Table 24 的跨客户端版本
# ---------------------------------------------------------------------------
def recovery_score(model: nn.Module, target_loader, delta_fn: Optional[Callable],
                   xi_fn: Callable, *, device, target_class: int,
                   eps: float = 4.0 / 255.0, pixel_min: float = 0.0,
                   pixel_max: float = 1.0) -> Dict[str, float]:
    """Table 24 的单模型版本：δ 为该模型恢复了多少目标类识别能力。

    只用**目标类**样本（``probe.target``）::

        acc_clean = 干净目标类样本的准确率
        acc_xi    = 加 ξ 后的目标类准确率        （原文 ≈ 6.70%）
        acc_dxi   = 加 δ+ξ 后的目标类准确率      （原文 ≈ 39.10%）
        recovery  = acc_dxi - acc_xi

    这里的"准确率"即被判为 ``target_class`` 的比例（因为样本全部属于该类）。

    返回值含 ``n_target_samples``，供下游作为**协变量**记录。
    注意：改用共享探针集后，所有客户端的 ``n_target_samples`` 都相同且充足，
    该字段的价值在于存档与校验；反映客户端差异的是
    ``n_target_samples_local``（由 meta.json 提供，见 exp_b.py）。
    """
    if delta_fn is None:
        raise ValueError("recovery_score 需要 delta_fn（无 δ 就无从谈恢复）")

    def _acc(mode: str) -> tuple:
        transform = _make_transform(mode, delta_fn, xi_fn, eps, None,
                                    pixel_min, pixel_max)
        out = extract_penultimate(model, target_loader, device,
                                  source="probe", transform=transform)
        return _fraction_predicted(out.logits, target_class), int(out.feats.shape[0])

    acc_clean, n_samples = _acc("clean")
    acc_xi, _ = _acc("xi_only")
    acc_dxi, _ = _acc("delta_plus_xi")

    return {"acc_clean": acc_clean, "acc_xi": acc_xi, "acc_dxi": acc_dxi,
            "recovery": acc_dxi - acc_xi, "n_target_samples": n_samples}


# ---------------------------------------------------------------------------
# 实验 C：超额响应（ground truth）
# ---------------------------------------------------------------------------
def excess_response(clean_model: nn.Module, attacked_model: nn.Module, loader,
                    delta_fn: Optional[Callable], xi_fn_clean: Callable,
                    xi_fn_attacked: Callable, target_class: int, *, device,
                    eps: float = 4.0 / 255.0, pixel_min: float = 0.0,
                    pixel_max: float = 1.0) -> Dict[str, float]:
    """超出自然语义所能解释的那部分响应，即"真正的后门强度"。

    ::

        nat_rate = 在【干净】本地模型上，非目标类样本加 δ+ξ 被判为目标类的比例
        asr      = 在【攻击 run】本地模型上，同样输入被判为目标类的比例
        E        = asr - nat_rate

    预期：良性客户端 ``E ≈ 0``；恶意客户端 ``E`` 显著为正。

    ⚠️ 这是 **ground truth，防御方看不到** —— 它同时用到了干净模型和 δ。
    对照的可观测量是 ``observable_score``，那个函数的签名里刻意不包含这两者。

    ⚠️ ξ 由**被评估的那个模型**生成（保持 fba.py:53 的原有语义），
    因此 ``xi_fn_clean`` 与 ``xi_fn_attacked`` 必须分别绑定对应的模型。
    调用方若想隔离 δ 的影响，可以把两者都绑到同一个参考模型上 ——
    这是一个刻意留给调用方的实验设计开关，务必在结果中记录实际用法。

    ⚠️ 本函数使用 ``probe.query``（**全部是非目标类样本**），因此
    与原实现的 ASR 口径不同：原实现把目标类原样本也算作攻击成功
    （fba.py:52 把全部标签改成 target_label），系统性抬高 ASR。这里不含该偏差。
    """
    if delta_fn is None:
        raise ValueError("excess_response 需要 delta_fn")

    def _rate(model: nn.Module, xi_fn: Callable) -> float:
        transform = _make_transform("delta_plus_xi", delta_fn, xi_fn, eps, None,
                                    pixel_min, pixel_max)
        out = extract_penultimate(model, loader, device, source="probe",
                                  transform=transform)
        return _fraction_predicted(out.logits, target_class)

    nat_rate = _rate(clean_model, xi_fn_clean)
    asr = _rate(attacked_model, xi_fn_attacked)
    return {"nat_rate": nat_rate, "asr": asr, "E": asr - nat_rate}


# ---------------------------------------------------------------------------
# 实验 C：可观测量（防御方视角）
# ---------------------------------------------------------------------------
def observable_score(margin_matrices: List[torch.Tensor],
                     lambda_std: float = 1.0) -> torch.Tensor:
    """防御方**仅凭客户端上传的信息**能算出的后门强度代理量。

    ::

        z^(k)[c,t] = (M^(k)[c,t] - median_j M^(j)[c,t]) / MAD_j M^(j)[c,t]
        A^(k)      = max_t [ -nanmean_{c!=t} z^(k)[c,t]
                             - lambda_std * nanstd_{c!=t} z^(k)[c,t] ]

    直观含义：若某个客户端把**所有**类到类 ``t`` 的边距都压低了（z 普遍为负）
    且压得很**均匀**（std 小），那它很可能植入了以 ``t`` 为目标的后门。

    ⚠️ **信息隔离约束（方法有效性的前提）**：本函数的签名中**不出现**
    ``delta``、``clean_model``、``target_class`` 或任何真实标签之外的信息，
    输入只有各客户端的边距矩阵 —— 而边距矩阵仅由**类原型 + 分类头**构成，
    两者都是客户端本来就要上传的东西。这条约束是代码层面强制的：
    如果将来有人需要往这个函数里传 δ 或干净模型，说明设计出了问题，
    应当停下讨论而不是加参数。

    Parameters
    ----------
    margin_matrices:
        长度为 K 的列表，每个元素是 ``[C, C]`` 的边距矩阵（对角线为 nan）。

    Returns
    -------
    ``[K]`` 张量。某客户端全部为 nan 时其得分为 nan。

    nan 行为：MAD 退化为 0 的 ``(c, t)`` 位置其 z 记为 nan（避免除零产生 inf）；
    所有归约均使用 nan-safe 统计。
    """
    if len(margin_matrices) == 0:
        return torch.empty(0)

    stacked = torch.stack([m.float() for m in margin_matrices], dim=0)  # [K,C,C]
    num_clients, num_classes, _ = stacked.shape

    median = nanmedian(stacked, dim=0, keepdim=True)          # [1,C,C]
    mad = nanmad(stacked, dim=0, keepdim=True)                # [1,C,C]

    nan_value = torch.tensor(float("nan"))
    z = (stacked - median) / torch.where(mad > 1e-12, mad, torch.ones_like(mad))
    z = torch.where(mad > 1e-12, z, nan_value)

    # 对每个候选目标类 t，沿 c 归约（对角线本就是 nan，c != t 自动满足）。
    mean_over_c = nanmean(z, dim=1)                           # [K,C]  索引为 t
    std_over_c = nanstd(z, dim=1)                             # [K,C]
    per_target = -mean_over_c - float(lambda_std) * std_over_c

    scores = torch.full((num_clients,), float("nan"))
    for k in range(num_clients):
        row = per_target[k]
        valid = row[~torch.isnan(row)]
        if valid.numel() > 0:
            scores[k] = valid.max()
    return scores


def baseline_signals(client_updates: List[torch.Tensor]) -> pd.DataFrame:
    """两个既有防御思路的对照信号，用于实验 C 的三线对比图（图 C2）。

    ``l2_to_median``
        到**逐维中位数更新**的 L2 距离。Krum / FLAME 一类基于距离的
        鲁棒聚合所依赖的信号。

    ``sign_consistency``
        与多数派符号的一致程度，对应 Invariant Aggregator
        (Wang et al., AISTATS'24) 的思路。

        原始表述"``|mean_i sign(g_i)|`` 逐维求和"是一个**全局**量，
        无法区分客户端。这里采用其自然的**逐客户端**推广::

            consensus_d = mean_i sign(u_i[d])            # [-1, 1]
            score_k     = mean_d [ sign(u_k[d]) * consensus_d ]

        取值 ∈ [-1, 1]，越大表示与多数派方向越一致。
        ⚠️ 这是一处**解释性选择**，不是文献中的原文定义，
        已在 README 的"已知限制"中标注，正式使用前需要人工确认。
        全局量本身也一并返回在 ``consensus_strength`` 列（所有行相同）。

    Parameters
    ----------
    client_updates:
        长度为 K 的列表，每个元素是**已展平**的更新向量 ``[D]``
        （通常是 ``local_params - global_params``）。

    nan 行为：输入为空返回空 DataFrame；某客户端向量含 nan 时其得分为 nan。
    """
    if len(client_updates) == 0:
        return pd.DataFrame(columns=["client_id", "l2_to_median",
                                     "sign_consistency", "consensus_strength"])

    stacked = torch.stack([u.float().flatten() for u in client_updates], dim=0)
    median = stacked.median(dim=0).values
    signs = torch.sign(stacked)
    consensus = signs.mean(dim=0)
    consensus_strength = float(consensus.abs().sum().item())

    rows = []
    for k in range(stacked.shape[0]):
        rows.append({
            "client_id": k,
            "l2_to_median": float((stacked[k] - median).norm().item()),
            "sign_consistency": float((signs[k] * consensus).mean().item()),
            "consensus_strength": consensus_strength,
        })
    return pd.DataFrame(rows)
