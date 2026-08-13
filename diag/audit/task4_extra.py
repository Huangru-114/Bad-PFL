"""任务 4.1 / 4.3：补相关性，以及拆开实验 D 的归一化。

# 4.1 为什么必须补相关性

AUC 衡量的是**可分性**，相关系数衡量的是**代理有效性** —— 两者不是一回事。
``A^(k)`` 完全可能可分性差但相关性好（或反之）。图 C1 需要的是后者。

# 4.3 为什么必须拆开 J

    J_c^(k) = tr(Sigma_c^(k)) / ||mu_c^(k) - mu_global^(k)||^2

H2 预测**分子**（类内散度）在恶意客户端更大，但实测 AUC 反向。一个可能是
**分母的效应压过了分子**：后门训练把目标类原型推离全局中心，导致 J 反而变小。
若成立，J 测的是"原型偏离度"而非"类内混乱度"，两个物理量被归一化搅在了一起。

注意 ``exp_d`` 用的是 ``probe.ref ∪ probe.other``（类别均衡、所有客户端完全相同），
所有类都有效、不产生 nan —— 所以实验 D 的反向 AUC **不是数值失效**，
只能是分子/分母的真实效应。这让本节的拆解成为决定性检验。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from ..analysis import load_raw, pearson_spearman
from ..features import class_prototypes, extract_penultimate
from ..probe import make_eval_loader
from .common import (THRESHOLDS, AuditContext, Verdict, load_run_bundle,
                     md_table, safe_auc)

__all__ = ["run"]


# ---------------------------------------------------------------------------
# 4.1 相关性
# ---------------------------------------------------------------------------
def _correlations(ctx: AuditContext) -> Optional[pd.DataFrame]:
    try:
        raw = load_raw(str(ctx.raw_dir / "expC_*.csv"))
    except FileNotFoundError:
        return None
    if not {"E", "A_observable"}.issubset(raw.columns):
        return None

    rows: List[Dict[str, Any]] = []
    for (alpha, seed), block in raw.groupby(["alpha", "seed"]):
        malicious = block["is_malicious"].astype(bool)
        for scope, sub in (("全体", block), ("仅良性", block[~malicious]),
                           ("仅恶意", block[malicious])):
            r, rho = pearson_spearman(sub["A_observable"], sub["E"])
            rows.append({
                "alpha": alpha, "seed": seed, "范围": scope, "n": int(len(sub)),
                "pearson_r": r, "spearman_rho": rho,
                "E_mean": float(np.nanmean(sub["E"])) if len(sub) else float("nan"),
                "E_std": (float(np.nanstd(sub["E"], ddof=1)) if len(sub) > 1
                          else float("nan")),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.3 分子 / 分母拆解
# ---------------------------------------------------------------------------
def _components(ctx: AuditContext, alpha: float, seed: int) -> Optional[pd.DataFrame]:
    """在 exp_d 的同一份探针数据上，分别导出分子、分母与比值。"""
    run_path = ctx.run_dir("attack", alpha, seed)
    if run_path is None:
        return None

    bundle = load_run_bundle(ctx, run_path, cache_models=False)
    combined = ConcatDataset([bundle.probe.ref, bundle.probe.other])
    batch_size = (int(ctx.cfg.smoke.probe.batch_size) if bundle.smoke
                  else int(ctx.cfg.probe.batch_size))
    loader = make_eval_loader(combined, batch_size=batch_size)
    min_class_count = int(ctx.cfg.metrics.min_class_count)

    rows: List[Dict[str, Any]] = []
    for record in bundle.client_records():
        model = bundle.client_model(record["client_id"])
        out = extract_penultimate(model, loader, bundle.device, source="probe")
        stats = class_prototypes(out.feats, out.labels, bundle.num_classes,
                                 min_class_count=min_class_count)
        proto, scatter, count = stats["proto"], stats["scatter"], stats["count"]

        valid = ~torch.isnan(proto).any(dim=1)
        weights = count[valid].clamp_min(0.0)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        mu_global = (proto[valid] * weights.unsqueeze(1)).sum(0) / weights.sum()

        # 每个模型内部先做尺度归一化：不同客户端（FedBN 下 BN 参数各异）
        # 的特征尺度不可比，直接比 tr(Sigma) 或 ||mu - mu_g||^2 都没有意义。
        mean_scatter = float(np.nanmean(scatter.numpy()))
        dev2_all = np.full(bundle.num_classes, np.nan)
        for cls in range(bundle.num_classes):
            if bool(valid[cls]):
                dev2_all[cls] = float(((proto[cls] - mu_global) ** 2).sum())
        mean_dev2 = float(np.nanmean(dev2_all))

        for cls in range(bundle.num_classes):
            numerator = float(scatter[cls])
            denominator = dev2_all[cls]
            rows.append({
                "alpha": alpha, "seed": seed,
                "client_id": int(record["client_id"]),
                "is_malicious": bool(record["is_malicious"]),
                "class_id": cls,
                "is_target_class": cls == bundle.target_class,
                "numerator": (numerator / mean_scatter
                              if np.isfinite(numerator) and mean_scatter > 0
                              else np.nan),
                "denominator": (denominator / mean_dev2
                                if np.isfinite(denominator) and mean_dev2 > 0
                                else np.nan),
                "ratio_J": (numerator / denominator
                            if np.isfinite(numerator) and np.isfinite(denominator)
                            and denominator > 0 else np.nan),
            })
        del model
    return pd.DataFrame(rows)


def _signal_auc(frame: pd.DataFrame, column: str) -> float:
    """复刻 exp_d 的流程：v = log(x / 良性中位数) -> max_c v -> AUC。"""
    benign = frame[~frame["is_malicious"].astype(bool)]
    baseline = benign.groupby("class_id")[column].median()

    work = frame.copy()
    work["baseline"] = work["class_id"].map(baseline)
    with np.errstate(all="ignore"):
        work["v"] = np.where(
            (work[column] > 0) & (work["baseline"] > 0),
            np.log(work[column] / work["baseline"]), np.nan)

    per_client = (work.groupby(["client_id", "is_malicious"])["v"]
                  .max().reset_index())
    return safe_auc(per_client["v"].to_numpy(dtype=float),
                    per_client["is_malicious"].astype(bool).to_numpy())


def run(ctx: AuditContext) -> str:
    sections = ["## 任务 4：补充缺失的量", ""]

    # ---------------- 4.1 ----------------
    sections += ["### 4.1 实验 C 的相关性（AUC 之外必须看的东西）", ""]
    correlations = _correlations(ctx)
    if correlations is None:
        sections += ["⚠️ 找不到含 `E` / `A_observable` 的 expC CSV，跳过。", ""]
        ctx.add(Verdict(item="E_k 与 A^(k) 的相关性", conclusion="未能确定",
                        rule="需要 expC CSV 中的 E 与 A_observable 列",
                        measured="未找到", action="确认 --raw-dir"))
    else:
        ctx.save_evidence(correlations, "task4_correlations")
        sections += [md_table(correlations), "",
                     "⚠️ `仅恶意` 一行的 n 很小（正式配置下只有 10 个恶意客户端），"
                     "其相关系数的统计功效很低，不应单独据此下结论。", ""]

        benign = correlations[correlations["范围"] == "仅良性"]
        worst_benign = (float(np.nanmax(np.abs(benign["E_mean"])))
                        if len(benign) else float("nan"))
        overall = correlations[correlations["范围"] == "全体"]
        best_r = (float(np.nanmax(np.abs(overall["pearson_r"])))
                  if len(overall) else float("nan"))

        ctx.add(Verdict(
            item="良性客户端 E_k ≈ 0（E_k 定义的前提）",
            conclusion=("成立" if np.isfinite(worst_benign)
                        and worst_benign <= THRESHOLDS["benign_E_tolerance"]
                        else "不成立" if np.isfinite(worst_benign) else "未能确定"),
            rule=f"各 α 下 |mean(E_k | 良性)| <= {THRESHOLDS['benign_E_tolerance']}",
            measured=f"|均值| 最大 = {worst_benign:.4f}",
            action=("E_k 可作为 ground truth" if np.isfinite(worst_benign)
                    and worst_benign <= THRESHOLDS["benign_E_tolerance"]
                    else "E_k 的基线不为零，需检查 nat_rate 的定义与计算")))

        ctx.add(Verdict(
            item="A^(k) 作为 E_k 代理的有效性",
            conclusion=("有效（r > 0.7）" if best_r > 0.7 else
                        "部分有效（0.4 <= r <= 0.7）" if best_r >= 0.4 else
                        "无效（r < 0.4）" if np.isfinite(best_r) else "未能确定"),
            rule="全体客户端上的 |Pearson r|（取各 α 的最大值）",
            measured=f"|r| 最大 = {best_r:.3f}",
            action=("进入方法设计" if best_r > 0.7 else
                    "尝试改进可观测量（如加入类内散度项）" if best_r >= 0.4 else
                    "边距矩阵不是好代理，考虑在边缘侧做轻量 UAP 搜索直接估计 E_k")))

    # ---------------- 4.3 ----------------
    sections += ["### 4.3 实验 D 的归一化拆解（分子 / 分母 / 比值）", ""]
    frames: List[pd.DataFrame] = []
    for alpha, seed in ctx.pairs():
        component = _components(ctx, alpha, seed)
        if component is not None:
            frames.append(component)
            print(f"[task4] α={alpha} seed={seed} 分子/分母拆解完成")

    if not frames:
        sections += ["⚠️ 无可用 attack checkpoint，跳过。", ""]
        ctx.add(Verdict(item="实验 D 方向相反", conclusion="未能确定",
                        rule="需要 attack checkpoint", measured="未找到",
                        action="确认 --ckpt-root"))
        return "\n".join(sections)

    components = pd.concat(frames, ignore_index=True)
    ctx.save_evidence(components, "task4_d_components")

    rows = []
    for (alpha, seed), block in components.groupby(["alpha", "seed"]):
        rows.append({
            "alpha": alpha, "seed": seed,
            "AUC(分子 tr(Σ))": _signal_auc(block, "numerator"),
            "AUC(分母 ||μ-μg||²)": _signal_auc(block, "denominator"),
            "AUC(比值 J = 当前指标)": _signal_auc(block, "ratio_J"),
        })
    decomposition = pd.DataFrame(rows).sort_values(
        ["alpha", "seed"], ascending=[False, True])
    ctx.save_evidence(decomposition, "task4_d_decomposition")
    sections += [md_table(decomposition), "",
                 "三个信号都走同一条流程：先在模型内部做尺度归一化，"
                 "再 `v = log(x / 良性中位数)`，再 `max_c v`，最后算 AUC。", ""]

    margin = THRESHOLDS["auc_signal_margin"]
    numerator_auc = float(np.nanmean(decomposition["AUC(分子 tr(Σ))"]))
    denominator_auc = float(np.nanmean(decomposition["AUC(分母 ||μ-μg||²)"]))
    ratio_auc = float(np.nanmean(decomposition["AUC(比值 J = 当前指标)"]))

    if numerator_auc > 0.5 + margin and ratio_auc < 0.5:
        conclusion = "H2 成立，是归一化毁了它"
        action = "改用分子单独作为信号，并在报告中说明归一化的问题"
    elif numerator_auc < 0.5 and ratio_auc < 0.5:
        conclusion = "H2 证伪（ξ 未留下散度痕迹）"
        action = "放弃 H2 这条信号；不影响主路线"
    elif abs(denominator_auc - 0.5) > margin:
        conclusion = "分母（原型偏离度）本身是判别信号"
        action = "这是一个新发现，值得单独report；但需先确认不是别的混杂"
    else:
        conclusion = "未能确定"
        action = "三个信号都无明显判别力，需要更多 seed 或更强的指标"

    ctx.add(Verdict(
        item="实验 D 方向相反",
        conclusion=conclusion,
        rule=f"分子 AUC > 0.5+{margin} 且比值 AUC < 0.5 -> 归一化 artifact；"
             f"分子与比值均 < 0.5 -> H2 证伪；"
             f"|分母 AUC - 0.5| > {margin} -> 分母本身是信号",
        measured=f"分子 {numerator_auc:.3f}，分母 {denominator_auc:.3f}，"
                 f"比值 {ratio_auc:.3f}（各 α 均值）",
        action=action))

    return "\n".join(sections)
