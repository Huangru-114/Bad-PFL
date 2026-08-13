"""任务 3：``A_observable`` 在 α=0.05 崩塌的排查。

两条主线：

**A. 有效类数混杂（只需 meta.json，无需模型）**
    ``exp_c`` 的类原型来自 ``RunBundle.local_train_loader``，读的是 meta.json 中
    ``train_indices`` 对应的**原始标签** —— 投毒只发生在训练时的 batch 取数环节
    （``fba.py:46-58``），不改变落盘的分区标签。所以有效类数应当纯由 Dirichlet
    决定、与是否恶意无关。本模块直接检验这一点。

**B. 数值健康度（需要 checkpoint）**
    重算边距矩阵并导出 nan 占比、MAD≈0 占比、|z| 极值、**每客户端的有效 t 列数**。

已知的结构性缺陷（代码审查确立，见 ``diag/metrics.py::observable_score``）：
``A^(k) = max_t(...)``，而每个客户端的有效 t 列数不同 ——
客户端有 V 个有效类时 ``nanstd`` 需要 ≥2 个有效 c 才不返回 nan，V=1 时整客户端全 nan。
**在更多列上取 max 系统性地更大**（次序统计量偏置），因此 ``A^(k)``
在有效类数不同的客户端之间不可比。α 越小 V 越小，与断崖出现在 α=0.05 相符。

另确认：``z = where(mad > 1e-12, z, nan)`` 有除零保护，**不会产生 inf**；
失效形式是 nan 蔓延，不是数值溢出。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..analysis import pearson_spearman
from ..features import (class_prototypes, extract_penultimate, find_last_linear,
                        margin_matrix)
from ..metrics import observable_score
from ..nanstats import nanmad, nanmean, nanmedian, nanstd
from .common import (THRESHOLDS, AuditContext, Verdict, load_run_bundle,
                     load_run_meta, md_table, safe_auc, valid_class_count)

__all__ = ["run"]


# ---------------------------------------------------------------------------
# A. 有效类数混杂（meta-only）
# ---------------------------------------------------------------------------
def _valid_class_audit(ctx: AuditContext) -> pd.DataFrame:
    min_class_count = int(ctx.cfg.metrics.min_class_count)
    rows: List[Dict[str, Any]] = []
    for alpha, seed in ctx.pairs():
        run_path = ctx.run_dir("attack", alpha, seed)
        if run_path is None:
            continue
        meta = load_run_meta(run_path)
        for record in meta["clients"]:
            histogram = record.get("label_histogram")
            if histogram is None:
                continue
            rows.append({
                "alpha": alpha, "seed": seed,
                "client_id": int(record["client_id"]),
                "is_malicious": bool(record["is_malicious"]),
                "n_valid_classes": valid_class_count(histogram, min_class_count),
                "n_local_samples": int(record.get("n_local_samples", 0)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. 数值健康度（需要 checkpoint）
# ---------------------------------------------------------------------------
def _numeric_health(ctx: AuditContext, alpha: float, seed: int
                    ) -> Optional[Dict[str, Any]]:
    """重算某个 (α, seed) 的 attack run 的边距矩阵与中间量。"""
    run_path = ctx.run_dir("attack", alpha, seed)
    if run_path is None:
        return None

    bundle = load_run_bundle(ctx, run_path, cache_models=False)
    min_class_count = int(ctx.cfg.metrics.min_class_count)
    lambda_std = float(ctx.cfg.metrics.lambda_std)
    records = bundle.client_records()

    margins: List[torch.Tensor] = []
    per_client: List[Dict[str, Any]] = []
    for record in records:
        model = bundle.client_model(record["client_id"])
        loader = bundle.local_train_loader(record)
        out = extract_penultimate(model, loader, bundle.device, source="local")
        stats = class_prototypes(out.feats, out.labels, bundle.num_classes,
                                 min_class_count=min_class_count)
        head = find_last_linear(model)
        matrix = margin_matrix(stats["proto"], head.weight.detach().cpu(),
                               head.bias.detach().cpu())
        margins.append(matrix)

        off_diagonal = ~torch.eye(bundle.num_classes, dtype=torch.bool)
        per_client.append({
            "alpha": alpha, "seed": seed,
            "client_id": int(record["client_id"]),
            "is_malicious": bool(record["is_malicious"]),
            # 有效类数按**实际用于建原型的标签**统计，与 meta 的直方图互为校验
            "n_valid_classes": int((stats["count"] >= min_class_count).sum().item()),
            "M_nan_fraction": float(torch.isnan(matrix[off_diagonal])
                                    .float().mean().item()),
        })
        del model

    # --- 跨客户端的 median / MAD / z（复刻 observable_score 的内部） ---
    stacked = torch.stack([m.float() for m in margins], dim=0)      # [K,C,C]
    median = nanmedian(stacked, dim=0, keepdim=True)
    mad = nanmad(stacked, dim=0, keepdim=True)
    z = (stacked - median) / torch.where(mad > 1e-12, mad, torch.ones_like(mad))
    z = torch.where(mad > 1e-12, z, torch.tensor(float("nan")))

    per_target = -nanmean(z, dim=1) - lambda_std * nanstd(z, dim=1)   # [K,C]
    scores = observable_score(margins, lambda_std=lambda_std)

    off_diagonal_mask = ~torch.eye(stacked.shape[1], dtype=torch.bool)
    mad_off = mad[0][off_diagonal_mask]
    z_finite = z[torch.isfinite(z)]

    for index, entry in enumerate(per_client):
        row = per_target[index]
        entry["n_valid_t_columns"] = int(torch.isfinite(row).sum().item())
        entry["A_observable"] = float(scores[index])
        # 敏感性变体：用 mean_t 取代 max_t，规避次序统计量偏置。
        # 明确标注为诊断用，**不替换主定义**。
        finite = row[torch.isfinite(row)]
        entry["A_mean_t_variant"] = (float(finite.mean()) if finite.numel()
                                     else float("nan"))

    aggregate = {
        "alpha": alpha, "seed": seed, "n_clients": len(records),
        "M_nan_fraction": float(np.mean([e["M_nan_fraction"] for e in per_client])),
        "mad_zero_fraction": float((mad_off <= 1e-12).float().mean().item()),
        "valid_t_median": float(np.median([e["n_valid_t_columns"] for e in per_client])),
        "valid_t_min": int(np.min([e["n_valid_t_columns"] for e in per_client])),
        "valid_classes_median": float(np.median([e["n_valid_classes"] for e in per_client])),
        "z_abs_max": float(z_finite.abs().max().item()) if z_finite.numel() else float("nan"),
        "z_gt_100_count": int((z_finite.abs() > 100).sum().item()) if z_finite.numel() else 0,
        "z_inf_count": int(torch.isinf(z).sum().item()),
        "A_nan_count": int(torch.isnan(scores).sum().item()),
    }
    return {"per_client": pd.DataFrame(per_client), "aggregate": aggregate}


def _stratified_auc(frame: pd.DataFrame, score_column: str) -> float:
    """在**有效类数相同**的分层内分别算 AUC，再按层内客户端数加权平均。

    目的是把"有效类数"这个混杂固定住。层内良性/恶意任一为空则该层跳过。
    全部层都无法计算时返回 nan。
    """
    weighted, total = 0.0, 0.0
    for _, block in frame.groupby("n_valid_classes"):
        labels = block["is_malicious"].astype(bool).to_numpy()
        if labels.all() or not labels.any():
            continue
        value = safe_auc(block[score_column].to_numpy(dtype=float), labels)
        if np.isfinite(value):
            weighted += value * len(block)
            total += len(block)
    return weighted / total if total > 0 else float("nan")


def run(ctx: AuditContext) -> str:
    sections = ["## 任务 3：`A_observable` 在 α=0.05 的崩塌", ""]

    # ---------------- A. 有效类数混杂 ----------------
    valid_frame = _valid_class_audit(ctx)
    sections += ["### 3.1 有效类数混杂检验（仅用 meta.json，无需模型）", "",
                 "关键事实：`exp_c` 的类原型建立在 **原始（未投毒）标签** 之上 —— "
                 "投毒只发生在训练时的 batch 取数（`fba.py:46-58`），"
                 "不改变 `meta.json` 里落盘的分区。因此有效类数**理应**与是否恶意无关。",
                 ""]

    if valid_frame.empty:
        sections += ["⚠️ 没有可用的 attack meta.json，本节跳过。", ""]
        ctx.add(Verdict(item="有效类数混杂", conclusion="未能确定",
                        rule="需要 attack run 的 meta.json",
                        measured="未找到", action="确认 --ckpt-root"))
    else:
        ctx.save_evidence(valid_frame, "task3_valid_classes")
        rows = []
        for (alpha, seed), block in valid_frame.groupby(["alpha", "seed"]):
            from .common import group_compare
            stats = group_compare(block["n_valid_classes"],
                                  block["is_malicious"], "n_valid_classes")
            stats.update({"alpha": alpha, "seed": seed})
            stats["auc_using_valid_classes_only"] = safe_auc(
                block["n_valid_classes"].to_numpy(dtype=float),
                block["is_malicious"].astype(bool).to_numpy())
            rows.append(stats)
        confound_frame = pd.DataFrame(rows).sort_values(
            ["alpha", "seed"], ascending=[False, True])
        ctx.save_evidence(confound_frame, "task3_valid_class_confound")

        display = confound_frame[["alpha", "seed", "benign_mean", "malicious_mean",
                                  "benign_min", "benign_max", "malicious_min",
                                  "malicious_max", "separated",
                                  "auc_using_valid_classes_only", "mannwhitney_p"]]
        sections += [md_table(display), ""]

        worst = float(np.nanmax(np.abs(
            confound_frame["auc_using_valid_classes_only"] - 0.5)))
        any_separated = bool(confound_frame["separated"].any())
        if any_separated or worst > THRESHOLDS["confound_auc_margin"]:
            conclusion = "存在混杂"
            action = "A_observable 的判读必须做有效类数分层，见 3.3"
        else:
            conclusion = "无混杂（假设被证伪）"
            action = "有效类数不解释 A_observable 的翻转，需另找原因"
        ctx.add(Verdict(
            item="有效类数混杂（恶意 vs 良性）",
            conclusion=conclusion,
            rule=f"两组范围完全不重叠，或 |AUC(仅用有效类数) - 0.5| > "
                 f"{THRESHOLDS['confound_auc_margin']}",
            measured=f"是否有 α 出现完全不重叠：{'是' if any_separated else '否'}；"
                     f"|AUC-0.5| 最大 {worst:.3f}",
            action=action))

    # ---------------- B. 数值健康度 ----------------
    sections += ["### 3.2 数值健康度（需 checkpoint）", ""]
    aggregates: List[Dict[str, Any]] = []
    per_client_all: List[pd.DataFrame] = []
    for alpha, seed in ctx.pairs():
        result = _numeric_health(ctx, alpha, seed)
        if result is None:
            continue
        aggregates.append(result["aggregate"])
        per_client_all.append(result["per_client"])
        print(f"[task3] α={alpha} seed={seed} 数值健康度完成")

    if not aggregates:
        sections += ["⚠️ 没有可用的 attack checkpoint，本节跳过。", ""]
        ctx.add(Verdict(item="A_observable 数值健康度", conclusion="未能确定",
                        rule="需要 attack checkpoint", measured="未找到",
                        action="确认 --ckpt-root"))
        return "\n".join(sections)

    aggregate_frame = pd.DataFrame(aggregates).sort_values(
        ["alpha", "seed"], ascending=[False, True])
    per_client_frame = pd.concat(per_client_all, ignore_index=True)
    ctx.save_evidence(aggregate_frame, "task3_numeric_health")
    ctx.save_evidence(per_client_frame, "task3_per_client")

    sections += ["纵向对比（看从哪个 α 开始劣化）：", "",
                 md_table(aggregate_frame), "",
                 "`z_inf_count` 恒为 0 印证了 `observable_score` 的 1e-12 保护生效："
                 "失效形式是 **nan 蔓延**，不是数值溢出。", ""]

    # 逐 α 判定有效性
    invalid_alphas: List[float] = []
    for _, row in aggregate_frame.iterrows():
        bad = (row["valid_t_median"] < THRESHOLDS["min_valid_t_median"]
               or row["M_nan_fraction"] > THRESHOLDS["max_nan_fraction"])
        if bad:
            invalid_alphas.append(float(row["alpha"]))
    invalid_alphas = sorted(set(invalid_alphas), reverse=True)
    ctx.notes["invalid_alphas_A_observable"] = invalid_alphas

    ctx.add(Verdict(
        item="A_observable 数值有效性",
        conclusion=(f"α={invalid_alphas} 判为 invalid" if invalid_alphas
                    else "各 α 数值健康"),
        rule=f"有效 t 列数中位数 < {THRESHOLDS['min_valid_t_median']} "
             f"或 M 的 nan 占比 > {THRESHOLDS['max_nan_fraction']}",
        measured="；".join(
            f"α={r['alpha']}: 有效t中位数={r['valid_t_median']:.1f}, "
            f"nan占比={r['M_nan_fraction']:.3f}"
            for _, r in aggregate_frame.iterrows()),
        action=("invalid 的 α 不参与趋势判读，并据此确定 A^(k) 的适用下界"
                if invalid_alphas else "无需处理")))

    # ---------------- C. 相关性与分层 ----------------
    sections += ["### 3.3 A^(k) 与有效类数的关系 + 分层 AUC", ""]
    relation_rows = []
    for (alpha, seed), block in per_client_frame.groupby(["alpha", "seed"]):
        labels = block["is_malicious"].astype(bool).to_numpy()
        r, rho = pearson_spearman(block["n_valid_classes"], block["A_observable"])
        relation_rows.append({
            "alpha": alpha, "seed": seed,
            "pearson_r(A, 有效类数)": r,
            "spearman_rho(A, 有效类数)": rho,
            "AUC(A_observable)": safe_auc(
                block["A_observable"].to_numpy(dtype=float), labels),
            "AUC(仅有效类数)": safe_auc(
                block["n_valid_classes"].to_numpy(dtype=float), labels),
            "AUC(A, 有效类数分层)": _stratified_auc(block, "A_observable"),
            "AUC(mean_t 变体)": safe_auc(
                block["A_mean_t_variant"].to_numpy(dtype=float), labels),
            "有效t列数中位数": float(block["n_valid_t_columns"].median()),
        })
    relation_frame = pd.DataFrame(relation_rows).sort_values(
        ["alpha", "seed"], ascending=[False, True])
    ctx.save_evidence(relation_frame, "task3_relation")
    sections += [md_table(relation_frame), "",
                 "- `AUC(A, 有效类数分层)`：在有效类数相同的客户端内部分别算 AUC "
                 "再加权平均，把该混杂固定住。",
                 "- `AUC(mean_t 变体)`：把 `max_t` 换成 `mean_t` 以规避次序统计量偏置。"
                 "**这是诊断用的敏感性分析，不替换主定义。**", ""]

    max_rho = float(np.nanmax(np.abs(relation_frame["spearman_rho(A, 有效类数)"])))
    ctx.add(Verdict(
        item="A_observable 是否退化为有效类数的代理",
        conclusion=("是（强相关）" if max_rho > THRESHOLDS["confound_rho"]
                    else "否（相关性不足以支持该解释）"),
        rule=f"|Spearman(A, 有效类数)| > {THRESHOLDS['confound_rho']}",
        measured=f"各 α 中 |ρ| 最大 = {max_rho:.3f}",
        action=("以分层 AUC 为准，原始 AUC 不可直接判读"
                if max_rho > THRESHOLDS["confound_rho"] else "无需分层")))

    # ---------------- D. 适用下界 ----------------
    if invalid_alphas:
        healthy = aggregate_frame[~aggregate_frame["alpha"].isin(invalid_alphas)]
        if not healthy.empty:
            bound = float(healthy["valid_classes_median"].min())
            sections += [
                "### 3.4 `A^(k)` 的适用条件（必须写进文档）", "",
                f"- 数值健康的 α 中，有效类数中位数最低为 **{bound:.1f}**；"
                f"invalid 的 α 为 {invalid_alphas}。",
                f"- 因此建议的适用条件：**每客户端至少 ~{max(3, int(np.ceil(bound)))} "
                f"个有效类**（`count[c] >= {int(ctx.cfg.metrics.min_class_count)}`）。",
                "- 该下界是从**这一批数据**归纳的，换数据集或类别数需重新确认。", ""]

    return "\n".join(sections)
