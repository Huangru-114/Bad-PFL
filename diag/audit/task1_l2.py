"""任务 1：``l2_to_median`` 完美分离（AUC=1.000）是真的吗？

完美分离通常不是因为信号强，而是因为存在**与后门无关的系统性差异**，让恶意
客户端在任何距离度量下都是离群点。本模块逐条排查混杂源。

# 已由代码审查确立的一条候选机制：梯度泄漏

``client.py:34`` 的 ``optimizer.zero_grad()`` 发生在 ``client.py:36``
``fetch_data()`` **之前**；而 ``PoisonClient.fetch_data`` (client.py:124-125)
→ ``our_poison_func`` (fba.py:46-58) → ``pgd_attack`` (fba.py:6-22) 内部的
``loss.backward()`` (fba.py:16) 会把 ∇CE 累加进**模型参数**。这些梯度不会被清除，
直接进入 ``client.py:39`` 的 ``scaler.step(optimizer)``。

**恶意客户端每个本地步都多吃一份全量梯度，良性客户端没有。**

但量级取决于 AMP：``scaler.scale(loss).backward()`` 让任务梯度被放大 S 倍
（典型 65536），``scaler.step`` 再整体除以 S。PGD 那份梯度是**未放大**的，
于是被除成 ∇_PGD/S —— GPU 上几乎归零；CPU 上 ``GradScaler`` 是 disabled（S=1），
泄漏是全强度的。

**所以不能断言这就是原因**，必须靠 ``leak_experiment.py`` 的受控 A/B 实测。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from ..hooks import flatten_state_dict
from .common import (THRESHOLDS, AuditContext, Verdict, group_compare,
                     load_run_bundle, load_run_meta, md_kv, md_table, safe_auc)

__all__ = ["run"]

STATIC_FINDINGS = [
    ("本地步数", "相同",
     "`fl_process.py:29-30` 对所有被选中的客户端一律 `for _ in range(local_steps)`，"
     "不区分类型；`diag/run_fl.py` 也只传一个 `local_steps`。"),
    ("学习率 / 优化器", "相同",
     "`diag/run_fl.py` 用同一个 `partial(torch.optim.SGD, lr=learning_rate)` "
     "构造全部客户端，`BasicClient.__init__`(client.py:13) 与 "
     "`PoisonClient`(client.py:118-125) 共用该路径。"),
    ("梯度/模型放缩、norm boosting、model replacement", "不存在",
     "`fba.py` 全文无 scale factor / norm clipping / 更新替换操作；"
     "`server.agg_and_update`(server.py:29-34) 恒调 `agg_avg`，"
     "`--agg_rule` 被解析(main.py:106)但从不读取。"),
    ("生成器训练是否反传进本地模型", "会，但被清除",
     "`trigger_gen_trainer`(fba.py:31-43) 的 `loss.backward()`(fba.py:41) 确实经过 "
     "`client.local_model`，但它在 `receive_model` 阶段执行，"
     "随后第一次 `local_update` 的 `zero_grad()`(client.py:34) 会清掉，**不泄漏**。"),
    ("投毒路径的 PGD 梯度是否泄漏进参数更新", "**会，且不被清除**",
     "`zero_grad()`(client.py:34) 早于 `fetch_data()`(client.py:36)，"
     "而 `pgd_attack` 的 `loss.backward()`(fba.py:16) 在 fetch 期间累加进参数，"
     "随后直接被 `scaler.step()`(client.py:72) 使用。**这是唯一确认的系统性差异。**"),
]

THREAT_MODEL = [
    ("是否实现了针对鲁棒聚合的隐蔽性", "否",
     "`fba.py` 中没有 norm clipping、没有模仿良性更新、没有对更新方向的约束。"
     "攻击者只是在本地数据上按投毒目标正常训练。"),
    ("威胁模型是否假设服务器做异常检测", "否",
     "`server.py` 只有 `agg_avg`；`main.py` 的 `--agg_rule` 解析后从不使用。"
     "整个实现假设的是**朴素 FedAvg，无任何防御**。"),
    ("是否针对个性化阶段而非聚合阶段", "是",
     "README 称核心是'用自然特征作触发器'；`pfl.py` 只实现 FedBN，"
     "ASR 在各客户端的 `local_model`（即个性化模型）上评估(main.py:131)。"),
]


def _layer_group(name: str) -> str:
    return name.split(".")[0] if "." in name else name


def _l2_audit(ctx: AuditContext, alpha: float, seed: int) -> Dict[str, Any]:
    run_path = ctx.run_dir("attack", alpha, seed)
    if run_path is None:
        return {}

    bundle = load_run_bundle(ctx, run_path, cache_models=False)
    records = bundle.client_records()
    global_state = bundle.global_state()

    updates: List[torch.Tensor] = []
    layer_rows: List[Dict[str, Any]] = []
    for record in records:
        state = bundle.client_state(record["client_id"])
        updates.append(flatten_state_dict(state, reference=global_state))

        by_layer: Dict[str, float] = defaultdict(float)
        for key in sorted(state.keys()):
            value = state[key]
            if not torch.is_tensor(value) or not value.dtype.is_floating_point:
                continue
            diff = (value.detach().cpu().float()
                    - global_state[key].detach().cpu().float())
            by_layer[_layer_group(key)] += float((diff ** 2).sum().item())
        row = {"alpha": alpha, "seed": seed,
               "client_id": int(record["client_id"]),
               "is_malicious": bool(record["is_malicious"])}
        row.update({f"l2_{k}": float(np.sqrt(v)) for k, v in by_layer.items()})
        layer_rows.append(row)

    stacked = torch.stack(updates, dim=0)
    median = stacked.median(dim=0).values

    per_client = pd.DataFrame(layer_rows)
    per_client["update_l2_to_global"] = [float(u.norm().item()) for u in updates]
    per_client["l2_to_median"] = [float((u - median).norm().item()) for u in updates]
    per_client["n_participations"] = [
        int(r.get("n_participations", 0)) for r in records]
    per_client["n_local_samples"] = [
        int(r.get("n_local_samples", 0)) for r in records]
    return {"per_client": per_client}


def run(ctx: AuditContext) -> str:
    sections = ["## 任务 1：`l2_to_median` 的完美分离是真的吗？", ""]

    # ---------------- A. 静态代码审计 ----------------
    static_frame = pd.DataFrame(
        [{"混杂源": a, "结论": b, "证据（代码行号）": c} for a, b, c in STATIC_FINDINGS])
    ctx.save_evidence(static_frame, "task1_static_audit")
    sections += ["### 1.1 静态代码审计", "", md_table(static_frame), ""]

    # AMP 状态：决定梯度泄漏的量级
    scaler = torch.cuda.amp.GradScaler()
    amp_info = {
        "当前运行环境 cuda 可用": torch.cuda.is_available(),
        "当前 GradScaler.is_enabled()": scaler.is_enabled(),
        "当前 GradScaler 初始 scale": (float(scaler.get_scale())
                                       if scaler.is_enabled() else 1.0),
    }
    for alpha, seed in ctx.pairs():
        run_path = ctx.run_dir("attack", alpha, seed)
        if run_path is not None:
            amp_info[f"训练时 device (α={alpha}, s={seed})"] = \
                load_run_meta(run_path).get("device", "未记录")
            break
    sections += ["### 1.2 AMP 状态（决定梯度泄漏的量级）", "", md_kv(amp_info), "",
                 "若 `GradScaler` 是 enabled（GPU），任务梯度被放大 S 倍后在 "
                 "`scaler.step` 整体除回，**未放大的 PGD 梯度被除成 ∇/S ≈ 0**，"
                 "泄漏几乎不起作用；若 disabled（CPU，S=1），泄漏是全强度的。", ""]

    # ---------------- B. meta.json 统计 ----------------
    sections += ["### 1.3 配置类混杂（meta.json）", ""]
    meta_rows: List[Dict[str, Any]] = []
    for alpha, seed in ctx.pairs():
        run_path = ctx.run_dir("attack", alpha, seed)
        if run_path is None:
            continue
        meta = load_run_meta(run_path)
        frame = pd.DataFrame(meta["clients"])
        for column in ("n_participations", "n_local_samples"):
            if column not in frame.columns:
                continue
            stats = group_compare(frame[column], frame["is_malicious"], column)
            stats.update({"alpha": alpha, "seed": seed})
            meta_rows.append(stats)
    if meta_rows:
        meta_frame = pd.DataFrame(meta_rows)
        ctx.save_evidence(meta_frame, "task1_meta_confounds")
        sections += [md_table(meta_frame[[
            "alpha", "seed", "signal", "benign_mean", "malicious_mean",
            "benign_min", "benign_max", "malicious_min", "malicious_max",
            "separated", "mannwhitney_p"]]), ""]
        separated_any = bool(meta_frame["separated"].any())
        ctx.add(Verdict(
            item="配置类混杂（参与轮次 / 本地样本数）",
            conclusion=("存在系统性差异" if separated_any else "无系统性差异"),
            rule="良性组与恶意组的取值范围完全不重叠",
            measured=f"{int(meta_frame['separated'].sum())}/{len(meta_frame)} "
                     f"项出现完全不重叠",
            action=("需在报告中标注该混杂" if separated_any
                    else "可排除这些混杂源")))
    else:
        sections += ["⚠️ 无可用 meta.json，跳过。", ""]

    # ---------------- C. checkpoint L2 ----------------
    sections += ["### 1.4 更新范数分布（需 checkpoint）", ""]
    all_clients: List[pd.DataFrame] = []
    for alpha, seed in ctx.pairs():
        result = _l2_audit(ctx, alpha, seed)
        if result:
            all_clients.append(result["per_client"])
            print(f"[task1] α={alpha} seed={seed} L2 审计完成")

    if not all_clients:
        sections += ["⚠️ 无可用 attack checkpoint，跳过。", ""]
        ctx.add(Verdict(item="l2_to_median = 1.000", conclusion="未能确定",
                        rule="需要 attack checkpoint", measured="未找到",
                        action="确认 --ckpt-root"))
    else:
        per_client = pd.concat(all_clients, ignore_index=True)
        ctx.save_evidence(per_client, "task1_per_client_l2")

        norm_rows = []
        for (alpha, seed), block in per_client.groupby(["alpha", "seed"]):
            for column in ("update_l2_to_global", "l2_to_median"):
                stats = group_compare(block[column], block["is_malicious"], column)
                stats.update({"alpha": alpha, "seed": seed})
                norm_rows.append(stats)
        norm_frame = pd.DataFrame(norm_rows).sort_values(
            ["alpha", "seed", "signal"], ascending=[False, True, True])
        ctx.save_evidence(norm_frame, "task1_norm_groups")
        sections += [md_table(norm_frame[[
            "alpha", "seed", "signal", "benign_mean", "malicious_mean",
            "benign_max", "malicious_min", "separation_gap", "separated",
            "auc", "mannwhitney_p"]]), ""]

        l2_block = norm_frame[norm_frame["signal"] == "l2_to_median"]
        perfect = l2_block[l2_block["auc"] >= THRESHOLDS["perfect_auc"]]
        separated_any = bool(l2_block["separated"].any())

        if separated_any:
            conclusion = "两组完全不重叠 —— 存在系统性差异"
            action = ("结合 1.1/1.2 与受控对照实验判断成因；"
                      "若成因是投毒目标本身而非实现缺陷，则应表述为"
                      "'该攻击未做隐蔽性优化'而非'我们的实验有问题'")
        else:
            conclusion = "两组有重叠，非完美分离"
            action = "l2 的高 AUC 可能是分布偏移而非系统性差异"

        ctx.add(Verdict(
            item="l2_to_median = 1.000",
            conclusion=conclusion,
            rule=f"良性/恶意的 L2 取值范围完全不重叠（间隙 > "
                 f"{THRESHOLDS['separation_gap']}），且 AUC >= "
                 f"{THRESHOLDS['perfect_auc']}",
            measured=f"{len(perfect)}/{len(l2_block)} 个 α 达到完美 AUC；"
                     f"{int(l2_block['separated'].sum())}/{len(l2_block)} "
                     f"个 α 完全不重叠",
            action=action))

        # 逐层分解
        layer_columns = [c for c in per_client.columns if c.startswith("l2_")
                         and c != "l2_to_median"]
        if layer_columns:
            layer_rows = []
            for (alpha, seed), block in per_client.groupby(["alpha", "seed"]):
                for column in layer_columns:
                    stats = group_compare(block[column], block["is_malicious"],
                                          column.replace("l2_", ""))
                    stats.update({"alpha": alpha, "seed": seed})
                    layer_rows.append(stats)
            layer_frame = pd.DataFrame(layer_rows)
            ctx.save_evidence(layer_frame, "task1_layer_breakdown")
            summary = (layer_frame.groupby("signal")[["auc"]].mean()
                       .sort_values("auc", ascending=False).reset_index()
                       .rename(columns={"signal": "层", "auc": "平均 AUC"}))
            sections += ["#### 逐层分解（哪些层贡献了差异）", "",
                         md_table(summary), ""]

    # ---------------- D. 威胁模型 ----------------
    threat_frame = pd.DataFrame(
        [{"问题": a, "回答": b, "依据": c} for a, b, c in THREAT_MODEL])
    ctx.save_evidence(threat_frame, "task1_threat_model")
    sections += ["### 1.5 威胁模型确认（决定 l2=1.0 该怎么表述）", "",
                 md_table(threat_frame), "",
                 "**这一条决定了后续处理方向**：该攻击的实现层面**没有任何隐蔽性机制**，"
                 "威胁模型也不假设服务器做异常检测。因此即使 `l2_to_median` 完美分离，"
                 "正确的表述是「该攻击本来就不针对鲁棒聚合」，"
                 "而不是「我们的实验设置有问题」—— 前提是能排除实现缺陷（见受控对照实验）。",
                 "",
                 "但由此产生一个**对比公平性**问题：把 `l2_to_median` 当作 baseline "
                 "与 `A^(k)` 并列，等于让一个不设防的攻击去对抗专门针对它的检测器。"
                 "图 C2 若要成立，需要在报告中明确这一前提，"
                 "或引入一个做了范数约束的隐蔽性增强攻击者作为补充对照。", ""]

    return "\n".join(sections)
