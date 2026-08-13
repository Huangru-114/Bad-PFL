"""P0 排查的编排器：跑完全部检查并生成 ``diag/audit/P0_AUDIT.md``。

用法（在集群上，checkpoint 所在处）::

    python -m diag.audit.run_audit \\
        --ckpt-root ./checkpoints \\
        --raw-dir ./results/raw \\
        --out diag/audit/P0_AUDIT.md \\
        --leak-experiment

只想快速看不需要模型权重的部分（任务 2 全部 + 任务 3.1 + exp_D 缺失点诊断）::

    python -m diag.audit.run_audit --meta-only
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from ..analysis import load_raw, summarize_c
from ..config import load_config
from ..exp_common import resolve_device
from . import inventory, task1_l2, task2_sign, task3_numerics, task4_extra
from .common import THRESHOLDS, AuditContext, md_kv, md_table

__all__ = ["main"]

CODE_CHANGES = [
    ("diag/analysis.py",
     "新增 `SIGNAL_DIRECTION` 与 `signal_direction()`；`summarize_c` 先按极性"
     "修正再算 AUC，并新增 `auc_raw` / `direction` / `direction_declared` 列",
     "**已有 expC_summary.csv 中 `sign_consistency` 的 AUC 作废**，"
     "需从 raw CSV 重算。`A_observable` 与 `l2_to_median` 不受影响。"
     "**不需要重新训练。**"),
    ("diag/exp_common.py",
     "`RunBundle` / `load_bundle` 新增 `cache_models` 开关（默认 True，行为不变）",
     "纯新增。审计脚本关掉它以避免 100 个 resnet10 常驻（约 2GB/bundle）。"
     "对已有结果无影响。"),
    ("diag/metrics.py",
     "`baseline_signals` docstring 写明极性约定",
     "仅注释，无行为变化。"),
    ("diag/audit/**",
     "新增一次性排查工具（本目录全部文件）",
     "只读诊断 + 一次明确标注的受控对照重训。对已有结果无影响。"),
    ("原仓库文件",
     "零改动（`git diff --stat main -- . ':(exclude)diag'` 为空）",
     "梯度泄漏对照臂用 monkey-patch 实现，`client.py` / `fba.py` 未被修改。"),
]

LIMITATIONS = [
    "**seed 覆盖有限**：若本次只跑了 seed=0，所有『完全不重叠』『AUC=1.000』的"
    "判断都只有单点支撑，无法区分真实效应与单次抽样偶然。上文任何结论都不应"
    "在补齐多 seed 之前写进论文。",
    "**恶意客户端数量少**（正式配置 10 个）：仅恶意组的相关系数、分层 AUC "
    "统计功效都很低，表中已给出 n，请连同 n 一起判读。",
    "**受控对照的规模远小于正式配置**：它只能回答『泄漏是否产生可测的更新范数"
    "差异』这一定性问题，**不能外推**到 100 客户端 / 300 轮的绝对量级。",
    "**`A^(k)` 的适用下界是从这一批数据归纳的**：换数据集或类别数需要重新确认。",
    "**`sign_consistency` 的定义仍是对文献的推广**：原文 (Wang et al., "
    "AISTATS'24) 的 `m_k = 1[|mean_i sign(g_ik)| > tau]` 是逐维度掩码，"
    "不是 per-client 分数。本次只修了极性，**没有解决定义是否忠实于原文的问题**，"
    "仍需人工对照原文确认。",
    "**`max_t` 的次序统计量偏置只做了诊断，没有修**：报告中的 `mean_t` 变体是"
    "敏感性分析，主定义未改动。是否要改指标定义，需要你拍板。",
]


def _corrected_results(ctx: AuditContext) -> str:
    """第 3 节：修正后的结果表。"""
    sections = ["## 3. 修正后的结果表", ""]
    invalid = ctx.notes.get("invalid_alphas_A_observable", [])

    # --- 实验 C 的 AUC（方向修正后） ---
    sections += ["### 3.1 实验 C：AUC vs α（方向修正后）", ""]
    try:
        summary = summarize_c(load_raw(str(ctx.raw_dir / "expC_*.csv")))
    except FileNotFoundError:
        sections += ["⚠️ 找不到 expC CSV，跳过。", ""]
    else:
        pivot = summary.pivot_table(index=["alpha", "seed"], columns="signal",
                                    values="auc").reset_index()
        pivot = pivot.sort_values(["alpha", "seed"], ascending=[False, True])
        if invalid and "A_observable" in pivot.columns:
            pivot["A_observable_有效性"] = [
                "invalid" if float(a) in invalid else "valid"
                for a in pivot["alpha"]]
        ctx.save_evidence(pivot, "corrected_expC_auc")
        sections += [md_table(pivot), ""]
        if invalid:
            sections += [
                f"⚠️ α ∈ {invalid} 的 `A_observable` 已被判为 **invalid**"
                "（见任务 3），**不参与趋势判读**。在这些 α 上把 0.259 当作"
                "『方向翻转』来解释是没有依据的 —— 那个数字本身不成立。", ""]

    # --- 相关性 ---
    sections += ["### 3.2 实验 C：相关性（Pearson / Spearman）", ""]
    path = ctx.evidence_dir / "task4_correlations.csv"
    if path.exists():
        sections += [md_table(pd.read_csv(path)), "",
                     "AUC 衡量可分性，相关系数衡量代理有效性 —— **两个都要看**。", ""]
    else:
        sections += ["⚠️ 未生成，见任务 4.1。", ""]

    # --- 实验 D 三信号 ---
    sections += ["### 3.3 实验 D：分子 / 分母 / 比值 三信号对比", ""]
    path = ctx.evidence_dir / "task4_d_decomposition.csv"
    if path.exists():
        sections += [md_table(pd.read_csv(path)), ""]
    else:
        sections += ["⚠️ 未生成，见任务 4.3。", ""]
    return "\n".join(sections)


def build_report(ctx: AuditContext, body_sections: List[str],
                 started: str) -> str:
    verdict_frame = pd.DataFrame([v.as_row() for v in ctx.verdicts])

    header = [
        "# P0_AUDIT：实验 C / D 结果有效性排查", "",
        "> **本报告由 `python -m diag.audit.run_audit` 自动生成。**",
        "> 每条判定都附带**写死的规则**与**实测值**，可自行复核；"
        "规则不满足时输出「未能确定」，不猜。", "",
        md_kv({
            "生成时间": started,
            "checkpoint 根目录": str(ctx.ckpt_root),
            "原始 CSV 目录": str(ctx.raw_dir),
            "证据目录": str(ctx.evidence_dir),
            "torch": torch.__version__,
            "device": str(ctx.device),
        }), "",
        "## 1. 结论速查表", "",
        md_table(verdict_frame) if len(verdict_frame) else "_（无判定）_", "",
        "判定所用的全部阈值：", "", md_kv(THRESHOLDS), "",
        "---", "", "## 2. 逐任务的详细证据", "",
    ]

    footer = ["", "---", "", "## 4. 代码修改清单", "",
              md_table(pd.DataFrame(
                  [{"文件": a, "改动": b, "对已有结果的影响": c}
                   for a, b, c in CODE_CHANGES])), "",
              "**需要重跑的范围**：只需从现有 raw CSV 重算实验 C 的汇总"
              "（`summarize_c`），**不需要重新训练任何模型**。", "",
              "---", "", "## 5. 已知限制与未解决项", ""]
    footer += [f"{i}. {item}" for i, item in enumerate(LIMITATIONS, start=1)]
    footer += ["", f"证据 CSV 全部落在 `{ctx.evidence_dir}/`，"
                   "每张表都可以独立复核。"]

    return "\n".join(header + body_sections + footer)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P0 排查编排器")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--raw-dir", default="./results/raw")
    parser.add_argument("--out", default="diag/audit/P0_AUDIT.md")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--alphas", type=float, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--meta-only", action="store_true",
                        help="跳过所有需要模型权重的检查（快速预览）")
    parser.add_argument("--leak-experiment", action="store_true",
                        help="跑梯度泄漏的受控 A/B 对照重训")
    parser.add_argument("--leak-synthetic", action="store_true",
                        help="受控对照使用合成数据（不需要 torchvision）")
    args = parser.parse_args(argv)

    started = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = Path(args.out)
    evidence_dir = Path(args.evidence_dir or (out_path.parent / "evidence"))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    ctx = AuditContext(
        ckpt_root=Path(args.ckpt_root), raw_dir=Path(args.raw_dir),
        evidence_dir=evidence_dir, cfg=load_config(args.config),
        device=resolve_device(args.device),
        alphas=args.alphas, seeds=args.seeds)

    steps = [("步骤 0：清点", inventory.run),
             ("任务 2：符号约定", task2_sign.run),
             ("任务 3：数值健康度", task3_numerics.run),
             ("任务 1：l2 混杂源", task1_l2.run),
             ("任务 4：补充量", task4_extra.run)]

    body: List[str] = []
    for name, func in steps:
        print(f"\n===== {name} =====")
        try:
            body.append(func(ctx))
            body.append("")
        except Exception:  # noqa: BLE001 - 单项失败不应中断整份审计
            trace = traceback.format_exc()
            print(trace)
            body += [f"## {name}", "",
                     "⚠️ **本项执行失败**，结论为「未能确定」。堆栈：", "",
                     "```", trace, "```", ""]

    if args.leak_experiment:
        print("\n===== 受控对照：梯度泄漏 =====")
        try:
            from . import leak_experiment
            body.append(leak_experiment.run(ctx, synthetic=args.leak_synthetic))
            body.append("")
        except Exception:  # noqa: BLE001
            trace = traceback.format_exc()
            print(trace)
            body += ["## 受控对照：梯度泄漏", "",
                     "⚠️ **执行失败**，泄漏成因判定为「未能确定」。堆栈：", "",
                     "```", trace, "```", ""]
    else:
        body += ["## 受控对照：梯度泄漏", "",
                 "未运行（需加 `--leak-experiment`）。**因此无法区分**"
                 "『实现缺陷造成 l2 完美分离』与『投毒目标本身就不隐蔽』，"
                 "任务 1 的成因判定停留在「未能确定」。", ""]

    body.append(_corrected_results(ctx))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(ctx, body, started), encoding="utf-8")
    print(f"\n报告已生成：{out_path}")
    print(f"证据 CSV：{evidence_dir}")
    print(f"共 {len(ctx.verdicts)} 条判定，其中 "
          f"{sum(1 for v in ctx.verdicts if v.conclusion == '未能确定')} 条为「未能确定」")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
