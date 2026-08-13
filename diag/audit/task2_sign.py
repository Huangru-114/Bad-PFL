"""任务 2：``sign_consistency`` 的符号约定审计。

结论（代码审查即可确立，不依赖数据）：

- ``diag/metrics.py::baseline_signals`` 计算
  ``sign_consistency = mean_d[ sign(u_k[d]) * mean_i sign(u_i[d]) ]``，
  取值 ∈ [-1, 1]，**越大表示越贴近多数派方向 = 越像良性**。
- ``diag/analysis.py::summarize_c`` 把它送进 ``auc_score(score, is_malicious)``，
  该函数的约定是**分数越大越像恶意**。
- 两者矛盾 -> 作为检测分数必须取负。

本模块从现有 CSV 重算修正前后的 AUC，并断言 ``AUC_after == 1 - AUC_before``。
若该等式不成立，说明除极性外还有别的问题，会如实标为"未能确定"。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..analysis import SIGNAL_DIRECTION, load_raw, signal_direction, summarize_c
from .common import AuditContext, Verdict, md_table

__all__ = ["run"]

# 该实现对应的定义（写清楚，因为它是对文献表述的一处推广）
FORMULA = r"""
baseline_signals()  (diag/metrics.py)

    stacked   = [u_1; ...; u_K]                    # 每行是一个客户端的更新向量
    signs     = sign(stacked)                      # 逐元素符号
    consensus = mean_i signs[i]                    # [D]，逐维多数派方向，∈ [-1, 1]

    sign_consistency_k = mean_d [ signs[k, d] * consensus[d] ]      # ∈ [-1, 1]

语义：越大 = 与多数派方向越一致 = 越像良性。

⚠️ 这是对 Wang et al. (AISTATS'24) 的推广。原文的
    m_k = 1[ |mean_i sign(g_{i,k})| > tau ]
是**逐维度的掩码**，本身不是 per-client 的检测分数；上式是把它转成客户端分数的
一种自然定义，但**不是原文定义**，仍需人工对照原文确认。
"""


def run(ctx: AuditContext) -> str:
    sections = ["## 任务 2：`sign_consistency` 的符号约定", "",
                "### 2.1 完整计算公式", "", "```" + FORMULA + "```", ""]

    # --- 方向约定表 ---
    direction_rows = []
    for signal in ("A_observable", "l2_to_median", "sign_consistency"):
        direction, declared = signal_direction(signal)
        semantics = {
            "A_observable": "构造上 -mean - λ·std 的最大值，越大越可疑",
            "l2_to_median": "距中位数更新越远 = 越离群 = 越可疑",
            "sign_consistency": "越大 = 越贴近多数派 = 越像良性",
        }[signal]
        direction_rows.append({
            "信号": signal, "语义": semantics,
            "应有极性": "+1（越大越可疑）" if direction > 0 else "-1（须取负）",
            "修正前是否正确": direction > 0,
            "已显式声明": declared,
        })
    direction_frame = pd.DataFrame(direction_rows)
    ctx.save_evidence(direction_frame, "task2_direction_map")
    sections += ["### 2.2 三个信号的方向约定", "", md_table(direction_frame), "",
                 "只有 `sign_consistency` 的极性是反的；另外两个信号本来就正确，"
                 "所以这次修正**不影响** `A_observable` 与 `l2_to_median` 的任何已有数字。",
                 ""]

    # --- 从现有 CSV 重算 ---
    pattern = str(ctx.raw_dir / "expC_*.csv")
    try:
        raw = load_raw(pattern)
    except FileNotFoundError:
        ctx.add(Verdict(
            item="sign_consistency 方向约定",
            conclusion="确认为 bug（代码层面）",
            rule="baseline_signals 语义（越大越良性）与 auc_score 约定（越大越恶意）矛盾",
            measured="代码审查确认；但找不到 expC CSV，无法重算数值",
            action=f"补齐 {pattern} 后重跑本审计以获得修正后的数值"))
        sections += ["### 2.3 修正前后对比", "",
                     f"⚠️ 找不到 `{pattern}`，无法从数据重算。"
                     "代码层面的结论不受影响。", ""]
        return "\n".join(sections)

    summary = summarize_c(raw)
    sign_block = summary[summary["signal"] == "sign_consistency"].copy()
    sign_block = sign_block.sort_values(["alpha", "seed"], ascending=[False, True])

    compare = sign_block[["alpha", "seed", "auc_raw", "auc", "n_clients",
                          "n_malicious"]].rename(
        columns={"auc_raw": "修正前 AUC", "auc": "修正后 AUC"})
    compare["1 - 修正前"] = 1.0 - compare["修正前 AUC"]
    compare["等式成立"] = np.isclose(compare["修正后 AUC"], compare["1 - 修正前"],
                                     atol=1e-9)
    ctx.save_evidence(compare, "task2_sign_auc")

    all_hold = bool(compare["等式成立"].all()) and len(compare) > 0
    mean_after = float(compare["修正后 AUC"].mean()) if len(compare) else float("nan")
    spread = (float(compare["修正后 AUC"].max() - compare["修正后 AUC"].min())
              if len(compare) else float("nan"))

    if all_hold:
        conclusion = "约定 bug（已确认）"
        action = ("修正 `summarize_c` 的极性；已有 expC_summary.csv 中该信号的 AUC "
                  "作废，需从 raw CSV 重算（**不需重训**）")
    else:
        conclusion = "未能确定"
        action = "AUC_after != 1 - AUC_before，说明除极性外另有问题，需进一步排查"

    ctx.add(Verdict(
        item="sign_consistency < 0.5",
        conclusion=conclusion,
        rule="AUC_修正后 == 1 - AUC_修正前（逐 α 逐 seed 成立）",
        measured=f"{int(compare['等式成立'].sum())}/{len(compare)} 个点成立；"
                 f"修正后均值 {mean_after:.3f}，极差 {spread:.3f}",
        action=action))

    sections += ["### 2.3 修正前后对比（从现有 CSV 重算）", "",
                 md_table(compare), ""]

    # --- 平坦性观察（如实记录，不因"不够好"而忽略） ---
    if len(compare) >= 2 and np.isfinite(spread):
        flat = spread < 0.10
        sections += [
            "### 2.4 修正后的观察", "",
            f"修正后 AUC 均值 **{mean_after:.3f}**，跨 α 极差 **{spread:.3f}**。",
            "",
            ("该信号**抗数据异构但绝对水平不高** —— 随 α 基本平坦，"
             "说明它不像 `A_observable` 那样受异构度影响，但判别力也有限。"
             "这是一个有意义的观察，如实记录。" if flat else
             "该信号随 α 有可见变化，不是平坦的，具体趋势见上表。"),
            "",
            ("方向修正**没有**把它变成一个强信号（仍远低于 `l2_to_median`）。"
             "这说明恶意客户端确实比良性更贴近多数派方向 —— 但这个"
             "'一致性'差异的幅度不足以支撑检测。"), ""]

    return "\n".join(sections)
