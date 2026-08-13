"""步骤 0：清点现有产物，并诊断 exp_D α=1.0 缺失的原因。

关键判断：如果 α=1.0 的 **attack checkpoint 存在**而 `expD_a1.0_*.csv` 不存在，
那么补齐这个点**根本不需要重训**，重跑一次 `exp_d` 即可 —— 这会把原计划里
"任务 4.2 需要重训"降级成一条几分钟的命令。
"""

from __future__ import annotations

import glob as globlib
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .common import AuditContext, Verdict, discover_runs, md_table

__all__ = ["run"]

EXP_CSV_RE = re.compile(r"exp(?P<exp>[A-D])_.*?a(?P<alpha>[0-9.]+)_s(?P<seed>\d+)\.csv$")


def _csv_inventory(raw_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(globlib.glob(str(Path(raw_dir) / "exp*_*.csv"))):
        match = EXP_CSV_RE.search(Path(path).name)
        if not match:
            continue
        try:
            frame = pd.read_csv(path)
            n_rows, n_cols = len(frame), len(frame.columns)
        except Exception as exc:  # noqa: BLE001 - 清点阶段不因单个坏文件中断
            n_rows, n_cols = -1, -1
            print(f"[inventory] 读取失败 {path}: {exc}")
        rows.append({"exp": match.group("exp"), "alpha": float(match.group("alpha")),
                     "seed": int(match.group("seed")), "path": path,
                     "n_rows": n_rows, "n_cols": n_cols})
    return pd.DataFrame(rows)


def run(ctx: AuditContext) -> str:
    ctx.runs = discover_runs(ctx.ckpt_root)
    csvs = _csv_inventory(ctx.raw_dir)

    ctx.save_evidence(ctx.runs, "inventory_checkpoints")
    ctx.save_evidence(csvs, "inventory_csv")

    sections = ["## 步骤 0：产物清点", "",
                f"- checkpoint 根目录：`{ctx.ckpt_root}`",
                f"- 原始 CSV 目录：`{ctx.raw_dir}`", "",
                "### checkpoint", "", md_table(ctx.runs), "",
                "### 结果 CSV", "", md_table(csvs), ""]

    # --- 完整性检查 ---
    problems: List[str] = []
    for _, row in ctx.runs.iterrows():
        if not row["has_meta"]:
            problems.append(f"`{Path(row['path']).name}` 缺少 meta.json")
        if not row["has_global"]:
            problems.append(f"`{Path(row['path']).name}` 缺少 global.pt")
        if row["mode"] == "attack" and not row["has_generator"]:
            problems.append(f"`{Path(row['path']).name}` 是 attack run 但缺少 generator.pt")
    if problems:
        sections += ["### 完整性问题", ""] + [f"- {p}" for p in problems] + [""]

    # --- exp_D 缺失点诊断 ---
    missing: List[str] = []
    commands: List[str] = []
    have_d = set()
    if not csvs.empty:
        have_d = {(float(a), int(s))
                  for a, s in csvs[csvs["exp"] == "D"][["alpha", "seed"]]
                  .itertuples(index=False)}

    attack_runs = ctx.runs[ctx.runs["mode"] == "attack"] if not ctx.runs.empty \
        else pd.DataFrame()
    for _, row in attack_runs.iterrows():
        key = (float(row["alpha"]), int(row["seed"]))
        if key in have_d:
            continue
        missing.append(f"α={key[0]}, seed={key[1]}")
        commands.append(
            f"python -m diag.exp_d --ckpt-dir {row['path']} "
            f"--out {ctx.raw_dir}/expD_a{key[0]}_s{key[1]}.csv")

    if missing:
        ctx.add(Verdict(
            item="exp_D 结果缺失点",
            conclusion="无需重训",
            rule="attack checkpoint 存在但 expD CSV 缺失 -> 只需重跑 exp_d",
            measured=f"缺失 {len(missing)} 个点：{'; '.join(missing)}",
            action="执行下方命令补齐（每个点几分钟，不需重新训练）",
            detail="\n".join(f"    {c}" for c in commands)))
        sections += ["### exp_D 缺失点诊断", "",
                     f"以下 (α, seed) 有 attack checkpoint 但没有 expD CSV，"
                     f"**只需重跑 `exp_d`，不需要重训**：", "",
                     "```bash"] + commands + ["```", ""]
    elif not attack_runs.empty:
        sections += ["### exp_D 缺失点诊断", "",
                     "所有 attack checkpoint 都有对应的 expD CSV，无缺失。", ""]
    else:
        ctx.add(Verdict(
            item="exp_D 结果缺失点",
            conclusion="未能确定",
            rule="需要至少一个 attack checkpoint 才能诊断",
            measured="未发现任何 attack checkpoint",
            action="确认 --ckpt-root 指向正确"))

    return "\n".join(sections)
