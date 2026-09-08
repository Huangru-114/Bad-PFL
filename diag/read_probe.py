#!/usr/bin/env python3
"""读 `probe_fedrep.sbatch` 的四个格子 —— 纯 stdlib，登录节点直接跑。

    python3 -m diag.read_probe                       # results/raw 下的 probe_* 
    python3 -m diag.read_probe --tail 3              # 末 N 个评估点取均值

**为什么不用 `read_calibration`**：那个只认 `e1calib` 网格（`TAG_RE`），
探针是另一套预算，混进同一张表会把「谁最快到平台」的结论搞错。

**为什么默认列是 unfiltered**：要对读的锚点是 `diag/HANDOFF.md:325` 的
「上游 main.py 复现出 Avg ASR 91.95%」，那是**论文口径 = 不过滤目标类**
（`main.py:127-138`）。报告正文用 filtered，但**对拍复现必须用同一把尺**。
换算见 diag/METRICS.md：unfiltered ≈ filtered * 0.9 + 0.1。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

TAG_RE = re.compile(r"_probe_(?P<name>[A-Z]\d?)_(?P<arm>fedbn|fedrep)_s(?P<steps>\d+)\.csv$")


def _f(x):
    """空串 / nan / 非数 → None。**绝不返回 0.0**（铁律 #5）。"""
    if x is None or str(x).strip() == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def tail_mean(rows, key, tail):
    """末 `tail` 个**有值**的点的均值；一个都没有 → None。"""
    vals = [v for v in (_f(r.get(key)) for r in rows) if v is not None]
    return sum(vals[-tail:]) / len(vals[-tail:]) if vals else None


def load(results_dir: Path):
    out = []
    for path in sorted(results_dir.glob("exp_ij_implantation_*probe_*.csv")):
        m = TAG_RE.search(path.name)
        if not m:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out.append({"name": m.group("name"), "arm": m.group("arm"),
                    "steps": int(m.group("steps")), "rows": rows, "path": path})
    return sorted(out, key=lambda d: d["name"])


def _fmt(v, spec="{:.4f}"):
    return "n/a" if v is None else spec.format(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--tail", type=int, default=3, help="末 N 个评估点取均值")
    ap.add_argument("--mta-col", default="mta_personalized")
    ap.add_argument("--asr-col", default="asr_paper_benign",
                    help="默认 unfiltered（论文口径），才能与 HANDOFF.md:325 对读")
    args = ap.parse_args(argv)

    cells = load(Path(args.results_dir))
    if not cells:
        print(f"[probe] {args.results_dir} 下没有 probe_* 的植入 CSV —— 先跑\n"
              f"        sbatch diag/probe_fedrep.sbatch")
        return 1

    print(f"[probe] 末 {args.tail} 点均值；ASR 列 = {args.asr_col}")
    print(f"[probe] 锚点：上游 main.py 论文配置复现 = 0.9195（论文报 0.8222）\n")
    hdr = f"{'格':>4} {'arm':>7} {'总步':>5} {'轮':>5} {'点':>4} {'MTA':>9} {'ASR':>9}  说明"
    print(hdr); print("-" * (len(hdr) + 20))
    NOTE = {"A": "锚点：论文配置 + fedbn",
            "B1": "现状 1:1（default_head_steps）",
            "B2": "5:1，表示=1 epoch（Flower Table-1）",
            "B3": "10:1，表示=1 epoch（FedRep 原文）"}
    for c in cells:
        rounds = [int(r["round"]) for r in c["rows"] if r.get("round")]
        print(f"{c['name']:>4} {c['arm']:>7} {c['steps']:>5} "
              f"{(max(rounds) if rounds else 0):>5} {len(c['rows']):>4} "
              f"{_fmt(tail_mean(c['rows'], args.mta_col, args.tail)):>9} "
              f"{_fmt(tail_mean(c['rows'], args.asr_col, args.tail)):>9}  "
              f"{NOTE.get(c['name'], '')}")

    print("\n判读：")
    print("  A 的 ASR 落在 0.82–0.92  → 管线成立，FedRep 的数才可解释；否则先修管线。")
    print("  B1→B2→B3 的 MTA 单调上升 → 坐实 1:1 的头:表示比是问题；不动则另有原因。")
    print("  n/a = 该列无定义或没量到，不是 0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
