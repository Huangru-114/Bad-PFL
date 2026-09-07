#!/usr/bin/env python3
"""Stage B 标定读数（Bad-PFL 侧）—— 与 tf-dpfl 的 `experiments/calibration/
read_calibration.py` **同一套判据**，只是数据源是 implantation CSV 而不是
metrics.json。两边口径必须一致，否则「哪个组合最快到平台」跨库没法一起看。

判据（与 tf-dpfl 侧逐字相同）：

  平台（plateau）
      令 `final` = 最后 3 个评估点的均值。平台轮 = 最小的评估轮 r，使得从 r 起
      **每一个**后续点都落在 `final ± tol` 内。找不到（含点数 ≤3）→ 报 `None`，
      **不猜**。「跑完 80 轮还没平」本身就是结论。

  到平台的 GPU-秒
      = sum(train_wall_s) + sum(eval_wall_s)，都只累加到平台轮为止。
      **实测求和**，不是「均值 × 轮数」。两列由 diag/track.py 写出
      （train 每评估轮一行、eval 由 _on_round_end 回填）；旧 CSV 没有这两列
      → 报 None 而不是 0。

纯 stdlib（csv + argparse），不需要 pandas / numpy / GPU。

    python -m diag.read_calibration
    python -m diag.read_calibration --tol 0.005 --results-dir results/raw --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# run_exp1 的 calib tag：e1calib_steps<N>_s<seed>（fedrep 臂是 e1calib_fedrep_...）
TAG_RE = re.compile(r"e1calib(?:_(?P<arm>\w+?))?_steps(?P<steps>\d+)_s(?P<seed>\d+)\.csv$")


def _f(x) -> Optional[float]:
    """空串 / 'nan' / 非数 → None。**绝不返回 0.0**（铁律 #5）。"""
    if x is None or str(x).strip() == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def plateau_round(rows: List[Dict[str, Any]], key: str, tol: float,
                  tail: int = 3) -> Optional[int]:
    usable = [(int(r["round"]), _f(r.get(key))) for r in rows if r.get("round")]
    usable = [(r, v) for r, v in usable if v is not None]
    if len(usable) <= tail:
        return None
    usable.sort()
    final = sum(v for _, v in usable[-tail:]) / tail
    for i, (r, _) in enumerate(usable):
        if all(abs(v - final) <= tol for _, v in usable[i:]):
            return r
    return None


def cost_up_to(rows: List[Dict[str, Any]], upto: Optional[int]) -> Dict[str, Optional[float]]:
    sel = [r for r in rows
           if r.get("round") and (upto is None or int(r["round"]) <= upto)]
    tr = [_f(r.get("train_wall_s")) for r in sel]
    ev = [_f(r.get("eval_wall_s")) for r in sel]
    tr = [v for v in tr if v is not None]
    ev = [v for v in ev if v is not None]
    return {
        "train_s": round(sum(tr), 1) if tr else None,
        "eval_s": round(sum(ev), 1) if ev else None,
        "total_s": round(sum(tr) + sum(ev), 1) if (tr and ev) else None,
        "n_eval_rows": len(sel) or None,
    }


def load_cells(results_dir: Path) -> List[Dict[str, Any]]:
    cells = []
    for path in sorted(results_dir.glob("*e1calib*.csv")):
        m = TAG_RE.search(path.name)
        if not m:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        cells.append({"path": path, "arm": m.group("arm") or "fedbn",
                      "local_steps": int(m.group("steps")),
                      "seed": int(m.group("seed")), "rows": rows})
    return cells


def _fmt(v, spec="{:.1f}"):
    return "n/a" if v is None else spec.format(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="平台带宽（绝对值）。默认 0.01 = 1 个百分点")
    ap.add_argument("--mta-col", default="mta_personalized")
    ap.add_argument("--asr-col", default="asr_paper_filtered_benign",
                    help="默认用 filtered 口径（排除目标类），与报告正文一致")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    cells = load_cells(results_dir)
    if not cells:
        print(f"[calib] {results_dir} 下没有 *e1calib*.csv —— 先跑\n"
              f"        python -m diag.run_exp1 --stage calib --seeds 0 1 --execute")
        return 1

    print(f"[calib] tol = ±{args.tol}；MTA={args.mta_col}  ASR={args.asr_col}")
    print(f"[calib] {len(cells)} 个格子，来自 {results_dir}\n")

    rows_out = []
    for c in cells:
        r = c["rows"]
        p_mta = plateau_round(r, args.mta_col, args.tol)
        p_asr = plateau_round(r, args.asr_col, args.tol)
        both = None if (p_mta is None or p_asr is None) else max(p_mta, p_asr)
        rows_out.append({
            "arm": c["arm"], "local_steps": c["local_steps"], "seed": c["seed"],
            "n_eval_points": len(r),
            "plateau_round_mta": p_mta, "plateau_round_asr": p_asr,
            "plateau_round": both,
            "cost_to_plateau": cost_up_to(r, both),
            "cost_full_run": cost_up_to(r, None),
            "final_mta": _f(r[-1].get(args.mta_col)) if r else None,
            "final_asr": _f(r[-1].get(args.asr_col)) if r else None,
        })

    hdr = (f"{'arm':>7} {'steps':>6} {'seed':>5} {'plat_MTA':>9} {'plat_ASR':>9} "
           f"{'GPU-s→plat':>11} {'train_s':>9} {'eval_s':>8} {'eval%':>6} "
           f"{'final_MTA':>10} {'final_ASR':>10}")
    print(hdr)
    print("-" * len(hdr))
    for o in sorted(rows_out, key=lambda d: (d["arm"], d["local_steps"], d["seed"])):
        full = o["cost_full_run"]
        frac = (full["eval_s"] / full["total_s"]
                if full["eval_s"] is not None and full["total_s"] else None)
        print(f"{o['arm']:>7} {o['local_steps']:>6} {o['seed']:>5} "
              f"{_fmt(o['plateau_round_mta'], '{:.0f}'):>9} "
              f"{_fmt(o['plateau_round_asr'], '{:.0f}'):>9} "
              f"{_fmt(o['cost_to_plateau']['total_s']):>11} "
              f"{_fmt(o['cost_to_plateau']['train_s']):>9} "
              f"{_fmt(o['cost_to_plateau']['eval_s']):>8} "
              f"{('n/a' if frac is None else f'{100 * frac:.0f}%'):>6} "
              f"{_fmt(o['final_mta'], '{:.4f}'):>10} "
              f"{_fmt(o['final_asr'], '{:.4f}'):>10}")

    print("\n读法：")
    print("  GPU-s→plat = 到平台的实测墙钟（train + eval 求和），**选预算就看这一列**")
    print("  eval%      = 评估占全程墙钟的比例；它高 -> 降轮数比提 local_steps 更划算")
    print("  n/a = 该量无定义或判不出来（如 80 轮还没平 / 旧 CSV 没有时间列），不是 0")
    print("\n注意：本表的 local budget 单位是 **step**，tf-dpfl 侧是 **epoch**。")
    print("      两库各标各的，只共享判据；不要把两边的 steps/epochs 直接对读。")

    if args.json:
        print("\n" + json.dumps(rows_out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
