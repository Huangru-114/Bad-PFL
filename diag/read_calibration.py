#!/usr/bin/env python3
"""Stage B 标定读数（Bad-PFL 侧）—— 与 tf-dpfl 的 `experiments/calibration/
read_calibration.py` **同一套判据**，只是数据源是 implantation CSV 而不是
metrics.json。两边口径必须一致，否则「哪个组合最快到平台」跨库没法一起看。

判据（与 tf-dpfl 侧逐字相同）：

  平台（plateau）
      令 `final` = 最后 3 个评估点的均值。平台轮 = 最小的评估轮 r，使得从 r 起
      **每一个**后续点都落在 `final ± tol` 内。找不到 → 报 `None` 并给出原因，
      **不猜**。「跑完 80 轮还没平」本身就是结论。

      **尾窗排除**（2026-09，实测教训）：r 之后不足 2×tail 个点时判 `None`。
      `final` 就是最后 tail 个点算出来的，平台落在尾窗里等于自证 ——
      第一批标定的四档全部报出 72–80 的"平台"，而 MTA 还在 0.29→0.65 一路爬。

  单位成本（而不是求和）
      `s_per_round` = **mean**(train_wall_s)，`s_per_eval` = mean(eval_wall_s)。
      ⚠️ 植入行只在**评估轮**写（`track.py` 每轮算 `_t_train_s`，但行是
      `_evaluate_now` 写的），每行携带的是**它自己那一轮**的训练秒数 ——
      所以 80 轮 / eval_every=2 的 run 只有 40 行，`sum(train_wall_s)`
      覆盖 40 轮而不是 80，**训练时间少算 eval_every 倍**。
      `eval_wall_s` 没有这个问题（评估只在评估轮发生，行与事件一一对应）。

      单位成本是唯一能跨 (轮数, eval_every) 外推的量 —— 标定跑 80 轮/ev=2、
      主力跑 200 轮/ev=5，评估是固定成本、训练随轮数走，两者不能直接对读。
      `--main-rounds` / `--main-eval-every` 就是用来做这个外推（= T_main）。

  到平台的 GPU-秒
      **只在平台成立时才有定义**；判不出来报 None，不是退回全程成本。

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


def _plateau_scan(rows: List[Dict[str, Any]], key: str, tol: float,
                  tail: int = 3, frac: float = 0.25) -> tuple:
    """→ ``(平台轮 or None, 判不出来的原因 or None)``。

    平台 = 最小的评估轮 r，使得从 r 起**每个**后续点都落在 `final ± tol` 内，
    `final` = 最后 `tail` 个点的均值。

    **两道排除**（2026-09 加，实测教训）：

    1. r 之后（含 r）不足 ``tail + 1`` 个点 —— `final` 本身就是最后 tail 个点
       算出来的，平台落在那里等于自证。
    2. 平坦段没有覆盖至少最后 ``frac`` 比例的轮数（即 ``r > (1-frac) * 最后一轮``）。
       「±0.01 内待了 6 轮」不是收敛，只是**走得慢**；要说收敛，平坦段至少得
       占住一段有意义的轮数。

    实测：80 轮 / eval_every=2 的第一批标定里，四档报出 72–80 的"平台"，
    而 MTA 还在 0.29 → 0.65 一路爬 —— 两道排除后全部正确地变成 None。
    """
    usable = [(int(r["round"]), _f(r.get(key))) for r in rows if r.get("round")]
    usable = [(r, v) for r, v in usable if v is not None]
    if len(usable) <= tail:
        return None, f"可用点只有 {len(usable)} 个（需 > {tail}）"
    usable.sort()
    last_round = usable[-1][0]
    latest_ok = (1.0 - frac) * last_round
    final = sum(v for _, v in usable[-tail:]) / tail
    for i, (r, _) in enumerate(usable):
        if all(abs(v - final) <= tol for _, v in usable[i:]):
            n_after = len(usable) - i
            if n_after < tail + 1:
                return None, (f"疑似平台在第 {r} 轮，其后只剩 {n_after} 个点"
                              f"（需 ≥ {tail + 1}）—— 与「尾部碰巧平」无法区分")
            if r > latest_ok:
                return None, (f"疑似平台在第 {r} 轮，但只覆盖最后 "
                              f"{100 * (last_round - r) / last_round:.0f}% 的轮数"
                              f"（需 ≥ {100 * frac:.0f}%）—— 是走得慢，不是收敛")
            return r, None
    return None, f"跑完 {last_round} 轮仍未进入 ±{tol} 带"


def plateau_round(rows: List[Dict[str, Any]], key: str, tol: float,
                  tail: int = 3) -> Optional[int]:
    return _plateau_scan(rows, key, tol, tail)[0]


def eval_spacing(rows: List[Dict[str, Any]]) -> Optional[int]:
    """评估轮之间的间隔（= 跑这一格时的 ``eval_every``），从数据自身推出来。

    不读 config —— CSV 要能独立判读。取相邻轮差的众数（中途漏一次评估不至于翻车）。
    """
    rs = sorted(int(r["round"]) for r in rows if r.get("round"))
    if len(rs) < 2:
        return None
    diffs = [b - a for a, b in zip(rs, rs[1:])]
    return max(set(diffs), key=diffs.count)


def cost_model(rows: List[Dict[str, Any]],
               upto: Optional[int] = None) -> Dict[str, Optional[float]]:
    """把两列墙钟拆成**可外推的单位成本**，而不是求一个不可比的和。

    ⚠️ **为什么不能直接 sum(train_wall_s)**：植入行只在**评估轮**写
    （``track.py`` 的 ``_on_round_end`` 每轮都算 ``_t_train_s``，但行是在
    ``_evaluate_now`` 里写的）。每行携带的是**它自己那一轮**的训练秒数 ——
    于是 80 轮 / eval_every=2 的 run 只有 40 行，求和覆盖的是 40 轮，
    **训练时间少算一半**。正确的单位量是 ``mean``，再乘轮数外推。
    （2026-09 实测：标定表里 fedbn/15 的 87.2 s 其实是 40 轮的训练时间，
    80 轮应为 ~174 s；由此估出来的 T_main 会低一倍。）

    ``eval_wall_s`` 没有这个问题：评估**只在评估轮发生**，行与事件一一对应，
    所以它是求和。

    返回的 ``train_s`` / ``run_s`` 是**外推值**（单位成本 × 轮数），
    字段名带 ``_est`` 提醒这一点。
    """
    sel = [r for r in rows
           if r.get("round") and (upto is None or int(r["round"]) <= upto)]
    tr = [v for v in (_f(r.get("train_wall_s")) for r in sel) if v is not None]
    ev = [v for v in (_f(r.get("eval_wall_s")) for r in sel) if v is not None]
    rounds = [int(r["round"]) for r in sel]
    last_round = max(rounds) if rounds else None

    s_round = (sum(tr) / len(tr)) if tr else None
    s_eval = (sum(ev) / len(ev)) if ev else None
    train_est = (s_round * last_round) if (s_round is not None and last_round) else None
    eval_total = round(sum(ev), 1) if ev else None
    run_est = (round(train_est + sum(ev), 1)
               if (train_est is not None and ev) else None)
    return {
        "s_per_round": round(s_round, 3) if s_round is not None else None,
        "s_per_eval": round(s_eval, 3) if s_eval is not None else None,
        "train_s_est": round(train_est, 1) if train_est is not None else None,
        "eval_s": eval_total,
        "run_s_est": run_est,
        "n_eval_rows": len(sel) or None,
        "last_round": last_round,
        "eval_every": eval_spacing(sel),
    }


def project_run(cm: Dict[str, Optional[float]], total_round: int,
                eval_every: int) -> Optional[float]:
    """外推「另一套 (total_round, eval_every) 下这一格要跑多久」，单位秒。

    这正是 T_main：标定跑的是 80 轮 / eval_every=2，主力是 200 轮 / eval_every=5，
    两者**不能直接对读** —— 评估是固定成本、训练随轮数走，比例完全不同。
    """
    if cm["s_per_round"] is None or cm["s_per_eval"] is None:
        return None
    n_eval = total_round // max(1, int(eval_every))
    return round(cm["s_per_round"] * total_round + cm["s_per_eval"] * n_eval, 1)


def load_cells(results_dir: Path) -> List[Dict[str, Any]]:
    cells = []
    # **只认 implantation CSV**：run_fl 每个 run 还会写一份逐 edge 的
    # `exp_ij_edge_<run_id>.csv`（run_fl.py:670），尾巴与 implantation 完全同名 ——
    # 用 `*e1calib*.csv` 会把它一起收进来，于是每个格子出现两行、其中一行
    # 因为没有 mta/asr/时间列而全是 n/a。（2026-09 实测：6 个格子印成 12 行。）
    for path in sorted(results_dir.glob("exp_ij_implantation_*e1calib*.csv")):
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
    ap.add_argument("--plateau-frac", type=float, default=0.25,
                    help="平坦段至少要覆盖最后这个比例的轮数才算平台（默认 0.25）")
    ap.add_argument("--main-rounds", type=int, default=200,
                    help="T_main 外推用的主力轮数（exp1.total_round）")
    ap.add_argument("--main-eval-every", type=int, default=5,
                    help="T_main 外推用的主力评估间隔（exp1.eval_every）")
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
        p_mta, why_mta = _plateau_scan(r, args.mta_col, args.tol, frac=args.plateau_frac)
        p_asr, why_asr = _plateau_scan(r, args.asr_col, args.tol, frac=args.plateau_frac)
        both = None if (p_mta is None or p_asr is None) else max(p_mta, p_asr)
        cm = cost_model(r)
        rows_out.append({
            "arm": c["arm"], "local_steps": c["local_steps"], "seed": c["seed"],
            "n_eval_points": len(r),
            "plateau_round_mta": p_mta, "plateau_round_asr": p_asr,
            "plateau_round": both,
            "why_no_plateau_mta": why_mta, "why_no_plateau_asr": why_asr,
            # 到平台的成本**只在平台成立时**才有定义。判不出来就是 None ——
            # 旧实现在这里退回「全程成本」，于是一个全程的数挂着「到平台」的表头。
            "cost_to_plateau": (cost_model(r, both) if both is not None
                                else {k: None for k in cost_model(r)}),
            "cost_full_run": cm,
            "final_mta": _f(r[-1].get(args.mta_col)) if r else None,
            "final_asr": _f(r[-1].get(args.asr_col)) if r else None,
            "projected_main_s": project_run(cm, args.main_rounds, args.main_eval_every),
        })

    hdr = (f"{'arm':>7} {'steps':>6} {'seed':>5} {'pts':>4} {'ev':>3} "
           f"{'plat_MTA':>9} {'plat_ASR':>9} {'GPU-s→plat':>11} "
           f"{'s/round':>8} {'s/eval':>7} {'run_s(est)':>11} {'eval%':>6} "
           f"{'final_MTA':>10} {'final_ASR':>10}")
    print(hdr)
    print("-" * len(hdr))
    for o in sorted(rows_out, key=lambda d: (d["arm"], d["local_steps"], d["seed"])):
        full = o["cost_full_run"]
        frac = (full["eval_s"] / full["run_s_est"]
                if full["eval_s"] is not None and full["run_s_est"] else None)
        print(f"{o['arm']:>7} {o['local_steps']:>6} {o['seed']:>5} "
              f"{o['n_eval_points']:>4} {_fmt(full['eval_every'], '{:.0f}'):>3} "
              f"{_fmt(o['plateau_round_mta'], '{:.0f}'):>9} "
              f"{_fmt(o['plateau_round_asr'], '{:.0f}'):>9} "
              f"{_fmt(o['cost_to_plateau']['run_s_est']):>11} "
              f"{_fmt(full['s_per_round'], '{:.2f}'):>8} "
              f"{_fmt(full['s_per_eval'], '{:.2f}'):>7} "
              f"{_fmt(full['run_s_est']):>11} "
              f"{('n/a' if frac is None else f'{100 * frac:.0f}%'):>6} "
              f"{_fmt(o['final_mta'], '{:.4f}'):>10} "
              f"{_fmt(o['final_asr'], '{:.4f}'):>10}")

    # ── 判不出平台时，把原因说出来 ───────────────────────────────────────
    reasons = [(o, k) for o in rows_out for k in ("mta", "asr")
               if o[f"why_no_plateau_{k}"]]
    if reasons:
        print("\n判不出平台的原因（n/a 不是 0，也不是「没到」）：")
        for o, k in sorted(reasons, key=lambda t: (t[0]["arm"], t[0]["local_steps"])):
            print(f"  {o['arm']:>7} steps={o['local_steps']:<4} {k.upper():<4} "
                  f"{o[f'why_no_plateau_{k}']}")

    # ── T_main 外推 ──────────────────────────────────────────────────────
    print(f"\nT_main 外推（{args.main_rounds} 轮 / eval_every={args.main_eval_every}）：")
    print(f"  {'arm':>7} {'steps':>6} {'train_s':>9} {'eval_s':>8} "
          f"{'T_main_s':>9} {'T_main_h':>9}")
    for o in sorted(rows_out, key=lambda d: (d["arm"], d["local_steps"])):
        cm = o["cost_full_run"]
        t = o["projected_main_s"]
        if t is None:
            print(f"  {o['arm']:>7} {o['local_steps']:>6} {'n/a':>9}")
            continue
        n_eval = args.main_rounds // args.main_eval_every
        print(f"  {o['arm']:>7} {o['local_steps']:>6} "
              f"{cm['s_per_round'] * args.main_rounds:>9.0f} "
              f"{cm['s_per_eval'] * n_eval:>8.0f} {t:>9.0f} {t / 3600:>9.2f}")

    print("\n读法：")
    print("  s/round / s/eval = **单位成本**，是唯一能跨 (轮数, eval_every) 外推的量。")
    print("  run_s(est)  = s/round × 最后一轮 + Σ eval_wall_s。**train 部分是外推**：")
    print("                植入行只在评估轮写，直接求和会把训练少算 eval_every 倍。")
    print("  GPU-s→plat  = 到平台的成本，**平台判不出来时是 n/a**（不是全程成本）。")
    print("  eval%       = 评估占全程；它高 -> 降评估密度比提 local_steps 更划算。")
    print("  n/a = 该量无定义或判不出来，不是 0。")
    print("\n注意：本表的 local budget 单位是 **step**，tf-dpfl 侧是 **epoch**。")
    print("      两库各标各的，只共享判据；不要把两边的 steps/epochs 直接对读。")

    if args.json:
        print("\n" + json.dumps(rows_out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
