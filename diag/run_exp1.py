"""实验 1 / 1B 的命令生成器（**默认 dry-run**）。

# 为什么是单轴扫描而不是全因子

全因子是 ``|N_m| × |ρ_p| × |seed| = 4 × 4 × 2 = 32`` 个 run。按 40 客户端 /
ResNet-18 / 200 轮估算，单个 run 在一张 GPU 上是小时量级 —— 32 个 run 会把
第一阶段拖成几天，而第一阶段要回答的只是"弱 / 过渡 / 强攻击各自在哪一段"。

所以默认是**十字扫描**：固定 ``ρ_p`` 扫 ``N_m``，固定 ``N_m`` 扫 ``ρ_p``,
交叉点共用同一个 run。4 + 4 − 1 = 7 个点 × 2 seed = 14 个 run。
确认了过渡区在哪里，再用 ``--grid`` 在那一小块上补全因子。

# 三件容易做错的事

1. **每个 run 必须有独立的 ``--run-tag``。** 否则 checkpoint 目录与植入 CSV
   会互相覆盖，而且覆盖是**静默**的。tag 由剂量参数拼出来，一定唯一。
2. **``ρ_p`` 是逐样本的伯努利概率，不是精确比例**（``fba.py:49`` 的
   ``torch.rand(...) <= poison_ratio``）。batch=32 时 ρ_p=0.1 有约 3.4% 的批次
   一个毒样本都没有。报告里写"投毒比例"时要说明这一点。
3. **``--eval-every`` 必须够密。** 实验 1 的核心图是 ASR vs MTA，
   每个 run 只有 ``total_round / eval_every`` 个 (MTA, ASR) 点。200 轮 / 每 5 轮
   = 40 个点；调到 20 就只剩 10 个点，散点图上看不出任何转折。

用法::

    python -m diag.run_exp1                 # 打印命令，不执行
    python -m diag.run_exp1 --stage 1b      # 只看 1B 的调度
    python -m diag.run_exp1 --execute       # 真跑
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from .config import Cfg, load_config
from .schedule import AttackSchedule

__all__ = ["dose_points", "build_commands", "estimate_cost", "main"]


def dose_points(bad_nums: Sequence[int], poison_rates: Sequence[float],
                bad_fixed: int, rate_fixed: float,
                full_grid: bool = False) -> List[Dict[str, Any]]:
    """要跑的 (N_m, ρ_p) 组合。

    默认十字扫描；``full_grid=True`` 才是全因子。交叉点只出现一次 ——
    重复跑同一个配置除了浪费机时，还会在并表时变成两个"独立"观测。
    """
    if full_grid:
        return [{"bad_num": int(b), "poison_rate": float(p)}
                for b in bad_nums for p in poison_rates]

    points: List[Dict[str, Any]] = []
    seen = set()
    for bad in bad_nums:
        key = (int(bad), float(rate_fixed))
        if key not in seen:
            seen.add(key)
            points.append({"bad_num": key[0], "poison_rate": key[1]})
    for rate in poison_rates:
        key = (int(bad_fixed), float(rate))
        if key not in seen:
            seen.add(key)
            points.append({"bad_num": key[0], "poison_rate": key[1]})
    return points


def _tag(prefix: str, **parts: Any) -> str:
    pieces = [prefix]
    for name, value in parts.items():
        text = str(value).replace(".", "p").replace("-", "m")
        pieces.append(f"{name}{text}")
    return "_".join(pieces)


def _base_command(cfg_exp1: Cfg, seed: int, alpha: float,
                  instrument_root: str, results_dir: str,
                  ckpt_root: str) -> List[str]:
    return [
        "python", "-m", "diag.run_fl",
        "--mode", "attack",
        "--alpha", str(alpha),
        "--seed", str(int(seed)),
        "--defense", "fedavg",
        "--client-num", str(int(cfg_exp1.client_num)),
        "--select-per-round", str(int(cfg_exp1.select_per_round)),
        "--local-steps", str(int(cfg_exp1.local_steps)),
        "--model-size", str(int(cfg_exp1.model_size)),
        "--total-round", str(int(cfg_exp1.total_round)),
        "--eval-every", str(int(cfg_exp1.eval_every)),
        "--eval-include-malicious",
        "--layer-metrics",
        "--instrument-dir", instrument_root,
        "--results-dir", results_dir,
        "--ckpt-root", ckpt_root,
    ]


def build_commands(cfg: Cfg, stage: str = "all", *,
                   instrument_root: str = "instrumentation",
                   results_dir: str = "results/raw",
                   ckpt_root: str = "./checkpoints",
                   full_grid: bool = False,
                   seeds: Optional[Sequence[int]] = None
                   ) -> List[Dict[str, Any]]:
    """返回 ``[{"tag", "stage", "cmd"}]``。``stage`` ∈ {1, 1b, all}。"""
    exp1 = cfg.exp1
    alpha = float(exp1.alpha)
    seeds = [int(s) for s in (seeds if seeds is not None else exp1.seeds)]
    jobs: List[Dict[str, Any]] = []

    if stage in ("1", "all"):
        points = dose_points(exp1.bad_nums, exp1.poison_rates,
                             int(exp1.bad_num_fixed),
                             float(exp1.poison_rate_fixed), full_grid)
        for point in points:
            for seed in seeds:
                tag = _tag("e1", bad=point["bad_num"],
                           rho=point["poison_rate"], s=seed)
                cmd = _base_command(exp1, seed, alpha, instrument_root,
                                    results_dir, ckpt_root)
                cmd += ["--bad-client-num", str(point["bad_num"]),
                        "--poison-rate", str(point["poison_rate"]),
                        "--run-tag", tag]
                jobs.append({"tag": tag, "stage": "1", "cmd": cmd})

    if stage in ("1b", "all"):
        # 1B 固定剂量，只变时间结构 —— 剂量和时间同时变就分不清是谁的作用
        bad = int(exp1.bad_num_fixed)
        rate = float(exp1.poison_rate_fixed)
        for spec in exp1.schedules:
            spec = dict(spec)
            kind = str(spec.pop("kind"))
            # 先构造一次，参数不合法就地报错，而不是等集群上跑到才发现
            schedule = AttackSchedule(kind=kind, **spec)
            for seed in seeds:
                tag = _tag("e1b", sched=kind, s=seed)
                cmd = _base_command(exp1, seed, alpha, instrument_root,
                                    results_dir, ckpt_root)
                cmd += ["--bad-client-num", str(bad),
                        "--poison-rate", str(rate),
                        "--attack-schedule", kind,
                        "--run-tag", tag]
                for name, value in spec.items():
                    cmd += [f"--attack-{name.replace('_', '-')}", str(value)]
                jobs.append({"tag": tag, "stage": "1b", "cmd": cmd,
                             "describe": schedule.describe()})
    return jobs


def estimate_cost(jobs: Sequence[Dict[str, Any]], cfg: Cfg) -> Dict[str, Any]:
    """把机时说清楚 —— 这是决定要不要缩规模的唯一依据。"""
    exp1 = cfg.exp1
    rounds = int(exp1.total_round)
    per_round = int(exp1.select_per_round) * int(exp1.local_steps)
    evals = rounds // max(int(exp1.eval_every), 1)
    return {
        "n_runs": len(jobs),
        "rounds_per_run": rounds,
        "local_batches_per_run": rounds * per_round,
        "evaluations_per_run": evals,
        "total_local_batches": len(jobs) * rounds * per_round,
        "asr_mta_points_per_run": evals,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="实验 1 / 1B 的命令生成器（默认 dry-run）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all", choices=["1", "1b", "all"])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--full-grid", action="store_true",
                        help="全因子而不是十字扫描。第一阶段不要用 —— "
                             "4×4×2=32 个 run")
    parser.add_argument("--instrument-root", default="instrumentation")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--execute", action="store_true",
                        help="真正执行；缺省只打印")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    jobs = build_commands(cfg, args.stage,
                          instrument_root=args.instrument_root,
                          results_dir=args.results_dir,
                          ckpt_root=args.ckpt_root,
                          full_grid=args.full_grid, seeds=args.seeds)

    cost = estimate_cost(jobs, cfg)
    print(f"=== 实验 1 / 1B：{cost['n_runs']} 个 run ===")
    print(f"  每个 run {cost['rounds_per_run']} 轮 × "
          f"{cfg.exp1.select_per_round}×{cfg.exp1.local_steps} 本地 batch "
          f"= {cost['local_batches_per_run']:,} 个 batch")
    print(f"  合计约 {cost['total_local_batches']:,} 个本地 batch")
    print(f"  每个 run 有 {cost['asr_mta_points_per_run']} 个 (MTA, ASR) 配对 —— "
          f"ASR-vs-MTA 散点图的点数")
    if cost["asr_mta_points_per_run"] < 20:
        print(f"  ⚠️ 每个 run 只有 {cost['asr_mta_points_per_run']} 个配对，"
              f"散点图上看不出转折。把 exp1.eval_every 调小。")

    print()
    for job in jobs:
        note = f"   # {job['describe']}" if "describe" in job else ""
        print(" ".join(job["cmd"]) + note)

    if not args.execute:
        print(f"\n[dry-run] 以上 {len(jobs)} 条命令**未执行**。加 --execute 才真跑。")
        return 0

    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {job['tag']}")
        result = subprocess.run(job["cmd"])
        if result.returncode != 0:
            # 不继续跑剩下的：后面的分析会把缺失的 run 当成"没测过"，
            # 而实际是"跑挂了"，两者在图上无法区分
            print(f"[run_exp1] {job['tag']} 失败（returncode="
                  f"{result.returncode}），中止")
            return result.returncode
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
