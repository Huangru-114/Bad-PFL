"""恶意客户端密度扫描的命令生成器（**默认 dry-run**）。

# 为什么要扫这个

两条都指向同一个设定：

1. **论文自己的消融**把恶意客户端降到 1 个仍能植入后门。我们的实测显示
   Multi-Krum 以 76% 的检出率仍拦不住（漏过 124 次，而 1 个客户端的
   ~50 次就够）。降低密度能直接检验这条推理链在本复现里成立与否。
2. **ΔT 衰减曲线（实验 J §5.2 / 图 J-4）在当前密度下画不出来**。
   10 个恶意 / 100 客户端 / 每轮选 10 时，``rounds_since_last_malicious``
   实测取值只有 {0: 33, 1: 14, 2: 2, 3: 1} —— 最大间隔 3 轮，没有动态范围。
   密度降下来才谈得上"稀释"。

# 间隔的期望值

每轮恰好 0 个恶意的概率服从超几何分布；连续无攻击的轮数近似几何分布，
期望间隔 ≈ ``1 / (1 - P(0 个恶意)) − 1``。本模块的 ``expected_gap``
会把每个密度下的这个数算出来，**跑之前就能知道 J-4 有没有动态范围**。

# 目录不会互相覆盖

每个密度用 ``--run-tag bad{N}``，checkpoint 落在
``attack_a0.5_s0_bad{N}``，逐轮 npz 落在 ``instrumentation/{defense}_attack_a0.5_s0_bad{N}``。
**不加 tag 会静默覆盖**，所以本模块总是加。

用法::

    python -m diag.run_density_sweep                     # 打印命令，不执行
    python -m diag.run_density_sweep --bad-nums 1 2 5 10
    python -m diag.run_density_sweep --defenses fedavg invariant
    python -m diag.run_density_sweep --execute           # 真正跑
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

from .config import load_config

__all__ = ["expected_gap", "build_commands", "main"]


def expected_gap(client_num: int, bad_client_num: int, select_per_round: int
                 ) -> Dict[str, float]:
    """每轮恰好 0 个恶意的概率，以及"距上次有恶意"的期望间隔。

    ``rounds_since_last_malicious`` 是 J-4 的横轴；若期望间隔接近 0，
    那张图就没有动态范围，跑了也画不出衰减。
    """
    from scipy.stats import hypergeom

    p_zero = float(hypergeom(client_num, bad_client_num, select_per_round).pmf(0))
    return {
        "p_no_malicious": p_zero,
        # 连续 k 轮无攻击的概率 p_zero^k -> 期望连续无攻击轮数 p/(1-p)
        "expected_gap": p_zero / (1.0 - p_zero) if p_zero < 1.0 else float("inf"),
        "expected_malicious_per_round": select_per_round * bad_client_num / client_num,
    }


def build_commands(bad_nums: Sequence[int], defenses: Sequence[str], *,
                   alpha: float, seed: int, total_round: Optional[int],
                   eval_every: int, ckpt_root: str, instrument_dir: str,
                   results_dir: Optional[str]) -> List[List[str]]:
    commands: List[List[str]] = []
    for bad in bad_nums:
        tag = f"bad{int(bad)}"
        for defense in defenses:
            cmd = [sys.executable, "-m", "diag.run_fl",
                   "--mode", "attack", "--alpha", str(alpha), "--seed", str(seed),
                   "--defense", defense,
                   "--bad-client-num", str(int(bad)),
                   "--run-tag", tag,
                   "--eval-every", str(eval_every),
                   "--instrument-dir", instrument_dir,
                   "--ckpt-root", ckpt_root]
            if total_round is not None:
                cmd += ["--total-round", str(int(total_round))]
            if results_dir:
                cmd += ["--results-dir", results_dir]
            commands.append(cmd)
            # 汇总紧跟其后，免得回头找不到是哪个 run
            commands.append([
                sys.executable, "-m", "diag.exp_ij",
                "--instrument-dir",
                f"{instrument_dir}/{defense}_attack_a{alpha}_s{seed}_{tag}",
                "--implantation-glob",
                f"{results_dir or 'results/raw'}/"
                f"exp_ij_implantation_{defense}_attack_a{alpha}_s{seed}_{tag}.csv",
                "--out-prefix", f"results/exp_ij_{defense}_{tag}",
                "--alpha-dirichlet", str(alpha), "--seed", str(seed)])
    return commands


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="恶意客户端密度扫描（默认 dry-run）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bad-nums", type=int, nargs="+", default=[1, 2, 5, 10],
                        help="要扫的恶意客户端数")
    parser.add_argument("--defenses", nargs="+", default=["fedavg", "invariant"],
                        help="要跑的防御；扫密度时建议至少含 fedavg 作对照")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-round", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--instrument-dir", default="instrumentation")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--execute", action="store_true",
                        help="真正执行；缺省只打印")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    client_num = int(cfg.fl.client_num)
    select = int(cfg.fl.select_per_round)

    print("=== 各密度下的攻击间隔（决定图 J-4 有没有动态范围）===")
    print(f"{'恶意数':>6} {'每轮期望恶意数':>14} {'P(0 个恶意)':>12} {'期望连续无攻击轮数':>18}")
    for bad in args.bad_nums:
        stats = expected_gap(client_num, bad, select)
        print(f"{bad:>6} {stats['expected_malicious_per_round']:>14.2f}"
              f" {stats['p_no_malicious']:>12.4f} {stats['expected_gap']:>18.2f}")
    print(f"\n（当前 config：{client_num} 客户端，每轮选 {select} 个）")
    print("实测参考：10 个恶意时 rounds_since_last_malicious 只取到 {0,1,2,3}，"
          "J-4 画不出衰减")

    commands = build_commands(
        args.bad_nums, args.defenses, alpha=args.alpha, seed=args.seed,
        total_round=args.total_round, eval_every=args.eval_every,
        ckpt_root=args.ckpt_root, instrument_dir=args.instrument_dir,
        results_dir=args.results_dir)

    print(f"\n=== {len(commands)} 条命令"
          f"（{len(args.bad_nums)} 密度 × {len(args.defenses)} 防御 × 训练+汇总）===")
    for cmd in commands:
        print("  " + " ".join(shlex.quote(part) for part in cmd))

    if not args.execute:
        print("\n默认 dry-run。确认无误后加 --execute 执行，"
              "或把上面的命令贴进作业脚本。")
        return 0

    for index, cmd in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] " +
              " ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ 第 {index} 条失败（退出码 {result.returncode}），中止。")
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
