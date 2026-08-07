"""实验矩阵启动器：生成 4 alpha × 3 seed × 2 mode = 24 次训练的命令。

**默认 dry-run，只打印命令不执行。** 真正执行需要显式 ``--execute``。

生成的命令分三段：
1. **训练**（24 条）：每条产出一个 checkpoint 目录。
2. **实验**：exp_a / exp_b 用 clean run 的模型 + attack run 的生成器；
   exp_c 需要 clean + attack 两个目录；exp_d 用 attack run。
3. **汇总与绘图**。

用法::

    python -m diag.run_matrix                 # 打印全部命令（dry-run）
    python -m diag.run_matrix --stage train   # 只看训练命令
    python -m diag.run_matrix --execute       # 真正执行（本次任务不要用）
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .hooks import run_dir_name

__all__ = ["build_commands", "main"]


def build_commands(cfg, stage: str = "all") -> List[List[str]]:
    """构造命令列表。每个元素是可直接交给 ``subprocess`` 的 argv。"""
    alphas = list(cfg.sweep.alpha)
    seeds = list(cfg.sweep.seed)
    modes = list(cfg.sweep.mode)
    ckpt_root = Path(cfg.paths.ckpt_root)
    results_root = Path(cfg.paths.results_root)
    raw_dir = results_root / "raw"
    figs_dir = results_root / "figs"
    python = sys.executable or "python"

    commands: List[List[str]] = []

    if stage in ("all", "train"):
        for alpha in alphas:
            for seed in seeds:
                for mode in modes:
                    commands.append([
                        python, "-m", "diag.run_fl", "--mode", str(mode),
                        "--alpha", str(alpha), "--seed", str(seed),
                        "--ckpt-root", str(ckpt_root)])

    if stage in ("all", "verify"):
        for alpha in alphas:
            for seed in seeds:
                clean = ckpt_root / run_dir_name("clean", alpha, seed)
                attack = ckpt_root / run_dir_name("attack", alpha, seed)
                commands.append([
                    python, "-c",
                    ("from diag.hooks import verify_partition_consistency as v;"
                     f"ok,rep=v(r'{clean}/meta.json', r'{attack}/meta.json');"
                     "print(rep);"
                     "raise SystemExit(0 if ok else 1)")])

    if stage in ("all", "exp"):
        for alpha in alphas:
            for seed in seeds:
                clean = ckpt_root / run_dir_name("clean", alpha, seed)
                attack = ckpt_root / run_dir_name("attack", alpha, seed)
                tag = f"a{alpha}_s{seed}"
                commands.append([
                    python, "-m", "diag.exp_a", "--ckpt-dir", str(clean),
                    "--gen-ckpt-dir", str(attack),
                    "--out", str(raw_dir / f"expA_{tag}.csv")])
                commands.append([
                    python, "-m", "diag.exp_b", "--ckpt-dir", str(clean),
                    "--gen-ckpt-dir", str(attack),
                    "--out", str(raw_dir / f"expB_{tag}.csv")])
                commands.append([
                    python, "-m", "diag.exp_c", "--ckpt-dir", str(attack),
                    "--clean-ckpt-dir", str(clean),
                    "--out", str(raw_dir / f"expC_{tag}.csv")])
                commands.append([
                    python, "-m", "diag.exp_d", "--ckpt-dir", str(attack),
                    "--out", str(raw_dir / f"expD_{tag}.csv")])

    if stage in ("all", "analysis"):
        commands.append([
            python, "-c",
            ("from diag.analysis import *;"
             f"sa=summarize(r'{raw_dir}/expA_*.csv');"
             f"sa.to_csv(r'{results_root}/summary/expA_summary.csv', index=False);"
             f"plot_a1(sa, r'{figs_dir}/expA_A1.png');"
             f"plot_a2(sa, r'{figs_dir}/expA_A2.png');"
             f"plot_a3(r'{raw_dir}/expA_*.csv', r'{figs_dir}/expA_A3.png');"
             f"plot_a4(r'{raw_dir}/expA_*.csv', r'{figs_dir}/expA_A4.png');"
             f"sc=summarize_c(r'{raw_dir}/expC_*.csv');"
             f"sc.to_csv(r'{results_root}/summary/expC_summary.csv', index=False);"
             f"plot_c1(r'{raw_dir}/expC_*.csv', r'{figs_dir}/expC_C1.png');"
             f"plot_c2(sc, r'{figs_dir}/expC_C2.png');"
             f"plot_c3(r'{raw_dir}/expC_*.csv', r'{figs_dir}/expC_C3.png');"
             f"sd=summarize_d(r'{raw_dir}/expD_*.csv');"
             f"sd.to_csv(r'{results_root}/summary/expD_summary.csv', index=False);"
             f"plot_d1(r'{raw_dir}/expD_*.csv', r'{figs_dir}/expD_D1.png');"
             f"plot_d2(sd, r'{figs_dir}/expD_D2.png');"
             "print('analysis done')")])

    return commands


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="诊断实验矩阵启动器（默认 dry-run）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default="all",
                        choices=["all", "train", "verify", "exp", "analysis"])
    parser.add_argument("--execute", action="store_true",
                        help="真正执行命令。默认只打印。")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    commands = build_commands(cfg, args.stage)

    print(f"# 共 {len(commands)} 条命令 (stage={args.stage}, "
          f"execute={args.execute})")
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))

    if not args.execute:
        print("\n# dry-run：未执行任何命令。加 --execute 才会真正运行。")
        return 0

    for index, command in enumerate(commands, start=1):
        print(f"\n# [{index}/{len(commands)}] 执行中 ...")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"# 命令失败（返回码 {result.returncode}），中止。")
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
