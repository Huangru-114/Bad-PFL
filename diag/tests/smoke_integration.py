"""集成冒烟测试：跑通 训练 -> 校验 -> 四个实验 -> 汇总绘图 的完整链路。

⚠️ **本脚本产出的所有数字没有任何科学含义。** 4 个客户端、2 轮训练、合成数据
得到的模型不代表任何东西。唯一的目的是证明：代码不崩、接口对得上、
维度算得对、CSV 列完整、图能生成、对照组划分一致。

通过标准（全部为硬性）：
1. clean run 与 attack run 都能跑完并落盘
2. ``verify_partition_consistency`` 返回 True
3. exp_a / exp_b / exp_c / exp_d 全部生成 CSV，列完整、行数正确
4. 全部图文件生成且非空

用法::

    python -m diag.tests.smoke_integration [--workdir DIR] [--keep]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd

from diag.config import load_config
from diag.exp_common import COMMON_COLUMNS

STEPS: List[str] = []


def _step(message: str) -> None:
    STEPS.append(message)
    print(f"\n=== [{len(STEPS)}] {message} ===")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"    ok: {message}")


def run(workdir: Path) -> int:
    from diag.analysis import (plot_a1, plot_a2, plot_a3, plot_a4, plot_c1,
                               plot_c2, plot_c3, plot_d1, plot_d2, summarize,
                               summarize_c, summarize_d)
    from diag.exp_a import run_exp_a
    from diag.exp_b import run_exp_b
    from diag.exp_c import run_exp_c
    from diag.exp_d import run_exp_d
    from diag.exp_common import load_bundle
    from diag.hooks import verify_partition_consistency
    from diag.run_fl import run_fl

    cfg = load_config()
    ckpt_root = workdir / "checkpoints"
    raw_dir = workdir / "results" / "raw"
    figs_dir = workdir / "results" / "figs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(cfg.smoke.alpha)
    seed = int(cfg.smoke.seed)
    n_clients = int(cfg.smoke.num_clients)
    num_classes = int(cfg.smoke.num_classes)

    # --- 1 & 2. 两次极小规模训练 ----------------------------------------
    _step("跑一次极小的 clean run")
    clean_dir = run_fl(cfg, "clean", alpha, seed, smoke=True, ckpt_root=ckpt_root)
    _check((clean_dir / "global.pt").exists(), "clean run 保存了 global.pt")
    _check(not (clean_dir / "generator.pt").exists(),
           "clean run 没有生成器（符合预期：δ 只存在于攻击 run）")

    _step("用同样设置跑一次极小的 attack run")
    attack_dir = run_fl(cfg, "attack", alpha, seed, smoke=True, ckpt_root=ckpt_root)
    _check((attack_dir / "generator.pt").exists(), "attack run 保存了 generator.pt")
    _check((attack_dir / "delta.pt").exists(),
           "attack run 保存了 delta.pt（仅存档，不参与计算）")

    # --- 3. 对照组一致性（硬性要求） ------------------------------------
    _step("校验两次 run 的数据划分逐客户端一致")
    ok, report = verify_partition_consistency(clean_dir / "meta.json",
                                              attack_dir / "meta.json")
    print(report)
    _check(ok, "verify_partition_consistency 返回 True")

    # --- 4. 四个实验 ------------------------------------------------------
    class Args:
        config = None
        device = "cpu"
        smoke = True
        out = None
        ckpt_dir = None

    def _bundle(ckpt_dir):
        args = Args()
        args.ckpt_dir = str(ckpt_dir)
        return load_bundle(args, cfg)

    import torch

    from generator import Autoencoder
    generator = Autoencoder()
    generator.load_state_dict(torch.load(attack_dir / "generator.pt",
                                         map_location="cpu"), strict=True)
    generator.eval()

    _step("exp_a --smoke")
    frame_a = run_exp_a(_bundle(clean_dir), generator)
    path_a = raw_dir / f"expA_clean_a{alpha}_s{seed}.csv"
    frame_a.to_csv(path_a, index=False)
    n_groups = frame_a["group"].nunique()
    _check(set(COMMON_COLUMNS).issubset(frame_a.columns), "exp_a CSV 公共列完整")
    _check(len(frame_a) == n_clients * n_groups,
           f"exp_a 行数 = 客户端数 × 组数 = {n_clients} × {n_groups}")
    _check({"overlap", "nat_rate"}.issubset(frame_a.columns), "exp_a 指标列存在")

    _step("exp_b --smoke")
    frame_b = run_exp_b(_bundle(clean_dir), generator)
    path_b = raw_dir / f"expB_clean_a{alpha}_s{seed}.csv"
    frame_b.to_csv(path_b, index=False)
    _check(set(COMMON_COLUMNS).issubset(frame_b.columns), "exp_b CSV 公共列完整")
    _check(len(frame_b) == n_clients + 1,
           "exp_b 行数 = 客户端数 + 1（全局模型那一行）")
    _check({"acc_xi", "acc_dxi", "recovery"}.issubset(frame_b.columns),
           "exp_b 指标列存在")

    _step("exp_c --smoke")
    frame_c = run_exp_c(_bundle(attack_dir), _bundle(clean_dir), generator)
    path_c = raw_dir / f"expC_attack_a{alpha}_s{seed}.csv"
    frame_c.to_csv(path_c, index=False)
    _check(set(COMMON_COLUMNS).issubset(frame_c.columns), "exp_c CSV 公共列完整")
    _check(len(frame_c) == n_clients, "exp_c 行数 = 客户端数")
    _check({"A_observable", "E", "l2_to_median", "sign_consistency"}
           .issubset(frame_c.columns), "exp_c 指标列存在")
    _check(frame_c["E"].notna().all(), "exp_c 的 ground truth E 全部算出")

    _step("exp_d --smoke")
    frame_d = run_exp_d(_bundle(attack_dir))
    path_d = raw_dir / f"expD_attack_a{alpha}_s{seed}.csv"
    frame_d.to_csv(path_d, index=False)
    _check(set(COMMON_COLUMNS).issubset(frame_d.columns), "exp_d CSV 公共列完整")
    _check(len(frame_d) == n_clients * num_classes,
           f"exp_d 行数 = 客户端数 × 类别数 = {n_clients} × {num_classes}")
    _check({"J", "v"}.issubset(frame_d.columns), "exp_d 指标列存在")

    # --- 5. 汇总与绘图 ----------------------------------------------------
    _step("analysis: 汇总并生成全部图")
    summary_a = summarize(str(path_a))
    summary_c = summarize_c(str(path_c))
    summary_d = summarize_d(str(path_d))

    summary_dir = workdir / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_a.to_csv(summary_dir / "expA_summary.csv", index=False)
    summary_c.to_csv(summary_dir / "expC_summary.csv", index=False)
    summary_d.to_csv(summary_dir / "expD_summary.csv", index=False)

    figures = [
        plot_a1(summary_a, figs_dir / "expA_A1.png"),
        plot_a2(summary_a, figs_dir / "expA_A2.png"),
        plot_a3(str(path_a), figs_dir / "expA_A3.png"),
        plot_a4(str(path_a), figs_dir / "expA_A4.png"),
        plot_c1(str(path_c), figs_dir / "expC_C1.png"),
        plot_c2(summary_c, figs_dir / "expC_C2.png"),
        plot_c3(str(path_c), figs_dir / "expC_C3.png"),
        plot_d1(str(path_d), figs_dir / "expD_D1.png"),
        plot_d2(summary_d, figs_dir / "expD_D2.png"),
    ]
    for figure in figures:
        _check(figure.exists() and figure.stat().st_size > 1000,
               f"{figure.name} 已生成且非空")

    print("\n" + "=" * 62)
    print("集成冒烟测试通过。以上所有数值均无科学含义，仅证明代码可运行。")
    print("=" * 62)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="集成冒烟测试")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep", action="store_true", help="保留工作目录")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="diag_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"工作目录: {workdir}")

    try:
        return run(workdir)
    except Exception:
        traceback.print_exc()
        print(f"\n集成冒烟测试失败（在第 {len(STEPS)} 步之后）")
        return 1
    finally:
        if not args.keep and args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
