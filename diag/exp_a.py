"""实验 A：δ 的跨客户端自然度离散（验证 H1，⭐ 生死判据）。

在**干净 run**（无任何投毒）的每个客户端本地模型上，测量四组输入的
"目标类归属度"：

| 组               | 输入                          | 角色 | 预期 |
|------------------|-------------------------------|------|------|
| ``real``         | 真实目标类图片(probe.target)  | 正对照 | 跨客户端高度一致 |
| ``delta_only``   | 非目标类图片(probe.query) + δ | 待测 | ? |
| ``delta_plus_xi``| 非目标类图片 + δ + ξ          | 待测 | ? |
| ``random_noise`` | 非目标类图片 + 同预算随机噪声 | 负对照 | 跨客户端一致地低 |

⚠️ **必须用干净 run 的模型**。用攻击 run 的模型测会混入后门本身的效应，
逻辑上是循环论证。δ 则必须来自**攻击 run** 的生成器（clean run 没有生成器），
所以本脚本需要两个目录：``--ckpt-dir``（clean）与 ``--gen-ckpt-dir``（attack）。

⚠️ 由于 δ 确实携带真实语义（原论文 Table 24 已证明），单独测量 δ 的跨客户端
一致性没有意义。**核心量是 δ 与真实目标类图片的一致性之差**
（自然度差距 NG），由 ``diag.analysis.summarize`` 计算。

用法::

    python -m diag.exp_a --ckpt-dir checkpoints/clean_a0.5_s0 \\
                         --gen-ckpt-dir checkpoints/attack_a0.5_s0 \\
                         --out results/raw/expA_a0.5_s0.csv
    python -m diag.exp_a --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import load_config
from .exp_common import (add_common_args, common_row, load_bundle, resolve_device,
                         write_csv)
from .hooks import load_checkpoint
from .metrics import NATURALNESS_GROUPS, naturalness

__all__ = ["run_exp_a", "main"]


def run_exp_a(bundle, generator=None) -> pd.DataFrame:
    """返回每 (客户端 × 组) 一行的 DataFrame。"""
    loaders = bundle.probe_loaders()
    delta_fn = bundle.delta_fn(generator)
    k = bundle.knn_k()
    eps = float(bundle.cfg.perturb.eps_delta)
    noise_seed = int(bundle.cfg.perturb.noise_seed)

    rows: List[dict] = []
    for record in bundle.client_records():
        model = bundle.client_model(record["client_id"])
        # ξ 由被评估的那个模型生成 —— 保持 fba.py:53 的原有语义。
        xi_fn = bundle.xi_fn(model)
        result = naturalness(model, loaders, delta_fn, xi_fn,
                             bundle.target_class, device=bundle.device, k=k,
                             eps=eps, noise_seed=noise_seed,
                             pixel_min=float(bundle.cfg.perturb.pixel_min),
                             pixel_max=float(bundle.cfg.perturb.pixel_max),
                             groups=NATURALNESS_GROUPS)
        base_row = common_row(bundle, record)
        for group, values in result.items():
            rows.append({**base_row,
                         "group": group,
                         "overlap": values["overlap"],
                         "nat_rate": values["nat_rate"],
                         "n_probe_samples": values["n_samples"],
                         "knn_k": k,
                         "xi_source": "own_model"})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description="实验 A：δ 的跨客户端自然度离散"))
    parser.add_argument("--gen-ckpt-dir", default=None,
                        help="提供生成器的 attack run 目录（δ 只存在于攻击 run）")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bundle = load_bundle(args, cfg)

    if str(bundle.meta.get("mode")) == "attack":
        print("[warn] --ckpt-dir 指向的是 attack run。实验 A 要求用【干净 run】"
              "的模型，否则会混入后门本身的效应（循环论证）。")

    generator = None
    if args.gen_ckpt_dir:
        from generator import Autoencoder
        import torch
        generator = Autoencoder()
        generator.load_state_dict(
            load_checkpoint(Path(args.gen_ckpt_dir) / "generator.pt"), strict=True)
        generator.to(bundle.device).eval()
    elif not bundle.smoke:
        raise ValueError(
            "非冒烟模式下必须提供 --gen-ckpt-dir：δ 来自攻击 run 的生成器，"
            "而 --ckpt-dir 指向的干净 run 不含生成器。")
    # smoke 且未提供时使用随机初始化的生成器（bundle.generator() 会处理）。

    frame = run_exp_a(bundle, generator)
    default = (f"expA_{bundle.meta['mode']}_a{bundle.meta['alpha']}"
               f"_s{bundle.meta['seed']}.csv")
    write_csv(frame, args.out, default, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
