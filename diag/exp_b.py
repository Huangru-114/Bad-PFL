"""实验 B：Table 24 的跨客户端版本。

原论文附录 Table 24（在单个干净模型上）：

    目标类样本 + ξ       -> 目标类准确率 ≈ 6.70%
    目标类样本 + δ + ξ   -> 目标类准确率 ≈ 39.10%

即 ξ 掩盖了真实类特征，而 δ 把模型重新拉回目标类 —— 证明 δ 在模型看来
就是目标类的 ground-truth 特征。

本脚本把它扩展到每个客户端::

    Acc_xi^(k)   = 目标类样本加 ξ 后的准确率
    Acc_dxi^(k)  = 目标类样本加 δ+ξ 后的准确率
    Recovery_k   = Acc_dxi^(k) - Acc_xi^(k)

CSV 的第一行是 ``client_id = -1`` 的**全局干净模型**，用于先复现原文数字
（允许 ±5 个点偏差，但趋势必须一致）。对不上就先排查，不要继续。

⚠️ 与原 spec 的差异：``min_target_samples`` 过滤已删除。改用共享探针集后
所有客户端都在**同一批** 400 张目标类图片上测量，因此都能算出有效指标 ——
包括本地一张目标类图片都没有的客户端。本地目标类样本数改为协变量
``n_target_samples_local``。

用法::

    python -m diag.exp_b --ckpt-dir checkpoints/clean_a0.5_s0 \\
                         --gen-ckpt-dir checkpoints/attack_a0.5_s0
    python -m diag.exp_b --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import load_config
from .exp_common import add_common_args, common_row, load_bundle, write_csv
from .hooks import load_checkpoint
from .metrics import recovery_score

__all__ = ["run_exp_b", "main"]


def run_exp_b(bundle, generator=None) -> pd.DataFrame:
    loaders = bundle.probe_loaders()
    target_loader = loaders["target"]
    delta_fn = bundle.delta_fn(generator)
    eps = float(bundle.cfg.perturb.eps_delta)

    def _score(model, base_row):
        xi_fn = bundle.xi_fn(model)   # ξ 由被评估的模型生成（fba.py:53 语义）
        result = recovery_score(model, target_loader, delta_fn, xi_fn,
                                device=bundle.device,
                                target_class=bundle.target_class, eps=eps,
                                pixel_min=float(bundle.cfg.perturb.pixel_min),
                                pixel_max=float(bundle.cfg.perturb.pixel_max))
        return {**base_row, **result}

    rows: List[dict] = []

    # --- 全局干净模型：先复现原文的 6.70% / 39.10% ---
    global_row = {
        "client_id": -1, "is_malicious": False, "is_malicious_slot": False,
        "poison_ratio": 0.0, "n_local_samples": -1,
        "n_target_samples_local": -1, "n_participations": -1,
        "alpha": float(bundle.meta["alpha"]), "seed": int(bundle.meta["seed"]),
        "mode": str(bundle.meta["mode"]), "scope": "global",
    }
    rows.append(_score(bundle.global_model(), global_row))

    # --- 每个客户端 ---
    for record in bundle.client_records():
        base_row = {**common_row(bundle, record), "scope": "client"}
        rows.append(_score(bundle.client_model(record["client_id"]), base_row))

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description="实验 B：Table 24 的跨客户端版本"))
    parser.add_argument("--gen-ckpt-dir", default=None,
                        help="提供生成器的 attack run 目录")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bundle = load_bundle(args, cfg)

    generator = None
    if args.gen_ckpt_dir:
        import torch

        from generator import Autoencoder
        generator = Autoencoder()
        generator.load_state_dict(
            load_checkpoint(Path(args.gen_ckpt_dir) / "generator.pt"), strict=True)
        generator.to(bundle.device).eval()
    elif not bundle.smoke:
        raise ValueError("非冒烟模式下必须提供 --gen-ckpt-dir")

    frame = run_exp_b(bundle, generator)
    default = (f"expB_{bundle.meta['mode']}_a{bundle.meta['alpha']}"
               f"_s{bundle.meta['seed']}.csv")
    write_csv(frame, args.out, default, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
