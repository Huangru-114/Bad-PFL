"""实验 C：超额响应与可观测量的相关性（验证 H3）。

H3：防御方在**不知道 δ 的情况下**，能否通过模型的决策几何近似出后门强度？

# 两侧的量

**ground truth（防御方看不到）** —— 需要干净模型 + δ::

    nat_rate_k = 在客户端 k 的【干净】本地模型上，非目标类样本加 δ+ξ 被判为目标类的比例
    asr_k      = 在客户端 k 的【攻击 run】本地模型上，同样输入的同样比例
    E_k        = asr_k - nat_rate_k

**可观测量（防御方能算）** —— 只用攻击 run 客户端上传的类原型 + 分类头::

    M^(k)      = margin_matrix(proto_k, W_k, b_k)
    A^(k)      = observable_score([M^(1..K)], lambda_std)

⚠️ **类原型用客户端的本地训练数据算，不用探针集。** 这是刻意为之：
``observable_score`` 模拟的是真实部署中防御方能拿到的东西 —— 客户端在自己的
本地数据上算原型然后上传。用探针集算就等于假设防御方持有一份干净的全局数据，
那是作弊。直接后果是本地数据少的客户端原型噪声大，这是防御方法本身必须面对
的真实困难，应当被如实测量而不是被掩盖。

同时输出两条对照信号（图 C2 的另外两条曲线）：``l2_to_median``（Krum/FLAME 类）
与 ``sign_consistency``（Invariant Aggregator 类）。

用法::

    python -m diag.exp_c --ckpt-dir checkpoints/attack_a0.5_s0 \\
                         --clean-ckpt-dir checkpoints/clean_a0.5_s0
    python -m diag.exp_c --smoke
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .exp_common import add_common_args, common_row, load_bundle, write_csv
from .features import (class_prototypes, extract_penultimate, find_last_linear,
                       margin_matrix)
from .hooks import flatten_state_dict
from .metrics import baseline_signals, excess_response, observable_score

__all__ = ["run_exp_c", "main"]


def run_exp_c(attack_bundle, clean_bundle=None, generator=None) -> pd.DataFrame:
    """返回每客户端一行的 DataFrame。

    ``clean_bundle`` 为 None 时跳过 ground truth（``E_k``）的计算，
    只输出可观测量与对照信号 —— 用于快速检查可观测量侧的流程。
    """
    cfg = attack_bundle.cfg
    records = attack_bundle.client_records()
    min_class_count = int(cfg.metrics.min_class_count)
    lambda_std = float(cfg.metrics.lambda_std)

    # --- 可观测量侧：本地类原型 -> 边距矩阵 -> A^(k) ---------------------
    margins: List[torch.Tensor] = []
    proto_counts: List[np.ndarray] = []
    for record in records:
        model = attack_bundle.client_model(record["client_id"])
        local_loader = attack_bundle.local_train_loader(record)
        out = extract_penultimate(model, local_loader, attack_bundle.device,
                                  source="local")   # ← 唯一使用本地数据的地方
        stats = class_prototypes(out.feats, out.labels, attack_bundle.num_classes,
                                 min_class_count=min_class_count)
        head = find_last_linear(model)
        margins.append(margin_matrix(stats["proto"],
                                     head.weight.detach().cpu(),
                                     head.bias.detach().cpu()))
        proto_counts.append(stats["count"].numpy())

    scores = observable_score(margins, lambda_std=lambda_std)

    # λ 敏感性（图 C3）
    lambda_sweep = list(cfg.metrics.lambda_sweep)
    sweep_scores = {float(lam): observable_score(margins, lambda_std=float(lam))
                    for lam in lambda_sweep}

    # --- 对照信号：客户端更新向量 ---------------------------------------
    global_state = attack_bundle.global_state()
    updates = [flatten_state_dict(attack_bundle.client_state(r["client_id"]),
                                  reference=global_state) for r in records]
    baselines = baseline_signals(updates)

    # --- ground truth 侧：E_k -------------------------------------------
    excess_by_client = {}
    if clean_bundle is not None:
        loaders = clean_bundle.probe_loaders()
        query_loader = loaders["query"]
        delta_fn = clean_bundle.delta_fn(generator)
        eps = float(cfg.perturb.eps_delta)
        clean_ids = {r["client_id"] for r in clean_bundle.client_records()}
        for record in records:
            cid = record["client_id"]
            if cid not in clean_ids:
                continue
            clean_model = clean_bundle.client_model(cid)
            attacked_model = attack_bundle.client_model(cid)
            excess_by_client[cid] = excess_response(
                clean_model, attacked_model, query_loader, delta_fn,
                clean_bundle.xi_fn(clean_model),
                attack_bundle.xi_fn(attacked_model),
                attack_bundle.target_class, device=clean_bundle.device, eps=eps,
                pixel_min=float(cfg.perturb.pixel_min),
                pixel_max=float(cfg.perturb.pixel_max))

    # --- 拼装 ------------------------------------------------------------
    rows: List[dict] = []
    for position, record in enumerate(records):
        cid = record["client_id"]
        row = {**common_row(attack_bundle, record)}
        row["A_observable"] = float(scores[position])
        for lam, values in sweep_scores.items():
            row[f"A_lambda_{lam}"] = float(values[position])
        row["l2_to_median"] = float(baselines.loc[position, "l2_to_median"])
        row["sign_consistency"] = float(baselines.loc[position, "sign_consistency"])
        row["n_valid_prototypes"] = int(
            np.sum(proto_counts[position] >= min_class_count))

        excess = excess_by_client.get(cid)
        row["nat_rate"] = float(excess["nat_rate"]) if excess else float("nan")
        row["asr"] = float(excess["asr"]) if excess else float("nan")
        row["E"] = float(excess["E"]) if excess else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = add_common_args(
        argparse.ArgumentParser(description="实验 C：超额响应与可观测量"))
    parser.add_argument("--clean-ckpt-dir", default=None,
                        help="对应的 clean run 目录（计算 ground truth E_k）")
    parser.add_argument("--gen-ckpt-dir", default=None,
                        help="提供生成器的 attack run 目录；默认用 --ckpt-dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    attack_bundle = load_bundle(args, cfg)

    clean_bundle = None
    if args.clean_ckpt_dir:
        clean_args = copy.copy(args)
        clean_args.ckpt_dir = args.clean_ckpt_dir
        clean_bundle = load_bundle(clean_args, cfg)
    elif not args.smoke:
        print("[warn] 未提供 --clean-ckpt-dir，将跳过 ground truth E_k 的计算。")

    generator = None
    gen_dir = args.gen_ckpt_dir or args.ckpt_dir
    if gen_dir and (Path(gen_dir) / "generator.pt").exists():
        from generator import Autoencoder
        generator = Autoencoder()
        generator.load_state_dict(
            torch.load(Path(gen_dir) / "generator.pt", map_location="cpu"),
            strict=True)
        generator.to(attack_bundle.device).eval()

    frame = run_exp_c(attack_bundle, clean_bundle, generator)
    default = (f"expC_{attack_bundle.meta['mode']}_a{attack_bundle.meta['alpha']}"
               f"_s{attack_bundle.meta['seed']}.csv")
    write_csv(frame, args.out, default, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
