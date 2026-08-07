"""实验 D：源类表征混乱（验证 H2）。

假设 H2：ξ 破坏源类特征，会在恶意客户端的模型上留下"源类表征混乱"的痕迹
（类内散度异常增大），该痕迹在良性客户端上不存在。

流程（两遍）：
1. 第一遍：对**攻击 run** 的每个客户端模型，在共享探针集上算每个类的
   Fisher 型归一化散度 ``J_c``。
2. 计算基线：``benign_baseline[c] = median_{j ∈ 良性} J_c^(j)``。
3. 第二遍：回填 ``v_c^(k) = log(J_c^(k) / benign_baseline[c])``。

⚠️ 稀释效应：恶意客户端通常只投毒一部分数据（``poison_ratio``，默认 0.2），
其干净数据仍在教正确的源类特征，信号会被稀释。``poison_ratio`` 已写入 CSV，
便于后续做投毒比例扫描并找出信号消失的阈值。

用法::

    python -m diag.exp_d --ckpt-dir checkpoints/attack_a0.5_s0 --out results/raw/expD_a0.5_s0.csv
    python -m diag.exp_d --smoke
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .exp_common import (add_common_args, common_row, load_bundle, write_csv)
from .metrics import dispersion_scores

__all__ = ["run_exp_d", "main"]


def run_exp_d(bundle) -> pd.DataFrame:
    """返回每 (客户端 × 类别) 一行的 DataFrame。"""
    loaders = bundle.probe_loaders()
    # 探针集覆盖全部类别，用 ref+other+query+target 拼起来算散度最全面；
    # 这里用 'other' + 'ref' 两个子集（它们按类均衡，且不含扰动）。
    from torch.utils.data import ConcatDataset

    from .probe import make_eval_loader

    combined = ConcatDataset([bundle.probe.ref, bundle.probe.other])
    batch_size = (int(bundle.cfg.smoke.probe.batch_size) if bundle.smoke
                  else int(bundle.cfg.probe.batch_size))
    loader = make_eval_loader(combined, batch_size=batch_size)

    records = bundle.client_records()
    min_class_count = int(bundle.cfg.metrics.min_class_count)

    # --- 第一遍：收集 J ---
    per_client = {}
    for record in records:
        model = bundle.client_model(record["client_id"])
        frame = dispersion_scores(model, loader, None, device=bundle.device,
                                  num_classes=bundle.num_classes,
                                  min_class_count=min_class_count,
                                  source="probe")
        per_client[record["client_id"]] = frame

    # --- 基线：良性客户端 J 的中位数（nan-safe） ---
    benign_ids = [r["client_id"] for r in records if not r["is_malicious"]]
    if not benign_ids:
        # 全是恶意客户端时无法定义基线；v 全部为 nan，但 J 仍然有效。
        baseline = torch.full((bundle.num_classes,), float("nan"))
    else:
        stacked = np.stack([per_client[cid]["J"].to_numpy() for cid in benign_ids])
        with np.errstate(all="ignore"):
            baseline = torch.tensor(np.nanmedian(stacked, axis=0), dtype=torch.float32)

    # --- 第二遍：回填 v ---
    rows: List[dict] = []
    for record in records:
        base_row = common_row(bundle, record)
        frame = per_client[record["client_id"]]
        for _, entry in frame.iterrows():
            cls = int(entry["class_id"])
            j_value = float(entry["J"])
            v_value = float("nan")
            base = float(baseline[cls])
            if np.isfinite(j_value) and np.isfinite(base) and base > 0 and j_value > 0:
                v_value = float(np.log(j_value / base))
            rows.append({**base_row,
                         "class_id": cls,
                         "is_target_class": cls == bundle.target_class,
                         "n_class_samples_probe": int(entry["n_class_samples"]),
                         "n_class_samples_local": int(
                             record.get("label_histogram", [0] * bundle.num_classes)[cls]),
                         "J": j_value,
                         "v": v_value,
                         "benign_baseline_J": base})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="实验 D：源类表征混乱"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bundle = load_bundle(args, cfg)
    frame = run_exp_d(bundle)
    default = (f"expD_{bundle.meta['mode']}_a{bundle.meta['alpha']}"
               f"_s{bundle.meta['seed']}.csv")
    write_csv(frame, args.out, default, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
