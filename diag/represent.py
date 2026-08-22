"""表征指标：从 checkpoint **离线**计算（正式实验的"表征指标"一节）。

训练期算这些太贵（每个快照都要对整个探针集做多次前向），而它们只依赖模型权重
与固定的探针集 —— 完全可以事后补算。这也是规格里允许的做法。

在固定探针集上，对每个快照的每个被评估模型，记录倒数第二层特征，并导出：

============================  =================================================
指标                            定义
============================  =================================================
``feature_drift``             与**上一个快照**同一模型的特征差的平均 L2 ——
                              表征变化有多快
``clean_repr_distance``       与**参照模型**（通常是 clean run 的同轮快照）
                              同一批样本的特征差的平均 L2
``target_class_repr_distance``干净样本特征到目标类原型的平均距离
``trigger_repr_distance``     **加了触发器之后**的特征到目标类原型的平均距离
``trigger_pull``              上面两者之差 —— **后门在表征空间里的位移量**
============================  =================================================

# 主指标是 ``trigger_pull``，不是 ``trigger_repr_distance``

绝对距离受特征尺度影响，而不同轮次、不同模型的特征范数可以差好几倍；直接比
绝对值会把"特征整体变大了"读成"后门变强了"。``trigger_pull`` 是同一个模型、
同一批样本上的**配对差**，尺度问题自动抵消。同时另存
``feature_norm_mean``，好让尺度变化本身也可见。

# 距离一律在同一批样本上配对计算

``feature_drift`` 与 ``clean_repr_distance`` 都是"同一张图片在两个模型下的
特征之差"，不是两堆特征的分布距离。探针集是固定且有序的，所以逐行相减即可 ——
但这要求两次提取用的是**同一个 loader 顺序**，因此这里一律用
``ExpEProbe.loader()``（``make_eval_loader`` 不打乱）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import Cfg, load_config
from .features import class_prototypes, extract_penultimate

__all__ = ["embed", "prototype_distances", "representation_row",
           "representation_table", "main"]


def embed(model: Any, probe: Any, device, *, batch_size: int = 256,
          transform: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
    """探针集上的倒数第二层特征。返回 ``{"feats": [N,d], "labels": [N]}``。"""
    out = extract_penultimate(model, probe.loader(batch_size), device,
                              source="probe", transform=transform)
    return {"feats": out.feats.detach().cpu(),
            "labels": out.labels.detach().cpu()}


def prototype_distances(feats: torch.Tensor, labels: torch.Tensor,
                        target_class: int, num_classes: int
                        ) -> Dict[str, float]:
    """到目标类原型的平均距离，以及特征尺度。

    原型用**干净**特征算：若用加了触发器的特征算原型，触发器把样本推向哪里，
    原型就跟到哪里，距离恒定 —— 那是个自证的量。
    """
    stats = class_prototypes(feats, labels, num_classes)
    proto = stats["proto"][int(target_class)]
    if not torch.isfinite(proto).all():
        return {"target_proto_valid": 0.0,
                "distance_to_target_proto": float("nan"),
                "feature_norm_mean": float(feats.norm(dim=1).mean())}
    distance = (feats - proto[None, :]).norm(dim=1)
    return {"target_proto_valid": 1.0,
            "distance_to_target_proto": float(distance.mean()),
            "feature_norm_mean": float(feats.norm(dim=1).mean())}


def representation_row(model: Any, probe: Any, device, *,
                       target_class: int, num_classes: int,
                       batch_size: int = 256,
                       trigger_transform: Optional[Callable] = None,
                       previous_feats: Optional[torch.Tensor] = None,
                       reference_feats: Optional[torch.Tensor] = None
                       ) -> Dict[str, Any]:
    """一个模型的全部表征指标。返回的 dict 里带 ``_feats`` 供下一轮做 drift。"""
    clean = embed(model, probe, device, batch_size=batch_size)
    feats, labels = clean["feats"], clean["labels"]
    row: Dict[str, Any] = {"n_probe": int(feats.shape[0]),
                           "feature_dim": int(feats.shape[1])}
    row.update(prototype_distances(feats, labels, target_class, num_classes))
    row["target_class_repr_distance"] = row.pop("distance_to_target_proto")

    # 配对差：同一批样本、两个模型。形状不一致说明探针集变了，直接报错 ——
    # 静默地按最短长度截断会让 drift 变成一个无意义的数。
    for name, other in (("feature_drift", previous_feats),
                        ("clean_repr_distance", reference_feats)):
        if other is None:
            row[name] = float("nan")
            continue
        if other.shape != feats.shape:
            raise ValueError(
                f"{name}: 特征形状 {tuple(feats.shape)} 与对照 "
                f"{tuple(other.shape)} 不一致 —— 探针集或模型宽度变了，"
                f"配对距离无意义")
        row[name] = float((feats - other).norm(dim=1).mean())

    if trigger_transform is not None:
        triggered = embed(model, probe, device, batch_size=batch_size,
                          transform=trigger_transform)
        stats = class_prototypes(feats, labels, num_classes)
        proto = stats["proto"][int(target_class)]
        if torch.isfinite(proto).all():
            distance = (triggered["feats"] - proto[None, :]).norm(dim=1)
            row["trigger_repr_distance"] = float(distance.mean())
            # 主指标：同模型同样本的配对差，特征尺度自动抵消
            row["trigger_pull"] = (row["target_class_repr_distance"]
                                   - row["trigger_repr_distance"])
        else:
            row["trigger_repr_distance"] = float("nan")
            row["trigger_pull"] = float("nan")
        row["trigger_feature_norm_mean"] = float(
            triggered["feats"].norm(dim=1).mean())
    else:
        row["trigger_repr_distance"] = float("nan")
        row["trigger_pull"] = float("nan")
        row["trigger_feature_norm_mean"] = float("nan")

    row["_feats"] = feats
    return row


def representation_table(models: Sequence[Any], probe: Any, device, *,
                         labels: Sequence[Any],
                         target_class: int, num_classes: int,
                         batch_size: int = 256,
                         trigger_transform: Optional[Callable] = None,
                         reference_feats: Optional[torch.Tensor] = None
                         ) -> "Any":
    """按顺序处理一串模型（通常是按轮次排序的快照）。

    ``feature_drift`` 是相对**前一个**模型的，所以顺序有意义 —— 调用方必须
    按轮次升序传入。第一个模型的 drift 是 nan（没有前一个），**不是 0**。
    """
    import pandas as pd

    if len(models) != len(labels):
        raise ValueError(f"models({len(models)}) 与 labels({len(labels)}) "
                         f"长度不一致")
    rows: List[Dict[str, Any]] = []
    previous: Optional[torch.Tensor] = None
    for model, label in zip(models, labels):
        row = representation_row(
            model, probe, device, target_class=target_class,
            num_classes=num_classes, batch_size=batch_size,
            trigger_transform=trigger_transform,
            previous_feats=previous, reference_feats=reference_feats)
        previous = row.pop("_feats")
        rows.append({**(label if isinstance(label, dict) else {"label": label}),
                     **row})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 checkpoint 离线计算表征指标")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-dir", required=True,
                        help="attack run 的 checkpoint 目录（含 round_XXXX/）")
    parser.add_argument("--clean-ckpt-dir", default=None,
                        help="对照 clean run；给了才有 clean_repr_distance")
    parser.add_argument("--client-id", type=int, default=None,
                        help="评估哪个客户端的个性化模型；缺省用全局模型（借 BN）")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/exp1_representation.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-trigger", action="store_true",
                        help="跳过触发器表征（没有生成器快照时用）")
    args = parser.parse_args(argv)

    from .exp_e import build_exp_e_probe
    from .exp_f import load_snapshot_model
    from .run_fl import build_datasets
    from .snapshots import available_rounds

    cfg = load_config(args.config)
    device = torch.device(args.device or
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = Path(args.ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    smoke = bool(meta.get("smoke", False))
    target_class = int(meta.get("target_class", cfg.data.target_class))
    num_classes = int(meta.get("num_classes", cfg.data.num_classes))
    model_size = int(meta.get("model_size", cfg.fl.get("model_size", 10)))

    _, test_dataset, _ = build_datasets(cfg, smoke,
                                        meta.get("synthetic_sizes"))
    probe_cfg = cfg
    if smoke:
        probe_cfg = Cfg({**cfg, "exp_e": {**cfg["exp_e"],
                                          "probe_n": int(cfg.smoke.exp_e_probe_n)}})
    probe = build_exp_e_probe(test_dataset, target_class, probe_cfg)

    kind = "personalized" if args.client_id is not None else "global_bn_borrowed"
    rounds = available_rounds(ckpt_dir,
                              "client" if args.client_id is not None else "global",
                              args.client_id)
    if not rounds:
        raise SystemExit(
            f"{ckpt_dir} 下没有按轮次快照。本脚本需要 --snapshot-every 跑过的 "
            f"run —— 只有最终模型时 feature_drift 全是 nan，这张表没有意义。")

    donor = args.client_id
    if kind == "global_bn_borrowed":
        benign = [int(c["client_id"]) for c in meta.get("clients", [])
                  if not c.get("is_malicious", False)]
        if not benign:
            raise SystemExit("meta.json 里没有良性客户端，无法借 BN")
        donor = min(benign)

    print(f"[represent] {len(rounds)} 个快照轮次，kind={kind}"
          + (f"，BN 供体 = client {donor}" if kind == "global_bn_borrowed"
             else ""))

    models, labels = [], []
    for round_number in rounds:
        models.append(load_snapshot_model(
            ckpt_dir, round_number, kind=kind, client_id=args.client_id,
            num_classes=num_classes, device=device, bn_donor_id=donor,
            model_size=model_size))
        labels.append({"round": int(round_number), "kind": kind,
                       "client_id": args.client_id})

    trigger_transform = None
    if not args.no_trigger:
        from .hooks import extract_generator  # noqa: F401
        from .perturb import make_delta_fn, make_xi_fn
        generator_path = ckpt_dir / "generator.pt"
        if not generator_path.exists():
            print("[represent] ⚠️ 没有 generator.pt，触发器表征列全为 nan")
        else:
            from generator import Autoencoder
            generator = Autoencoder().to(device)
            generator.load_state_dict(torch.load(generator_path,
                                                 map_location=device))
            generator.eval()
            delta_fn = make_delta_fn(generator,
                                     eps=float(cfg.perturb.eps_delta))
            # make_delta_fn 返回的是 images -> images（δ 与标签无关），
            # 而 extract_penultimate 的 transform 签名是 (images, labels)
            def trigger_transform(images, targets, _delta=delta_fn):
                return _delta(images)
            print("[represent] 触发器用 δ-only（生成器最终快照）。"
                  "ξ 依赖被评估模型，逐快照重建的成本在这里不划算 —— "
                  "口径与实验 F 的 delta_only 一致。")

    table = representation_table(
        models, probe, device, labels=labels, target_class=target_class,
        num_classes=num_classes, batch_size=args.batch_size,
        trigger_transform=trigger_transform)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"[represent] {len(table)} 行 -> {out}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
