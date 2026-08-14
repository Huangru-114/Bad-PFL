"""实验 E：Bad-PFL 的对抗地板测定。

# 要回答什么

Bad-PFL 声称 δ 是"目标类的自然特征"。但 δ（生成器对着 θ_g 做梯度优化）与
ξ（单步 FGSM）都是**对抗噪声的标准配方**。论文从未排除一种可能：

    δ+ξ 对一个**从未被攻击过**的干净模型可能就已经有效。

若如此，论文报告的 ASR 中有一部分不是"被植入的后门"，而是"模型固有的
非鲁棒特征依赖"被利用的结果。要得到的分解::

    ASR_attacked = ASR_clean_transfer + Delta
                   ────────────────────   ─────
                   对抗地板：任何后门        训练真正植入的部分：
                   防御都消不掉             不变性类防御能消除的上限

# 四个 cell

| cell | 被评估模型 | 生成器来源 | ξ 的计算模型 | 得到的量 |
|------|-----------|-----------|-------------|---------|
| E1 | attack run | attack run | 自身（白盒） | ``ASR_attacked``（复现论文） |
| E2 | **clean run** | attack run | 自身（白盒） | **``ASR_clean_transfer`` = 对抗地板** |
| E3 | clean run | 针对该模型现训 | 自身（白盒） | ``ASR_clean_oracle`` = 白盒天花板 |
| E4 | clean run | attack run | attack 的**全局模型**（黑盒） | 剥离测试时白盒访问后的地板 |

E2 是核心。E3 用于区分"干净模型原则上不响应"与"只是迁移性差"——两者
对论文叙事的含义完全不同。

# 两个必须分开报的指标

    ASR_targeted = P(pred == y_t | 施加扰动, y_true != y_t)     定向
    ErrorRate    = P(pred != y_true | 施加扰动, y_true != y_t)  非定向

**不能只报 ASR。** ξ 预期造成高 ErrorRate 但低 ASR_targeted —— 这正是
"对抗成分"与"后门成分"的分水岭，混在一起会直接得出错误结论。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset

from . import REPO_ROOT  # noqa: F401
from .config import Cfg, load_config
from .exp_common import resolve_device
from .hooks import load_checkpoint, load_meta, run_dir_name, state_dict_hash
from .perturb import (DATA_SOURCE_MODES, apply_perturbation, linf,
                      make_delta_fn, make_xi_fn)
from .probe import get_targets, make_eval_loader
from .run_fl import build_datasets

from generator import Autoencoder
from resnet import get_resnet

__all__ = ["EXP_E_MODES", "ExpEProbe", "build_exp_e_probe", "evaluate_mode",
           "train_oracle_generator", "run_cell", "main"]

EXP_E_MODES: Tuple[str, ...] = ("none", "full", "delta_only", "xi_only",
                                "random_eps4", "random_eps8", "real_target")

_MODE_NEEDS_DELTA = ("full", "delta_only")
_MODE_NEEDS_XI = ("full", "xi_only")
_RANDOM_MODES = ("random_eps4", "random_eps8")
_NOMINAL_LINF = {"none": 0.0, "full": 8.0 / 255.0, "delta_only": 4.0 / 255.0,
                 "xi_only": 4.0 / 255.0, "random_eps4": 4.0 / 255.0,
                 "random_eps8": 8.0 / 255.0, "real_target": 0.0}


# ---------------------------------------------------------------------------
# 探针集
# ---------------------------------------------------------------------------
@dataclass
class ExpEProbe:
    """共享探针集：从官方测试集固定采样 N 张，所有客户端模型共用同一份。

    理由是跨客户端可比性 —— 各客户端本地测试集在高异构下样本量与类分布差异
    巨大，无法支撑跨客户端比较。本地测试集的数字作为**次要指标**另行报告。

    ``target_subset`` 是其中真实属于目标类的那些图片，供 ``real_target``
    正对照使用（它换的是数据，不是扰动）。
    """

    subset: Subset
    target_subset: Subset
    indices: List[int]
    target_indices: List[int]
    target_class: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def loader(self, batch_size: int = 256):
        return make_eval_loader(self.subset, batch_size=batch_size)

    def target_loader(self, batch_size: int = 256):
        return make_eval_loader(self.target_subset, batch_size=batch_size)


def build_exp_e_probe(test_dataset, target_class: int, cfg: Cfg) -> ExpEProbe:
    """从测试集抽 ``cfg.exp_e.probe_n`` 张，固定 seed，与实验 seed 解耦。

    采用**不分层**的均匀随机抽样：保留 CIFAR-10 测试集的自然类别比例，
    使得 ``asr_targeted_unfiltered`` 与论文口径可比（论文不排除目标类样本）。
    抽样使用独立的 ``RandomState``，不触碰全局 numpy RNG。
    """
    exp_cfg = cfg.exp_e
    n = int(exp_cfg.probe_n)
    seed = int(exp_cfg.probe_seed)

    targets = get_targets(test_dataset)
    if n > len(targets):
        raise ValueError(f"probe_n={n} 超过测试集大小 {len(targets)}")

    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(len(targets), size=n, replace=False))
    labels = targets[indices]
    target_indices = indices[labels == int(target_class)]

    if len(target_indices) == 0:
        raise ValueError(
            f"探针集中没有目标类 {target_class} 的样本，real_target 无法评估")

    return ExpEProbe(
        subset=Subset(test_dataset, indices.tolist()),
        target_subset=Subset(test_dataset, target_indices.tolist()),
        indices=indices.tolist(),
        target_indices=target_indices.tolist(),
        target_class=int(target_class),
        meta={"probe_n": n, "probe_seed": seed,
              "n_target_in_probe": int(len(target_indices)),
              "n_non_target_in_probe": int(n - len(target_indices))},
    )


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------
def _predict(model: nn.Module, loader, device,
             transform: Optional[Callable] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """跑一遍 loader，返回 ``(preds, labels, 实测 L∞)``。

    - 强制 ``model.eval()``（陷阱清单第 2 条：FedBN 下 BN 统计量是私有的，
      train 模式会污染结果），返回前恢复原状态。
    - ``transform`` 在**梯度启用**区执行（ξ 的 PGD 需要反传），前向在
      ``no_grad`` 下；且**不套 autocast**，因此 PGD 全程 fp32
      （陷阱清单第 8 条）。
    """
    was_training = model.training
    model.eval()
    preds, labels_all, max_linf = [], [], 0.0
    try:
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            perturbed = images if transform is None else transform(images, labels)
            if transform is not None:
                max_linf = max(max_linf, linf(perturbed, images))
            with torch.no_grad():
                logits = model(perturbed)
            preds.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())
    finally:
        if was_training:
            model.train()
    return torch.cat(preds), torch.cat(labels_all), max_linf


def _metrics(preds: torch.Tensor, labels: torch.Tensor,
             target_class: int) -> Dict[str, float]:
    """双指标 + 分子分母计数（陷阱清单第 5 条要求计数可复核）。

    ``real_target`` 那一组的 ``eligible`` 为空（全部样本都是目标类），
    此时 ``asr_targeted`` / ``error_rate`` 为 **nan** —— 该组有意义的数字是
    ``asr_targeted_unfiltered``，它等于模型在目标类上的准确率。
    """
    eligible = labels != int(target_class)
    n_eligible = int(eligible.sum())
    n_all = int(len(preds))

    # 全部用整数计数相除：float32 的 .mean() 会引入 ~1e-8 的舍入
    # （0.8 会变成 0.800000011920929），而这些比率是要落盘复核的。
    hit = int((preds[eligible] == int(target_class)).sum()) if n_eligible else 0
    wrong = int((preds[eligible] != labels[eligible]).sum()) if n_eligible else 0
    hit_all = int((preds == int(target_class)).sum())
    return {
        "asr_targeted": hit / n_eligible if n_eligible else float("nan"),
        "error_rate": wrong / n_eligible if n_eligible else float("nan"),
        "asr_targeted_unfiltered": hit_all / n_all if n_all else float("nan"),
        "n_eval_samples": n_eligible,
        "n_eval_samples_unfiltered": n_all,
        "asr_numerator": hit,
        "error_numerator": wrong,
        "asr_unfiltered_numerator": hit_all,
    }


# ---------------------------------------------------------------------------
# 单个扰动模式的评估
# ---------------------------------------------------------------------------
def evaluate_mode(model: nn.Module, probe: ExpEProbe, mode: str, *, device,
                  cfg: Cfg, delta_fn: Optional[Callable] = None,
                  xi_fn: Optional[Callable] = None,
                  batch_size: int = 256,
                  noise_dist: str = "uniform") -> Dict[str, Any]:
    """在共享探针集上评估一个扰动模式，返回一行的指标。

    ``random_*`` 模式会重复采样 ``cfg.exp_e.random_repeats`` 次取均值 ——
    单次采样方差太大（陷阱清单第 9 条）。
    """
    if mode not in EXP_E_MODES:
        raise ValueError(f"未知模式 '{mode}'；可选 {list(EXP_E_MODES)}")
    if mode in _MODE_NEEDS_DELTA and delta_fn is None:
        raise ValueError(f"模式 '{mode}' 需要 delta_fn")
    if mode in _MODE_NEEDS_XI and xi_fn is None:
        raise ValueError(f"模式 '{mode}' 需要 xi_fn")

    pixel_min = float(cfg.perturb.pixel_min)
    pixel_max = float(cfg.perturb.pixel_max)
    repeats = int(cfg.exp_e.random_repeats) if mode in _RANDOM_MODES else 1

    # real_target 换的是数据不是扰动：用真实目标类图片、不加任何扰动。
    if mode in DATA_SOURCE_MODES:
        loader = probe.target_loader(batch_size)
        preds, labels, realized = _predict(model, loader, device)
        row = _metrics(preds, labels, probe.target_class)
        row.update({"perturb_mode": mode, "linf_budget": _NOMINAL_LINF[mode],
                    "linf_realized": realized, "noise_dist": "", "repeats": 1})
        return row

    loader = probe.loader(batch_size)
    if mode == "none":
        preds, labels, realized = _predict(model, loader, device)
        row = _metrics(preds, labels, probe.target_class)
        row.update({"perturb_mode": mode, "linf_budget": 0.0,
                    "linf_realized": realized, "noise_dist": "", "repeats": 1})
        return row

    collected: List[Dict[str, float]] = []
    realized_max = 0.0
    for repeat in range(repeats):
        counter = {"n": 0}
        base_seed = int(cfg.perturb.noise_seed) + repeat * 100003

        def transform(images, labels, _seed=base_seed, _counter=counter):
            rng = torch.Generator(device=images.device)
            rng.manual_seed(_seed + _counter["n"])
            _counter["n"] += 1
            return apply_perturbation(
                images, mode, delta_fn=delta_fn, xi_fn=xi_fn, labels=labels,
                rng=rng, noise_dist=noise_dist,
                pixel_min=pixel_min, pixel_max=pixel_max)

        preds, labels, realized = _predict(model, loader, device, transform)
        realized_max = max(realized_max, realized)
        collected.append(_metrics(preds, labels, probe.target_class))

    row: Dict[str, Any] = {}
    for key in collected[0]:
        values = [entry[key] for entry in collected]
        row[key] = float(np.nanmean(values))
    # 计数取整，均值只对比率有意义
    for key in ("n_eval_samples", "n_eval_samples_unfiltered"):
        row[key] = int(collected[0][key])
    row.update({"perturb_mode": mode, "linf_budget": _NOMINAL_LINF[mode],
                "linf_realized": realized_max,
                "noise_dist": noise_dist if mode in _RANDOM_MODES else "",
                "repeats": repeats})
    return row


# ---------------------------------------------------------------------------
# E3：针对干净模型现训生成器
# ---------------------------------------------------------------------------
def train_oracle_generator(clean_model: nn.Module, train_loader, target_class: int,
                           *, cfg: Cfg, device) -> Tuple[nn.Module, Dict[str, Any]]:
    """在**干净模型**上现训一个生成器，给出白盒天花板。

    协议与 ``fba.trigger_gen_trainer``（fba.py:31-43）**逐项一致**：
    Adam / lr=1e-2 / 30 步；每步 ``adv = pgd_attack(model, x, y_true)``、
    ``δ = gen(x)/255*4``、``CE(model(adv + δ), y_t)``，**只更新生成器**。

    陷阱清单第 7 条：训练前后比对 ``clean_model`` 的参数哈希，不一致立即报错。
    ``pgd_attack`` 与本函数的 backward 都会把梯度累加进 ``clean_model`` 的
    参数（fba.py:16），但优化器里只有生成器的参数，所以**权重不会变**；
    这里仍然显式清零梯度，避免给调用方留下副作用。
    """
    import fba

    steps = int(cfg.exp_e.oracle_steps)
    lr = float(cfg.exp_e.oracle_lr)
    eps_xi = float(cfg.perturb.eps_xi)
    eps_delta = float(cfg.perturb.eps_delta)

    clean_model.eval()
    hash_before = state_dict_hash(clean_model)

    generator = Autoencoder().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    iterator = iter(train_loader)
    for _ in range(steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        adv = fba.pgd_attack(clean_model, images, labels.long(),
                             epsilon=eps_xi, alpha=eps_xi, num_iter=1)
        trigger = generator(images) * eps_delta
        logits = clean_model(adv + trigger)
        target = torch.full((labels.size(0),), int(target_class),
                            device=labels.device, dtype=torch.long)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        clean_model.zero_grad(set_to_none=True)

    hash_after = state_dict_hash(clean_model)
    if hash_before != hash_after:
        raise RuntimeError(
            "E3 现训生成器改动了 clean_model 的参数！\n"
            f"  训练前: {hash_before}\n  训练后: {hash_after}\n"
            "这会让 E3 变成循环论证，必须中止。")

    generator.eval()
    return generator, {"oracle_steps": steps, "oracle_lr": lr,
                       "clean_model_hash": hash_before,
                       "clean_model_unchanged": True}


# ---------------------------------------------------------------------------
# cell 驱动
# ---------------------------------------------------------------------------
def _load_model(path: Path, num_classes: int, device) -> nn.Module:
    model = get_resnet(size=10, num_classes=num_classes)
    model.load_state_dict(load_checkpoint(path), strict=True)
    model.to(device)
    model.device = device
    model.eval()
    return model


def _load_generator(path: Path, device) -> nn.Module:
    generator = Autoencoder()
    generator.load_state_dict(load_checkpoint(path), strict=True)
    generator.to(device)
    generator.eval()
    return generator


def run_cell(cell: str, ckpt_root: Path, alpha: float, seed: int, *, cfg: Cfg,
             device, modes: Sequence[str] = EXP_E_MODES,
             client_ids: Optional[Sequence[int]] = None,
             noise_dist: str = "uniform") -> pd.DataFrame:
    """跑一个 cell（E1/E2/E3/E4），返回每（客户端 × 模式）一行的 DataFrame。"""
    cell = cell.upper()
    if cell not in ("E1", "E2", "E3", "E4"):
        raise ValueError(f"未知 cell '{cell}'")

    attack_dir = ckpt_root / run_dir_name("attack", alpha, seed)
    clean_dir = ckpt_root / run_dir_name("clean", alpha, seed)
    eval_dir = attack_dir if cell == "E1" else clean_dir
    run_mode = "attack" if cell == "E1" else "clean"

    meta = load_meta(eval_dir / "meta.json")
    # 陷阱清单第 1 条：E2/E3/E4 的被评估模型必须来自 clean run，
    # 否则就是拿被投毒的模型去论证"δ 是否天然有效"——循环论证。
    if cell in ("E2", "E3", "E4"):
        assert str(meta["mode"]) == "clean", (
            f"{cell} 必须评估 clean run 的模型，但 meta 显示 mode="
            f"{meta['mode']}。用被投毒的模型做这个论证是循环论证。")
    assert str(meta["mode"]) == run_mode

    num_classes = int(meta["num_classes"])
    target_class = int(meta["target_class"])
    batch_size = int(cfg.probe.batch_size)

    _, test_dataset, _ = build_datasets(
        cfg, bool(meta.get("smoke", False)),
        dataset_sizes=meta.get("synthetic_sizes"))
    probe = build_exp_e_probe(test_dataset, target_class, cfg)

    # 生成器来源
    if cell in ("E1", "E2", "E4"):
        generator = _load_generator(attack_dir / "generator.pt", device)
        generator_source = f"attack_a{alpha}_s{seed}/final"
    else:
        generator = None                    # E3 每个客户端各训一个
        generator_source = "oracle_per_client"

    # ξ 的计算模型：E4 用 attack run 的**全局模型**（黑盒），其余用被评估模型自身
    blackbox_model = None
    if cell == "E4":
        blackbox_model = _load_model(attack_dir / "global.pt", num_classes, device)

    records = list(meta["clients"])
    if client_ids is not None:
        wanted = set(int(c) for c in client_ids)
        records = [r for r in records if int(r["client_id"]) in wanted]

    train_dataset = None
    if cell == "E3":
        train_dataset, _, _ = build_datasets(
            cfg, bool(meta.get("smoke", False)),
            dataset_sizes=meta.get("synthetic_sizes"))

    rows: List[Dict[str, Any]] = []
    for record in records:
        cid = int(record["client_id"])
        model = _load_model(eval_dir / f"client_{cid}.pt", num_classes, device)

        cell_generator, oracle_info = generator, {}
        if cell == "E3":
            local = make_eval_loader(
                Subset(train_dataset, list(record["train_indices"])),
                batch_size=int(cfg.fl.batch_size))
            cell_generator, oracle_info = train_oracle_generator(
                model, local, target_class, cfg=cfg, device=device)

        delta_fn = make_delta_fn(cell_generator, eps=float(cfg.perturb.eps_delta))
        xi_model = blackbox_model if cell == "E4" else model
        xi_source = (f"attack_a{alpha}_s{seed}/global" if cell == "E4"
                     else f"self/client_{cid}")
        xi_fn = make_xi_fn(xi_model, eps=float(cfg.perturb.eps_xi),
                           alpha=float(cfg.perturb.pgd_alpha),
                           num_iter=int(cfg.perturb.pgd_num_iter),
                           seed=int(cfg.perturb.xi_seed))

        clean_row = evaluate_mode(model, probe, "none", device=device, cfg=cfg,
                                  batch_size=batch_size)
        clean_acc = 1.0 - float(clean_row["error_rate"])

        for mode in modes:
            row = evaluate_mode(model, probe, mode, device=device, cfg=cfg,
                                delta_fn=delta_fn, xi_fn=xi_fn,
                                batch_size=batch_size, noise_dist=noise_dist)
            row.update({
                "client_id": cid,
                "is_malicious": bool(record["is_malicious"]),
                "cell": cell,
                "generator_source": generator_source,
                "xi_source": xi_source,
                "alpha": float(alpha), "seed": int(seed),
                "probe_set": "shared", "clean_acc": clean_acc,
                "poison_ratio": float(record.get("poison_ratio", 0.0)),
                "eval_run_mode": run_mode,
                **oracle_info,
            })
            rows.append(row)
        del model
        print(f"[exp_e] {cell} α={alpha} s={seed} client {cid} 完成")

    frame = pd.DataFrame(rows)
    column_order = ["client_id", "is_malicious", "cell", "perturb_mode",
                    "generator_source", "xi_source", "alpha", "seed",
                    "probe_set", "n_eval_samples", "asr_targeted",
                    "asr_targeted_unfiltered", "error_rate", "clean_acc",
                    "linf_budget", "poison_ratio"]
    rest = [c for c in frame.columns if c not in column_order]
    return frame[[c for c in column_order if c in frame.columns] + rest]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 E：对抗地板测定")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--cell", required=True, choices=["E1", "E2", "E3", "E4"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--modes", nargs="*", default=list(EXP_E_MODES))
    parser.add_argument("--client-ids", type=int, nargs="*", default=None)
    parser.add_argument("--noise-dist", default="uniform",
                        choices=["uniform", "rademacher"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    frame = run_cell(args.cell, Path(args.ckpt_root), args.alpha, args.seed,
                     cfg=cfg, device=device, modes=args.modes,
                     client_ids=args.client_ids, noise_dist=args.noise_dist)

    out = Path(args.out) if args.out else (
        Path(cfg.paths.results_root) / "raw" /
        f"exp_e_{args.cell}_a{args.alpha}_s{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[exp_e] {len(frame)} 行 -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
