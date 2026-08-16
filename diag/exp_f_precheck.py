"""实验 F 的 Phase 0 核查：快照可用性 + FedBN 全局模型的 BN 诊断。

# 为什么需要这个脚本

实验 F 要构造 (生成器轮次 s, 模型轮次 t) 的矩阵，前提是两侧都有**按轮次**的
快照。代码审查已经能确定一件事、无需集群：

    按轮次的**模型**快照从来没有实现过。
    ``hooks.save_run`` 只在 ``run_fl.py`` 末尾调用一次；官方仓库里
    ``torch.save`` 一次都没出现。

所以"是否需要重跑"这个分支不用查就已确定：**需要**。本脚本要回答的是余下的
几个只有在集群上才能确定的问题，以及一个会影响实验设计的诊断。

# 四项检查

1. **盘点** —— 每个 run 实际拥有哪些轮次的生成器 / 模型快照。
2. **BN 诊断（关键）** —— 证明 FedBN 下 ``global.pt`` 的 BN 从未被更新。
   ``pfl.py:5-12`` 的 ``fedbn_update`` 在 ``before_update_global`` 阶段把所有
   含 ``bn`` / ``shortcut.1`` 的键从 ``server.update`` 里 pop 掉，随后
   ``server.py:34`` 的 ``load_state_dict(strict=False)`` 就再也碰不到 BN。
   检验方式是直接读盘：全部 BN 层应满足
   ``num_batches_tracked == 0``、``running_mean == 0``、``running_var == 1``、
   ``weight == 1``、``bias == 0``。这是不可辩驳的证据，强于任何间接推断。
3. **量化后果** —— 测三种评估对象在实验 E 探针集上的 clean accuracy：
   原始 ``global.pt`` / 借某个良性客户端 BN 的全局模型 / 该客户端自己的
   个性化模型。这个数字决定实验 F 的主评估对象，也量化了实验 E 里
   **E4 用 ``attack_dir/global.pt`` 当黑盒 ξ 模型**（``exp_e.py:406``）
   这一选择的代价。
4. **成本估算** —— 重跑要多少存储。

用法::

    python -m diag.exp_f_precheck --ckpt-root ./checkpoints --out F0_PRECHECK.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import REPO_ROOT  # noqa: F401
from .config import Cfg, load_config
from .exp_common import resolve_device
from .fedbn import is_fedbn_private_key
from .hooks import load_checkpoint, load_meta, run_dir_name
from .snapshots import available_rounds, build_grid

__all__ = ["inventory_snapshots", "diagnose_global_bn", "measure_clean_acc",
           "estimate_rerun_cost", "main"]

# BN 在 PyTorch 里的初始值。FedBN 若真的从不更新全局模型的 BN，落盘的值必须
# 与这张表逐项相符。
_BN_INIT = {"weight": 1.0, "bias": 0.0, "running_mean": 0.0, "running_var": 1.0}


# 判定规则的**唯一定义**在 diag/fedbn.py
_is_bn_key = is_fedbn_private_key


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float) and np.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame is None or len(frame) == 0:
        return "_（无数据）_"
    header = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for position in range(len(frame)):
        lines.append("| " + " | ".join(
            _fmt(frame[column].iloc[position], digits) for column in header) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. 盘点
# ---------------------------------------------------------------------------
def inventory_snapshots(ckpt_root: Path) -> Tuple[pd.DataFrame, str]:
    """每个 run 拥有哪些轮次的生成器 / 模型快照。

    生成器有两种可能的落盘布局：
    - 旧：``generator/generator_round{t:04d}.pt``（``attach_generator_checkpoint_hook``）
    - 新：``round_{t:04d}/generator.pt``（``SnapshotRecorder``，与模型同目录）
    两者都要盘点，因为旧 run 只有前者。
    """
    from .audit.common import discover_runs

    runs = discover_runs(ckpt_root)
    rows: List[Dict[str, Any]] = []
    for _, run in runs.iterrows():
        directory = Path(run["path"])
        legacy = sorted(
            int(path.stem.replace("generator_round", ""))
            for path in (directory / "generator").glob("generator_round*.pt"))
        new_gen = available_rounds(directory, "generator")
        model_rounds = available_rounds(directory, "global")
        client_dirs = sorted(directory.glob("round_*"))
        n_client_snaps = sum(len(list(d.glob("client_*.pt"))) for d in client_dirs)
        paired = sorted(set(legacy) | set(new_gen))
        paired = [r for r in paired if r in set(model_rounds)]
        rows.append({
            "mode": run["mode"], "alpha": run["alpha"], "seed": run["seed"],
            "gen_rounds_legacy": len(legacy),
            "gen_rounds_new": len(new_gen),
            "model_rounds": len(model_rounds),
            "client_snapshots": n_client_snaps,
            "paired_rounds": len(paired),
            "has_manifest": (directory / "snapshot_manifest.json").exists(),
        })
    frame = pd.DataFrame(rows)

    lines = ["### 1. 快照盘点", "", _table(frame, digits=2), ""]
    if frame.empty:
        lines += [f"- ⚠️ 在 `{ckpt_root}` 下没发现任何 run，**未能确定**。", ""]
        return frame, "\n".join(lines)

    attack = frame[frame["mode"] == "attack"]
    max_paired = int(attack["paired_rounds"].max()) if len(attack) else 0
    lines += [
        f"- 成对（同一轮次同时有生成器与模型）的最大轮次数：**{max_paired}**。",
        "",
    ]
    if max_paired >= 5:
        lines += ["- ✅ 已有 ≥5 个成对轮次，**可以直接进入 P2**。", ""]
    else:
        lines += [
            "- ❌ 成对轮次不足 5 个，**必须补一次带快照埋点的 attack run**。",
            "- 成因（代码级确定，无需数据）：按轮次的**模型**快照在本仓库中"
            "从未实现 —— `hooks.save_run` 只在 `run_fl.py` 末尾调用一次，"
            "官方仓库里 `torch.save` 一次都没出现。",
            "- 重跑命令：",
            "",
            "  ```bash",
            "  python -m diag.run_fl --mode attack --alpha 0.5 --seed 0 \\",
            "      --snapshot-every 50 \\",
            "      --verify-against ./checkpoints/attack_a0.5_s0",
            "  ```",
            "",
            "  `--verify-against` 会逐位比对新旧 run 的 checkpoint。一致则"
            "证明快照埋点没有改变训练动力学，实验 E 的结论可直接沿用。",
            "",
        ]
    return frame, "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. BN 诊断
# ---------------------------------------------------------------------------
def diagnose_global_bn(global_ckpt: Path) -> Tuple[bool, pd.DataFrame, str]:
    """检验 ``global.pt`` 里的 BN 是否停在初始化值。

    Returns
    -------
    ``(bn_frozen, frame, report)``：``bn_frozen=True`` 表示 BN 确实从未更新。
    """
    state = load_checkpoint(global_ckpt)
    bn_keys = [k for k in state if _is_bn_key(k)]
    non_bn = [k for k in state if not _is_bn_key(k)]

    rows: List[Dict[str, Any]] = []
    all_at_init = True
    for key in sorted(bn_keys):
        tensor = state[key]
        suffix = key.rsplit(".", 1)[-1]
        expected = _BN_INIT.get(suffix)
        if suffix == "num_batches_tracked":
            value = float(tensor.item())
            at_init = value == 0.0
            expected_str = "0"
        elif expected is None:
            continue
        else:
            value = float(tensor.detach().float().mean().item())
            at_init = bool(torch.allclose(tensor.detach().float(),
                                          torch.full_like(tensor.detach().float(),
                                                          float(expected))))
            expected_str = str(expected)
        all_at_init = all_at_init and at_init
        rows.append({"key": key, "expected": expected_str,
                     "observed_mean": value, "at_init": at_init})

    frame = pd.DataFrame(rows)
    n_at_init = int(frame["at_init"].sum()) if len(frame) else 0
    lines = [
        "### 2. FedBN 下全局模型的 BN 诊断", "",
        f"- 检查文件：`{global_ckpt}`",
        f"- BN 相关键 {len(bn_keys)} 个（判定规则与 `pfl.py:8` 逐字一致："
        f"`\"bn\" in key or \"shortcut.1\" in key`），非 BN 键 {len(non_bn)} 个。",
        f"- 停在初始化值的条目：**{n_at_init} / {len(frame)}**。",
        "",
    ]
    if all_at_init and len(frame):
        lines += [
            "- ✅ **确证：全局模型的 BN 从训练开始到结束一位都没变。**",
            "- 机制：`pfl.py:5-12` 的 `fedbn_update` 挂在 `before_update_global`，"
            "把全部 BN 键从 `server.update` 里 pop 掉；`server.py:34` 的"
            "`load_state_dict(self.update, strict=False)` 于是永远碰不到 BN。"
            "ResNet 的 BN 键全部匹配该规则（`resnet.py:22,25,32`）。",
            "- **这不是 bug，是 FedBN 的固有语义**（个性化模型 = 共享参数 +"
            "各自私有 BN），但它意味着 `global.pt` 不能被单独当作一个可用模型。",
            "",
        ]
    elif len(frame):
        lines += [
            f"- ⚠️ 有 {len(frame) - n_at_init} 个 BN 条目**不在**初始化值上，"
            "与代码推断不符。需人工确认 `pfl.py` 或聚合路径是否已被改动。",
            "", _table(frame[~frame["at_init"]].head(20)), "",
        ]
    else:
        lines += ["- ⚠️ 没有找到任何 BN 键，**未能确定**。", ""]
    return all_at_init, frame, "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. clean accuracy 三方对照
# ---------------------------------------------------------------------------
def borrow_bn(global_state: Dict[str, torch.Tensor],
              donor_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """全局共享参数 + 供体客户端的 BN = "借-BN 全局模型"。

    这是 FedBN 下唯一能把全局模型变成可评估对象的方式：共享参数确实是聚合过的，
    缺的只是 BN。供体的选择本身是一个自由度，必须在结果中如实记录。
    """
    merged = {key: value.clone() for key, value in global_state.items()}
    for key in merged:
        if _is_bn_key(key) and key in donor_state:
            merged[key] = donor_state[key].clone()
    return merged


def measure_clean_acc(ckpt_dir: Path, cfg: Cfg, device) -> Tuple[pd.DataFrame, str]:
    """在实验 E 的共享探针集上测三种评估对象的 clean accuracy。"""
    from resnet import get_resnet

    from .exp_e import build_exp_e_probe, _metrics, _predict
    from .run_fl import build_datasets

    meta = load_meta(ckpt_dir / "meta.json")
    num_classes = int(meta["num_classes"])
    target_class = int(meta["target_class"])
    _, test_dataset, _ = build_datasets(
        cfg, bool(meta.get("smoke", False)),
        dataset_sizes=meta.get("synthetic_sizes"))
    probe = build_exp_e_probe(test_dataset, target_class, cfg)
    loader = probe.loader(int(cfg.probe.batch_size))

    benign = [r for r in meta["clients"] if not bool(r["is_malicious"])]
    if not benign:
        return pd.DataFrame(), "### 3. clean accuracy\n\n- ⚠️ 没有良性客户端。\n"
    donor_id = int(sorted(r["client_id"] for r in benign)[0])

    global_state = load_checkpoint(ckpt_dir / "global.pt")
    donor_state = load_checkpoint(ckpt_dir / f"client_{donor_id}.pt")

    variants = {
        "global_raw": global_state,
        "global_bn_borrowed": borrow_bn(global_state, donor_state),
        f"personalized_client_{donor_id}": donor_state,
    }

    rows: List[Dict[str, Any]] = []
    for name, state in variants.items():
        model = get_resnet(size=10, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.device = device
        preds, labels, _ = _predict(model, loader, device)
        stats = _metrics(preds, labels, target_class)
        rows.append({
            "eval_target": name,
            "clean_acc": 1.0 - float(stats["error_rate"]),
            "asr_targeted_none": float(stats["asr_targeted"]),
            "n_eval_samples": int(stats["n_eval_samples"]),
        })
        del model

    frame = pd.DataFrame(rows)
    raw_acc = float(frame.loc[frame["eval_target"] == "global_raw",
                              "clean_acc"].iloc[0])
    borrowed_acc = float(frame.loc[frame["eval_target"] == "global_bn_borrowed",
                                   "clean_acc"].iloc[0])

    lines = [
        "### 3. 三种评估对象的 clean accuracy（实验 E 的共享探针集）", "",
        f"- BN 供体：良性客户端 `client_{donor_id}`（快照客户端里 id 最小者）。",
        "", _table(frame), "",
        f"- 原始全局模型 {raw_acc:.4f} vs 借-BN 全局模型 {borrowed_acc:.4f}，"
        f"差 {borrowed_acc - raw_acc:+.4f}。",
        "",
    ]
    if raw_acc < 0.2:
        lines += [
            "- ❌ **原始 `global.pt` 已接近随机水平**，不能作为实验 F 的评估对象"
            "——整个矩阵会淹没在地板噪声里。主实验用个性化模型，"
            "副实验用借-BN 全局模型。",
            "- **回溯影响**：实验 E 的 E4 正是用 `attack_dir/global.pt` 作黑盒 ξ "
            "模型（`exp_e.py:406`），其 ASR=0.0312。在这个 clean accuracy 下，"
            "该数字主要反映的是**模型失配**而非黑盒迁移困难，"
            "**E4 的结论应作废并用借-BN 全局模型重测**。",
            "",
        ]
    else:
        lines += [
            f"- 原始全局模型的 clean accuracy 为 {raw_acc:.4f}，并未退化到随机水平。"
            "BN 停在初始化仍使其与各客户端失配，但程度有限；"
            "两种全局口径都保留，在结果中并列报告。",
            "",
        ]
    return frame, "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 重跑成本
# ---------------------------------------------------------------------------
def estimate_rerun_cost(ckpt_dir: Path, cfg: Cfg) -> str:
    meta = load_meta(ckpt_dir / "meta.json")
    total_round = int(meta.get("total_round", 0))
    every = int(cfg.exp_f.snapshot_every)
    grid = build_grid(total_round, every)
    n_clients = (int(cfg.exp_f.snapshot_n_benign)
                 + int(cfg.exp_f.snapshot_n_malicious))

    global_pt = ckpt_dir / "global.pt"
    model_mb = global_pt.stat().st_size / 1e6 if global_pt.exists() else 0.0
    gen_pt = ckpt_dir / "generator.pt"
    gen_mb = gen_pt.stat().st_size / 1e6 if gen_pt.exists() else 0.0

    total_mb = len(grid) * (model_mb * (1 + n_clients) + gen_mb)
    return "\n".join([
        "### 4. 重跑成本估算", "",
        f"- `total_round` = {total_round}，快照间隔 {every} → **{len(grid)} 个网格点**",
        f"- 每个网格点存：全局模型 1 份 + 客户端模型 {n_clients} 份 + 生成器 1 份",
        f"- 单份模型 {model_mb:.1f} MB，生成器 {gen_mb:.1f} MB",
        f"- **存储增量 ≈ {total_mb / 1000:.2f} GB**",
        f"- 墙钟时间 ≈ 1× 原 attack run（埋点只做 I/O，不消耗 RNG、"
        f"不触碰 optimizer、不做前向）",
        "",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 F Phase 0 核查")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="跳过第 3 项（需要前向推理与数据集）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ckpt_root = Path(args.ckpt_root)
    device = resolve_device(args.device)
    attack_dir = ckpt_root / run_dir_name("attack", args.alpha, args.seed)

    sections = [
        "# 实验 F — Phase 0 核查报告", "",
        f"- checkpoint 根目录：`{ckpt_root}`",
        f"- 主 run：`{attack_dir}`",
        f"- 设备：`{device}`", "",
        "> 先决事实（代码级确定，无需数据）：**按轮次的模型快照在本仓库中从未",
        "> 实现过**。`hooks.save_run` 只在 `run_fl.py:324` 调用一次，官方仓库",
        "> 里 `torch.save` 一次都没出现。因此"
        "\"是否需要重跑\"这个分支不依赖本报告。", "",
    ]

    _, text = inventory_snapshots(ckpt_root)
    sections.append(text)

    if (attack_dir / "global.pt").exists():
        _, _, text = diagnose_global_bn(attack_dir / "global.pt")
        sections.append(text)
        if not args.skip_accuracy:
            try:
                _, text = measure_clean_acc(attack_dir, cfg, device)
                sections.append(text)
            except Exception as exc:  # noqa: BLE001
                sections += [
                    "### 3. 三种评估对象的 clean accuracy", "",
                    f"- ⚠️ **未能确定**：{type(exc).__name__}: {exc}", ""]
        sections.append(estimate_rerun_cost(attack_dir, cfg))
    else:
        sections += [f"### 2-4", "",
                     f"- ⚠️ `{attack_dir / 'global.pt'}` 不存在，"
                     f"**未能确定**。", ""]

    report = "\n".join(sections)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
