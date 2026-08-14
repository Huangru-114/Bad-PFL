"""实验 E 的 Phase 0 校验：跑正式实验之前必须全部通过。

检查项（任何一项失败即以非零码退出）：

1. **干净 run 的分支是否干净** —— 确认 ``bad_client_num=0`` 这条路不会
   注册任何攻击钩子。原实现 ``fba.use_our_attack`` 在没有 ``PoisonClient``
   时会抛 ``UnboundLocalError``（``eval_func`` 从未赋值，fba.py:60-66），
   所以正确做法是**完全跳过** ``use_our_attack``，而不是构造空的攻击对象。
   本脚本实测这个崩溃，把它变成有据可查的事实而不是传说。

2. **划分一致性** —— 对每个 ``(alpha, seed)`` 跑
   ``verify_partition_consistency``。返回 False 即中止：clean/attack 的划分
   一旦错位，E2 拿 attack 的生成器去打 clean 的模型就不再是同一批数据上的
   对照，整个"对抗地板"的分解失去意义。

3. **E1/E2 的可行性盘点** —— 检查现有 checkpoint 是否已经足够跑 E1/E2
   （需要 clean 模型 + attack 的 ``generator.pt``）。若齐全，**P2 不需要
   任何重新训练**。

4. **参数哈希工具可用** —— E3 要在干净模型上现训生成器，必须证明
   ``clean_model`` 前后未被改动（陷阱清单第 7 条）。这里验证
   ``state_dict_hash`` 对同一模型稳定、对改动敏感。

用法::

    python -m diag.exp_e_precheck --ckpt-root ./checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from . import REPO_ROOT  # noqa: F401
from .config import load_config
from .hooks import (load_meta, state_dict_hash, verify_partition_consistency)

__all__ = ["check_clean_branch", "check_partition_consistency",
           "check_e1_e2_feasibility", "check_hash_tool", "main"]


def _pass(ok: bool) -> str:
    return "通过" if ok else "**未通过**"


# ---------------------------------------------------------------------------
# 1. 干净分支
# ---------------------------------------------------------------------------
def check_clean_branch() -> Tuple[bool, str]:
    """实测：没有 PoisonClient 时 ``use_our_attack`` 会崩。"""
    from functools import partial

    from torch.utils.data import DataLoader, TensorDataset

    from client import BasicClient
    from fba import use_our_attack
    from resnet import get_resnet
    from server import BasicServer

    dataset = TensorDataset(torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = DataLoader(dataset, batch_size=4)
    clients = [BasicClient(get_resnet(size=10), loader, loader,
                           nn.CrossEntropyLoss(), partial(torch.optim.SGD, lr=0.1))
               for _ in range(2)]
    for client in clients:
        client.local_model.device = torch.device("cpu")
    server = BasicServer(get_resnet(size=10))
    server.global_model.device = torch.device("cpu")

    lines = ["### 1. 干净 run 的分支", ""]
    try:
        use_our_attack(clients, server, 0, 0.2)
    except UnboundLocalError as exc:
        lines += [
            "- 实测：`use_our_attack` 在没有 `PoisonClient` 时抛 "
            f"`UnboundLocalError`（{exc}）。",
            "- 成因：`fba.py:60-66` 只在遇到 Poison 类客户端时才给 `eval_func` 赋值。",
            "- 处理：`diag/run_fl.py` 的 clean 模式**完全跳过** `use_our_attack`，"
            "不构造空的攻击对象 —— 与 spec §2.2 的要求一致。", ""]
        return True, "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        lines += [f"- ⚠️ 抛出了非预期的异常：`{type(exc).__name__}: {exc}`，"
                  "与 spec 的描述不符，需人工确认。", ""]
        return False, "\n".join(lines)

    lines += ["- ⚠️ **没有崩溃** —— 与 spec 描述不符。可能 `fba.py` 已被改动，"
              "需人工确认干净 run 是否真的不会注册攻击钩子。", ""]
    return False, "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. 划分一致性
# ---------------------------------------------------------------------------
def _discover_pairs(ckpt_root: Path) -> List[Tuple[float, int]]:
    from .audit.common import discover_runs

    runs = discover_runs(ckpt_root)
    if runs.empty:
        return []
    pairs = set()
    for _, row in runs.iterrows():
        pairs.add((float(row["alpha"]), int(row["seed"])))
    return sorted(pairs, key=lambda p: (-p[0], p[1]))


def check_partition_consistency(ckpt_root: Path) -> Tuple[bool, str]:
    from .hooks import run_dir_name

    lines = ["### 2. clean / attack 的划分一致性", ""]
    pairs = _discover_pairs(ckpt_root)
    if not pairs:
        lines += [f"- ⚠️ 在 `{ckpt_root}` 下没发现任何 run，**未能确定**。", ""]
        return False, "\n".join(lines)

    all_ok, rows = True, []
    for alpha, seed in pairs:
        clean = ckpt_root / run_dir_name("clean", alpha, seed) / "meta.json"
        attack = ckpt_root / run_dir_name("attack", alpha, seed) / "meta.json"
        if not clean.exists() or not attack.exists():
            rows.append(f"| {alpha} | {seed} | 跳过 | 缺少 "
                        f"{'clean' if not clean.exists() else 'attack'} 侧的 meta.json |")
            continue
        ok, report = verify_partition_consistency(clean, attack)
        all_ok = all_ok and ok
        detail = "逐客户端（含索引级）一致" if ok else \
            report.splitlines()[5] if len(report.splitlines()) > 5 else "见日志"
        rows.append(f"| {alpha} | {seed} | {_pass(ok)} | {detail} |")
        if not ok:
            print(report)

    lines += ["| alpha | seed | 结果 | 说明 |", "|---|---|---|---|"] + rows + [""]
    if not all_ok:
        lines += ["- ❌ 存在不一致的配对。**必须中止** —— clean/attack 划分错位后，"
                  "E2 就不再是同一批数据上的对照，对抗地板的分解无意义。", ""]
    return all_ok, "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. E1/E2 可行性
# ---------------------------------------------------------------------------
def check_e1_e2_feasibility(ckpt_root: Path) -> Tuple[bool, str]:
    from .hooks import run_dir_name

    lines = ["### 3. E1 / E2 是否已经可以跑（无需重训）", ""]
    pairs = _discover_pairs(ckpt_root)
    if not pairs:
        lines += ["- ⚠️ 没有 run，**未能确定**。", ""]
        return False, "\n".join(lines)

    rows, ready = [], 0
    for alpha, seed in pairs:
        clean_dir = ckpt_root / run_dir_name("clean", alpha, seed)
        attack_dir = ckpt_root / run_dir_name("attack", alpha, seed)
        has_clean = (clean_dir / "global.pt").exists()
        has_attack = (attack_dir / "global.pt").exists()
        has_gen = (attack_dir / "generator.pt").exists()
        n_clean = len(list(clean_dir.glob("client_*.pt")))
        n_attack = len(list(attack_dir.glob("client_*.pt")))
        ok = has_clean and has_attack and has_gen and n_clean > 0 and n_attack > 0
        ready += int(ok)
        rows.append(f"| {alpha} | {seed} | {n_attack} | {n_clean} | "
                    f"{'有' if has_gen else '**无**'} | {_pass(ok)} |")

    lines += ["| alpha | seed | attack 客户端 ckpt | clean 客户端 ckpt | "
              "generator.pt | E1/E2 就绪 |",
              "|---|---|---|---|---|---|"] + rows + ["",
              f"- {ready}/{len(pairs)} 个 (alpha, seed) 已具备 E1/E2 所需的全部产物。", ""]
    if ready:
        lines += ["- ✅ **E1/E2 不需要重新训练** —— 现有 checkpoint 里已有 "
                  "clean 模型和 attack run 的最终生成器（陷阱清单第 6 条要求用 "
                  "`final`，正是这一份）。P2 的成本只有前向推理。", ""]
    return ready > 0, "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 参数哈希工具
# ---------------------------------------------------------------------------
def check_hash_tool() -> Tuple[bool, str]:
    from resnet import get_resnet

    model = get_resnet(size=10)
    first = state_dict_hash(model)
    second = state_dict_hash(model)
    stable = first == second

    with torch.no_grad():
        model.linear.weight[0, 0] += 1e-6
    sensitive = state_dict_hash(model) != first

    lines = ["### 4. 参数哈希工具（E3 的污染检测）", "",
             f"- 同一模型两次哈希一致：{_pass(stable)}（`{first[:16]}…`）",
             f"- 对 1e-6 的权重改动敏感：{_pass(sensitive)}", "",
             "- E3 现训生成器时，将在训练前后比对该哈希；不一致即报错中止"
             "（陷阱清单第 7 条）。", ""]
    return stable and sensitive, "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 E Phase 0 校验")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--out", default=None, help="把报告写到文件")
    parser.add_argument("--skip-partition", action="store_true",
                        help="跳过划分一致性检查（没有 checkpoint 时用）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ckpt_root = Path(args.ckpt_root)

    sections = ["# 实验 E — Phase 0 校验报告", "",
                f"- checkpoint 根目录：`{ckpt_root}`",
                f"- 目标类：{int(cfg.data.target_class)}",
                f"- epsilon (xi) = {float(cfg.perturb.eps_xi):.8f}，"
                f"sigma (delta) = {float(cfg.perturb.eps_delta):.8f}",
                f"- 生成器 checkpoint 间隔：每 "
                f"{int(cfg.exp_e.generator_ckpt_every)} 轮", ""]

    results: Dict[str, bool] = {}
    for name, func in [("干净分支", check_clean_branch),
                       ("参数哈希", check_hash_tool)]:
        ok, text = func()
        results[name] = ok
        sections.append(text)

    if args.skip_partition:
        sections += ["### 2. clean / attack 的划分一致性", "",
                     "- 已按 `--skip-partition` 跳过。", ""]
    else:
        ok, text = check_partition_consistency(ckpt_root)
        results["划分一致性"] = ok
        sections.append(text)

    ok, text = check_e1_e2_feasibility(ckpt_root)
    results["E1/E2 就绪"] = ok
    sections.append(text)

    failed = [k for k, v in results.items() if not v]
    sections += ["## 结论", "",
                 "| 检查项 | 结果 |", "|---|---|"]
    sections += [f"| {k} | {_pass(v)} |" for k, v in results.items()]
    sections += ["", ("✅ Phase 0 全部通过，可以进入 P1。" if not failed
                      else f"❌ 未通过：{', '.join(failed)}。**不要进入 P1。**"), ""]

    report = "\n".join(sections)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
