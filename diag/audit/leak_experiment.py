"""梯度泄漏的受控 A/B 对照重训。

# 要回答的问题

`l2_to_median` 完美分离，是因为**实现缺陷**（PGD 梯度泄漏进参数更新），
还是因为**投毒目标本身**就让更新方向不同？两者的后续处理完全不同：
前者是"我们的实验有问题，baseline 不公平"，后者是"该攻击本来就不隐蔽"。

# 设计

两臂使用**完全相同**的 seed / 划分 / 初始化，唯一差别是是否消除泄漏：

- **Arm A（原样）**：`client.py:34` 的 `zero_grad()` 早于 `client.py:36` 的
  `fetch_data()`，因此 `pgd_attack`（`fba.py:16`）累加进参数的梯度
  会被 `scaler.step()` 一并使用。
- **Arm B（去泄漏）**：monkey-patch `PoisonClient.fetch_data`，取数后立刻
  `local_model.zero_grad()`，精确移除这一份梯度，其余一切不变。

**两臂都仍然投毒**，所以：

    A 与 B 之差            = 泄漏效应
    B 内部的恶意 vs 良性差 = 投毒目标本身的效应

# 重要限制

规模远小于正式配置（默认 20 客户端 / 30 轮）。它只能回答"泄漏是否产生**可测的**
更新范数差异"，**不能外推**到 100 客户端 / 300 轮的绝对量级。

monkey-patch 只作用于本模块运行期间，`client.py` / `fba.py` **零改动**。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..config import Cfg, load_config
from ..hooks import flatten_state_dict, load_checkpoint, load_meta
from .common import THRESHOLDS, AuditContext, Verdict, group_compare, md_kv, md_table

__all__ = ["run", "leak_free_poison_client", "main"]

DEFAULTS = {"client_num": 20, "bad_client_num": 4, "select_per_round": 10,
            "total_round": 30, "local_steps": 15, "alpha": 0.5, "seed": 0}


@contextlib.contextmanager
def leak_free_poison_client():
    """临时移除 PGD 梯度泄漏（Arm B）。

    ``PoisonClient.fetch_data``（client.py:124-125）内部会经由
    ``our_poison_func`` → ``pgd_attack`` → ``loss.backward()``（fba.py:16）
    把梯度累加进 ``local_model`` 的参数。取数后立刻清零，正好移除这一份、
    且不影响随后正常的任务梯度。

    仅在 with 块内生效，退出即还原 —— 原仓库文件不受任何影响。
    """
    import client as client_module

    original = client_module.PoisonClient.fetch_data

    def patched(self):
        data = original(self)
        self.local_model.zero_grad(set_to_none=True)
        return data

    client_module.PoisonClient.fetch_data = patched
    try:
        yield
    finally:
        client_module.PoisonClient.fetch_data = original


def _arm_cfg(cfg: Cfg, synthetic: bool, overrides: Dict[str, Any]) -> Cfg:
    """复制配置并缩小规模。``synthetic=True`` 走合成数据（不需要 torchvision）。"""
    small = Cfg(copy.deepcopy(dict(cfg)))
    section = "smoke" if synthetic else "fl"
    if synthetic:
        small["smoke"].update({
            "num_clients": overrides["client_num"],
            "bad_client_num": overrides["bad_client_num"],
            "select_per_round": overrides["select_per_round"],
            "rounds": overrides["total_round"],
            "local_steps": overrides["local_steps"],
            "samples_per_client": 128,
            "test_samples_per_client": 128,
        })
    else:
        small["fl"].update({
            "client_num": overrides["client_num"],
            "bad_client_num": overrides["bad_client_num"],
            "select_per_round": overrides["select_per_round"],
            "total_round": overrides["total_round"],
            "local_steps": overrides["local_steps"],
        })
    return small


def _measure(run_dir: Path) -> pd.DataFrame:
    """从一臂的 checkpoint 算出每客户端的更新范数与 l2_to_median。"""
    meta = load_meta(run_dir / "meta.json")
    global_state = load_checkpoint(run_dir / "global.pt")

    updates, rows = [], []
    for record in meta["clients"]:
        state = load_checkpoint(run_dir / f"client_{int(record['client_id'])}.pt")
        updates.append(flatten_state_dict(state, reference=global_state))
        rows.append({"client_id": int(record["client_id"]),
                     "is_malicious": bool(record["is_malicious"]),
                     "n_participations": int(record.get("n_participations", 0))})

    stacked = torch.stack(updates, dim=0)
    median = stacked.median(dim=0).values
    frame = pd.DataFrame(rows)
    frame["update_l2_to_global"] = [float(u.norm().item()) for u in updates]
    frame["l2_to_median"] = [float((u - median).norm().item()) for u in updates]
    return frame


def run(ctx: AuditContext, synthetic: bool = False,
        overrides: Optional[Dict[str, Any]] = None,
        work_dir: Optional[Path] = None) -> str:
    from ..run_fl import run_fl

    overrides = {**DEFAULTS, **(overrides or {})}
    alpha, seed = float(overrides["alpha"]), int(overrides["seed"])
    work_dir = Path(work_dir or (ctx.evidence_dir / "leak_arms"))
    small = _arm_cfg(ctx.cfg, synthetic, overrides)

    scaler = torch.cuda.amp.GradScaler()
    amp = {"cuda 可用": torch.cuda.is_available(),
           "GradScaler.is_enabled()": scaler.is_enabled(),
           "初始 scale S": float(scaler.get_scale()) if scaler.is_enabled() else 1.0}

    sections = ["## 受控对照：梯度泄漏是否造成更新范数差异", "",
                "### 设置", "",
                md_kv({**{f"配置.{k}": v for k, v in overrides.items()},
                       "数据": "合成（synthetic）" if synthetic else "CIFAR-10",
                       **amp}), "",
                "两臂 seed / 划分 / 初始化完全相同，唯一差别是是否移除 "
                "`pgd_attack` 泄漏进参数的那份梯度。**两臂都仍然投毒。**", ""]

    if amp["GradScaler.is_enabled()"]:
        sections += [
            f"⚠️ `GradScaler` 处于 **enabled** 状态（S≈{amp['初始 scale S']:.0f}）。"
            "任务梯度被放大 S 倍后在 `scaler.step` 整体除回，而 PGD 那份梯度是"
            "**未放大**的，会被除成 ∇/S ≈ 0。**预期泄漏在此环境下几乎不起作用**，"
            "两臂应当看不出差别 —— 如果确实看不出，说明 l2 的分离来自投毒目标本身，"
            "而不是这个实现缺陷。", ""]
    else:
        sections += [
            "⚠️ `GradScaler` 处于 **disabled** 状态（S=1，CPU 路径）。"
            "泄漏是全强度的，预期两臂会有可见差别。", ""]

    # --- 跑两臂 ---
    arms: Dict[str, pd.DataFrame] = {}
    for arm, patcher in (("A_原样", contextlib.nullcontext),
                         ("B_去泄漏", leak_free_poison_client)):
        arm_root = work_dir / arm
        print(f"[leak] 开始 Arm {arm} ...")
        with patcher():
            run_dir = run_fl(small, "attack", alpha, seed, smoke=synthetic,
                             device=str(ctx.device), ckpt_root=arm_root)
        frame = _measure(Path(run_dir))
        frame["arm"] = arm
        arms[arm] = frame
        print(f"[leak] Arm {arm} 完成 -> {run_dir}")

    combined = pd.concat(arms.values(), ignore_index=True)
    ctx.save_evidence(combined, "leak_experiment_per_client")

    # --- 对比 ---
    rows = []
    for arm, frame in arms.items():
        for column in ("update_l2_to_global", "l2_to_median"):
            stats = group_compare(frame[column], frame["is_malicious"], column)
            stats["arm"] = arm
            rows.append(stats)
    comparison = pd.DataFrame(rows)
    ctx.save_evidence(comparison, "leak_experiment_groups")

    sections += ["### 结果", "", md_table(comparison[[
        "arm", "signal", "benign_mean", "malicious_mean", "benign_max",
        "malicious_min", "separation_gap", "separated", "auc",
        "mannwhitney_p"]]), ""]

    def _auc(arm: str, signal: str) -> float:
        block = comparison[(comparison["arm"] == arm)
                           & (comparison["signal"] == signal)]
        return float(block["auc"].iloc[0]) if len(block) else float("nan")

    auc_a = _auc("A_原样", "l2_to_median")
    auc_b = _auc("B_去泄漏", "l2_to_median")
    ratio_a = float(np.nanmean(arms["A_原样"].loc[
        arms["A_原样"]["is_malicious"], "update_l2_to_global"]) /
        np.nanmean(arms["A_原样"].loc[
            ~arms["A_原样"]["is_malicious"], "update_l2_to_global"]))
    ratio_b = float(np.nanmean(arms["B_去泄漏"].loc[
        arms["B_去泄漏"]["is_malicious"], "update_l2_to_global"]) /
        np.nanmean(arms["B_去泄漏"].loc[
            ~arms["B_去泄漏"]["is_malicious"], "update_l2_to_global"]))

    delta = auc_a - auc_b
    if not np.isfinite(delta):
        conclusion, action = "未能确定", "两臂的 AUC 无法计算，检查上表"
    elif abs(delta) < 0.05 and auc_b >= THRESHOLDS["perfect_auc"]:
        conclusion = "泄漏不是原因；分离来自投毒目标本身"
        action = ("l2=1.000 应表述为『该攻击未做隐蔽性优化』，"
                  "而非实现缺陷。但对比公平性问题仍然存在（见任务 1.5）")
    elif abs(delta) < 0.05:
        conclusion = "泄漏不是主要原因"
        action = "去掉泄漏后 AUC 基本不变，需另找分离来源"
    else:
        conclusion = "泄漏有实质贡献"
        action = ("l2 baseline 受实现缺陷污染，"
                  "实验 C 的 baseline 需在去泄漏条件下重跑")

    ctx.add(Verdict(
        item="梯度泄漏是否造成 l2 完美分离",
        conclusion=conclusion,
        rule="Arm A 与 Arm B 的 l2_to_median AUC 之差 |ΔAUC| >= 0.05 视为有实质贡献",
        measured=f"AUC(A)={auc_a:.3f}, AUC(B)={auc_b:.3f}, Δ={delta:.3f}；"
                 f"恶意/良性更新范数比 A={ratio_a:.3f}, B={ratio_b:.3f}",
        action=action))

    sections += [
        "### 判读", "",
        f"- 恶意/良性的平均更新范数比：Arm A = **{ratio_a:.3f}**，"
        f"Arm B = **{ratio_b:.3f}**",
        f"- `l2_to_median` 的 AUC：Arm A = **{auc_a:.3f}**，Arm B = **{auc_b:.3f}**，"
        f"Δ = **{delta:.3f}**",
        "",
        "**分解**：A 与 B 之差 = 泄漏效应；B 内部恶意 vs 良性之差 = 投毒目标本身的效应。",
        "",
        f"⚠️ 本对照为 {overrides['client_num']} 客户端 / {overrides['total_round']} 轮，"
        "不是正式配置的复现，只能回答定性问题，不能外推绝对量级。", ""]

    return "\n".join(sections)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="梯度泄漏受控对照")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="用合成数据（不需要 torchvision），便于本地验证")
    parser.add_argument("--work-dir", default=None)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value),
                            default=value)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    from ..exp_common import resolve_device
    device = resolve_device(args.device)
    evidence = Path("diag/audit/evidence")
    ctx = AuditContext(ckpt_root=Path("."), raw_dir=Path("."),
                       evidence_dir=evidence, cfg=cfg, device=device)
    overrides = {key: getattr(args, key) for key in DEFAULTS}
    print(run(ctx, synthetic=args.synthetic, overrides=overrides,
              work_dir=args.work_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
