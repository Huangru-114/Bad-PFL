"""T3：残留后门是被**幅度**打掉的，还是被**方向**打掉的？

# 逻辑

T0 实测（`PLAN_T0T4.md §9`）：攻击停止后的 200 轮干净训练把**实际参与聚合的
参数**移动了相对幅度 **0.156**，方向与 θ 近似正交，是近正交增量的随机游走
—— 这么大的改写都没把 ASR 打下 0.41 的地板。

T3 在**漂移之后**的模型上，沿不同方向再走同样一段，看谁能把残留打掉。

# ⚠️ 锚点语义（首跑后必须一次说清，初版在这里含糊过）

默认锚点是 run 结束时的**个性化客户端模型**（r=400），也就是漂移**已经走完**
的那个点。于是：

- ``zero`` 格 = 那 200 轮良性漂移的**终点本身**，它的 ASR 就是地板；
- ``real`` 格 = 沿同一方向**再外推**一段，**不是**复现那 200 轮。

所以 T3 回答的是「**继续沿良性方向走** vs **随便走**，哪个更能打掉残留」，
**不是**「那 200 轮为什么没打掉」。`--anchor-round 200` 可以把锚点换成刚植入
完的模型（代价：按轮次的客户端快照只覆盖被选中的约 10 个良性客户端，且带
staleness）。

# ⚠️ 必须按**等 ACC 代价**比，不能只按等位移幅度（首跑暴露的方法论问题）

1100 万维里随机方向几乎与损失面所有"陡"的方向正交，所以"随机扰动范数 0.078
却几乎不改变函数"近乎同义反复。首跑数据直接证实：同样 0.078 的位移，随机族
ACC 只掉 0.002–0.008，real 掉 0.047 —— 两者根本没在同一个"函数改变量"上比较。
`iso_acc_comparison` 因此把各族插值到同一个 ACC 代价再比，判词以它为准。

# 六个方向族（都只作用在**实际参与聚合**的参数上）

| 族 | 构造 | 它控制掉什么 |
|---|---|---|
| `zero` | 不动 | **自检锚点**：必须复现未扰动的 ASR/ACC，分毫不差 |
| `gaussian_global` | 各向同性高斯，只匹配**全局**范数 | 能量按层的大小分配，逐层剖面**不**匹配 |
| `gaussian_layer_matched` | 逐层各向同性，**逐层**相对位移匹配真实漂移 | 公平对照：同幅度、同逐层剖面、方向随机 |
| `shuffled` | 真实 Δθ 在**层内**随机置换坐标 | 保住幅度的边缘分布，打掉"是哪些坐标" |
| `sign_flipped` | 真实 Δθ 逐坐标随机翻符号 | 保住幅度**与坐标**，只打掉方向的相干性 |
| `real` | 真实 Δθ 本身 | 正对照：multiplier=1 = 沿良性方向再外推一整段 |

`gaussian_layer_matched` 是主对照；`shuffled` / `sign_flipped` 把"幅度"与
"坐标身份"、"方向相干性"逐层剥开，比单纯的高斯更能定位。

# 必须与 ASR 一起报 ACC —— 否则整个实验没有意义

把模型打坏也能让 ASR 掉。所以每个格子都同时评 `asr` 与 `acc`，判词在 ACC 掉
超过 `ACC_GUARD` 时拒绝对该幅度下的任何 ASR 结论，并且**主判据本身就建立在
等 ACC 代价上**。这不是保守，是这个实验唯一能立住的读法。

# 口径

- 扰动只加在 **`aggregated` 参数**（weight + bias）上。FedBN 下 BN 仿射与 buffer
  在全局模型里恒不更新（T0 实测位移**恰好为 0**），扰动它们等于做训练根本不会做的事。
- 评估在**每个客户端自己的个性化模型**上，各自保留自己的 BN，用
  `recompute_asr_final.client_asr` 的**原始触发器**口径 —— 与 B2 那条 0.4133
  的曲线同口径，可以直接比。
- 幅度基准 `--base-relative` 默认 **0.156**（T0 实测值），multiplier 扫
  `0.5 / 1 / 2 / 4`。**基准写在命令行里，不在代码里偷偷改**。

# 用法

    # 1) CPU：标定与清单（不需要 GPU，不需要数据）
    python -m diag.exp_t3 --mode build \\
        --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \\
        --drift-from 200 --drift-to 400 --out-dir results/t3

    # 2) GPU：真评（先看 dry-run 的格子数，再加 --execute）
    python -m diag.exp_t3 --mode eval \\
        --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \\
        --data-root ./data --model-size 18 --device 0 \\
        --out-dir results/t3 --execute

    # 3) 换锚点到刚植入完的模型（客户端数会掉到约 10 个，见上）
    python -m diag.exp_t3 --mode eval --anchor-round 200 \\
        --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \\
        --data-root ./data --model-size 18 --device 0 \\
        --out-dir results/t3_r200 --execute
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paramspace as ps
from .exp_t0 import AGGREGATED_KINDS, global_path, load_state, write_rows

__all__ = ["FAMILIES", "ACC_GUARD", "make_direction", "scale_to_relative",
           "perturbation_profile", "build_recipes", "apply_perturbation",
           "recipe_key", "load_drift", "calibrate", "eval_clients",
           "missing_eval_inputs", "iso_acc_comparison", "RANDOM_FAMILIES",
           "append_row", "load_raw_rows",
           "load_done_keys",
           "evaluate_recipes", "aggregate_results", "flatness_verdict", "main"]

#: 方向族。顺序即报告顺序；``zero`` 放第一个，因为它是自检锚点。
FAMILIES: Tuple[str, ...] = ("zero", "gaussian_global",
                             "gaussian_layer_matched", "shuffled",
                             "sign_flipped", "real")

#: ACC 相对未扰动基线掉超过这个绝对值，该幅度下的 ASR 结论一律作废。
#: **先写死再看数据**（`PLAN_T0T4.md` 坑 8）。
ACC_GUARD = 0.05

#: multiplier 默认扫描点。1.0 = T0 实测的那 200 轮漂移幅度。
DEFAULT_MULTIPLIERS: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# 方向构造（纯 numpy）
# ---------------------------------------------------------------------------
def _layer_slices(index: ps.ParamIndex) -> Dict[str, np.ndarray]:
    labels = np.asarray(index.group_labels("layer"))
    return {label: (labels == label) for label in sorted(set(labels.tolist()))}


def make_direction(family: str, delta_real: np.ndarray, theta: np.ndarray,
                   index: ps.ParamIndex, seed: int) -> np.ndarray:
    """构造一个**未定标**的方向向量（定标交给 ``scale_to_relative``）。

    ``delta_real`` / ``theta`` / ``index`` 都必须已经限制在同一个子空间上
    （本模块里是 ``aggregated``）。随机族用独立的 ``RandomState``，
    **不碰全局 RNG**（与 `snapshots.select_snapshot_clients` 同一条规矩）。
    """
    delta_real = np.asarray(delta_real, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if delta_real.shape != theta.shape:
        raise ValueError(f"形状不一致：{delta_real.shape} vs {theta.shape}")
    if delta_real.size != index.n_params:
        raise ValueError(
            f"向量长度 {delta_real.size} 与索引的 {index.n_params} 不一致 —— "
            f"索引与向量不是同一个掩码下的产物")
    rng = np.random.RandomState(int(seed))

    if family == "zero":
        return np.zeros_like(delta_real)
    if family == "real":
        return delta_real.copy()
    if family == "gaussian_global":
        return rng.standard_normal(delta_real.size)
    if family == "sign_flipped":
        signs = rng.randint(0, 2, size=delta_real.size).astype(np.float64)
        return delta_real * (2.0 * signs - 1.0)

    masks = _layer_slices(index)
    out = np.zeros_like(delta_real)
    if family == "shuffled":
        # 层内置换：保住每层的范数与幅度边缘分布，打掉"是哪些坐标"
        for mask in masks.values():
            block = delta_real[mask]
            out[mask] = block[rng.permutation(block.size)]
        return out
    if family == "gaussian_layer_matched":
        # 逐层各向同性，且每层的 ‖·‖ 与真实漂移该层的 ‖Δθ‖ 相同
        for mask in masks.values():
            noise = rng.standard_normal(int(mask.sum()))
            norm = np.linalg.norm(noise)
            target = np.linalg.norm(delta_real[mask])
            if norm > 0 and target > 0:
                out[mask] = noise * (target / norm)
        return out
    raise ValueError(f"未知方向族 '{family}'；可选 {FAMILIES}")


def scale_to_relative(direction: np.ndarray, theta: np.ndarray,
                      target_relative: float) -> np.ndarray:
    """把方向定标到 ``‖v‖ = target_relative · ‖θ‖``。

    零方向（``zero`` 族）原样返回 —— 定标一个零向量没有意义，也不该报错。
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return direction
    return direction * (float(target_relative) * ps.l2(theta) / norm)


def perturbation_profile(vector: np.ndarray, theta: np.ndarray,
                         delta_real: np.ndarray, index: ps.ParamIndex
                         ) -> Dict[str, Any]:
    """标定结果的可审计读数：到底加了多大、加在哪、跟真实漂移多像。

    ``layer_profile_max_rel_dev`` 回答"**形状**跟真实漂移一样吗"，而且是
    **无量纲**的，可以跨 multiplier 比：

        ratio_L = 该层的相对位移 / 真实漂移该层的相对位移
        dev     = max_L | ratio_L / ratio_global − 1 |

    逐层剖面被整体放大 k 倍时每个 ``ratio_L`` 都等于 ``ratio_global``，dev 恒为 0
    —— 所以 `gaussian_layer_matched` / `shuffled` / `sign_flipped` / `real`
    在**任何** multiplier 上都应当是 0，而 `gaussian_global`（能量按层的大小分配）
    不会。**直接拿绝对差是错的**：那样 multiplier=2 会把一个形状完美的扰动
    也报成偏差很大。
    """
    vector = np.asarray(vector, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    delta_real = np.asarray(delta_real, dtype=np.float64)
    masks = _layer_slices(index)

    ratio_global = np.nan
    want_global = ps.relative_displacement(delta_real, theta)
    got_global = ps.relative_displacement(vector, theta)
    if np.isfinite(want_global) and want_global > 0:
        ratio_global = got_global / want_global

    deviations = []
    if np.isfinite(ratio_global) and ratio_global > 0:
        for mask in masks.values():
            got = ps.relative_displacement(vector[mask], theta[mask])
            want = ps.relative_displacement(delta_real[mask], theta[mask])
            if np.isfinite(got) and np.isfinite(want) and want > 0:
                deviations.append(abs((got / want) / ratio_global - 1.0))

    return {
        "achieved_relative_displacement": got_global,
        "l2_perturbation": ps.l2(vector),
        "cos_with_real_drift": ps.cosine(vector, delta_real),
        "cos_with_theta": ps.cosine(vector, theta),
        "layer_profile_max_rel_dev": (float(max(deviations)) if deviations
                                      else float("nan")),
        "n_layers": len(masks),
    }


def build_recipes(families: Sequence[str] = FAMILIES,
                  multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
                  seeds: Sequence[int] = (0, 1, 2),
                  base_relative: float = 0.156) -> List[Dict[str, Any]]:
    """扰动配方表。**不存扰动向量本身**，只存"怎么造出来"。

    11M 个坐标 × float32 ≈ 45MB 一份，六族 × 四幅度 × 三 seed 会到 3GB。
    配方 + 固定 seed 就能确定性地重建，落盘的是**标定读数**（可审计），不是向量。

    ``zero`` 族只出一格（幅度与 seed 对它没有意义），它是自检锚点。
    确定性族（``real``）也只出一格 per multiplier —— 多个 seed 会得到同一个向量，
    重复评估只是浪费机时。
    """
    deterministic = {"zero", "real"}
    recipes: List[Dict[str, Any]] = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"未知方向族 '{family}'；可选 {FAMILIES}")
        if family == "zero":
            recipes.append({"family": "zero", "multiplier": 0.0, "seed": 0,
                            "target_relative": 0.0})
            continue
        for multiplier in multipliers:
            used = (0,) if family in deterministic else tuple(seeds)
            for seed in used:
                recipes.append({
                    "family": family, "multiplier": float(multiplier),
                    "seed": int(seed),
                    "target_relative": float(multiplier) * float(base_relative),
                })
    return recipes


def apply_perturbation(state: Dict[str, np.ndarray], vector: np.ndarray,
                       index: ps.ParamIndex) -> Dict[str, np.ndarray]:
    """把扰动加回 state_dict。**只碰索引里有的键**，其余原样带过。

    "其余原样带过"是关键：BN 仿射与 buffer 不在 ``aggregated`` 索引里，于是
    每个客户端自己的 BN 被完整保留 —— 这正是 FedBN 下该有的行为。
    """
    patch = ps.unflatten(vector, index)
    out: Dict[str, np.ndarray] = {}
    for key, value in state.items():
        array = np.asarray(value)
        if key in patch:
            out[key] = (array.astype(np.float64) + patch[key]).astype(
                array.dtype if np.issubdtype(array.dtype, np.floating)
                else np.float32)
        else:
            out[key] = array
    return out


# ---------------------------------------------------------------------------
# 判词
# ---------------------------------------------------------------------------
RANDOM_FAMILIES: Tuple[str, ...] = ("gaussian_global",
                                    "gaussian_layer_matched", "shuffled",
                                    "sign_flipped")


def iso_acc_comparison(rows: Sequence[Dict[str, Any]],
                       acc_cost: float = ACC_GUARD) -> Dict[str, Any]:
    """**按等 ACC 代价**比各族的 ASR，而不是按等位移幅度。

    # 为什么必须这么比（首跑暴露出来的方法论问题）

    等位移幅度的比较有一个致命混杂：**在 1100 万维里，随机方向几乎与损失面所有
    "陡"的方向正交**，所以"随机扰动范数 0.078 却几乎不改变函数"近乎同义反复。
    首跑数据直接证实了这点 —— 同样 0.078 的位移，随机族 ACC 只掉 0.002–0.008，
    而 real 掉 0.047。两者根本没在同一个"函数改变量"上比较。

    正确的对齐量是**函数效果**，这里取最直接的一个：干净准确率的代价。
    做法是按各族的 (ACC 代价, ASR) 曲线插值到同一个 ``acc_cost``，再比 ASR。
    只有在这个坐标下随机方向仍然打不掉后门，"方向特殊"才立得住。

    ``acc_cost`` 默认取 ``ACC_GUARD`` —— 预先写死的那个数，不是看完数据挑的。
    落在实测区间之外时**返回 nan 并标 ``extrapolated``**，不外推。
    """
    baseline = [row for row in rows if row["family"] == "zero"]
    if not baseline:
        return {"error": "缺 zero 基线格，等代价比较无从谈起"}
    acc0, asr0 = float(baseline[0]["acc"]), float(baseline[0]["asr"])

    per_family: Dict[str, Dict[str, Any]] = {}
    for family in sorted({str(row["family"]) for row in rows
                          if row["family"] != "zero"}):
        block = sorted((row for row in rows if row["family"] == family),
                       key=lambda row: float(row["multiplier"]))
        # 同一 multiplier 的多个 seed 先取均值，再按 ACC 代价排序
        by_multiplier: Dict[float, List[Dict[str, Any]]] = {}
        for row in block:
            by_multiplier.setdefault(float(row["multiplier"]), []).append(row)
        points = sorted(
            ((acc0 - float(np.mean([float(r["acc"]) for r in group])),
              float(np.mean([float(r["asr"]) for r in group])), multiplier)
             for multiplier, group in by_multiplier.items()),
            key=lambda point: point[0])
        costs = [0.0] + [point[0] for point in points]
        asrs = [asr0] + [point[1] for point in points]

        entry: Dict[str, Any] = {"n_points": len(points)}
        if acc_cost > max(costs):
            entry.update({"asr_at_cost": float("nan"), "extrapolated": True,
                          "max_acc_cost_measured": float(max(costs))})
        else:
            entry.update({"asr_at_cost": float(np.interp(acc_cost, costs,
                                                         asrs)),
                          "extrapolated": False})
            spans = [point[2] for point in points if point[0] >= acc_cost]
            entry["multiplier_at_cost"] = (float(min(spans)) if spans
                                           else float("nan"))
        if np.isfinite(entry["asr_at_cost"]) and asr0 > 0:
            entry["asr_retained"] = entry["asr_at_cost"] / asr0
        per_family[family] = entry

    return {"acc_cost": float(acc_cost), "baseline_acc": acc0,
            "baseline_asr": asr0, "per_family": per_family}


def flatness_verdict(rows: Sequence[Dict[str, Any]],
                     acc_guard: float = ACC_GUARD) -> Dict[str, Any]:
    """把 (族 × 幅度) 的 ASR/ACC 网格翻译成机制判词。

    # 这里的 `real` 是什么（首跑后必须写清楚的语义）

    锚点是**漂移之后**的模型（默认 r=400 的个性化模型）。所以：

    - ``zero`` 格 = 真实的 200 轮良性漂移**已经走完**的那个点（ASR 的地板本身）；
    - ``real`` 格 = 沿同一方向**再多走**一段（外推），**不是**复现那 200 轮。

    因此判词问的是"**继续沿良性方向走**还是**随便走**，哪个更能打掉残留后门"，
    而不是"那 200 轮为什么没打掉"。初版把这两件事混为一谈，判词因此在
    `real` 明明把 ASR 打到 0.055 的情况下仍然报"宽盆" —— 那是错的，已修。

    判据（两条都要，缺一不可）：

    1. **ACC 闸门**：某一族在某个幅度上 ACC 相对基线掉超过 ``acc_guard``，
       该格的 ASR 不参与结论 —— 模型被打坏时 ASR 掉是平凡的。
    2. **等 ACC 代价**（`iso_acc_comparison`）：把各族插值到同一个 ACC 代价再比。
       只按位移幅度比会被"高维随机方向天然是函数惰性的"这一条混杂掉。

    没有 ``zero`` 格就直接返回"未能确定" —— 没有基线的相对判断没有意义。
    """
    result: Dict[str, Any] = {"n_cells": len(rows), "acc_guard": acc_guard}
    baseline = [r for r in rows if r["family"] == "zero"]
    if not baseline:
        result["verdict"] = ("未能确定：缺 `zero` 自检格，没有未扰动的基线，"
                             "任何相对判断都无从谈起。")
        return result

    acc0 = float(baseline[0]["acc"])
    asr0 = float(baseline[0]["asr"])
    result.update({"baseline_acc": acc0, "baseline_asr": asr0})

    # -- 逐幅度：ACC 闸门**逐族**判（一族被打坏不该连累另一族的可读性）------
    per_multiplier: Dict[str, Any] = {}
    for multiplier in sorted({float(r["multiplier"]) for r in rows
                              if r["family"] != "zero"}):
        block = [r for r in rows if float(r["multiplier"]) == multiplier
                 and r["family"] != "zero"]
        random_block = [r for r in block if r["family"] in RANDOM_FAMILIES]
        real_block = [r for r in block if r["family"] == "real"]
        entry: Dict[str, Any] = {
            "n": len(block),
            "worst_acc_drop": float(max(acc0 - float(r["acc"])
                                        for r in block)),
            "random_worst_acc_drop": (float(max(acc0 - float(r["acc"])
                                                for r in random_block))
                                      if random_block else float("nan")),
            "asr_random_mean": (float(np.mean([float(r["asr"])
                                               for r in random_block]))
                                if random_block else float("nan")),
            "asr_random_min": (float(min(float(r["asr"])
                                         for r in random_block))
                               if random_block else float("nan")),
        }
        entry["random_acc_ok"] = bool(np.isfinite(entry["random_worst_acc_drop"])
                                      and entry["random_worst_acc_drop"]
                                      <= acc_guard)
        if real_block:
            entry["asr_real"] = float(np.mean([float(r["asr"])
                                               for r in real_block]))
            entry["real_acc_drop"] = float(acc0 - np.mean(
                [float(r["acc"]) for r in real_block]))
            entry["real_acc_ok"] = bool(entry["real_acc_drop"] <= acc_guard)
        entry["acc_ok"] = bool(entry["worst_acc_drop"] <= acc_guard)
        per_multiplier[f"{multiplier:g}"] = entry
    result["per_multiplier"] = per_multiplier

    # -- 等 ACC 代价的对齐比较（主判据）--------------------------------------
    iso = iso_acc_comparison(rows, acc_cost=acc_guard)
    result["iso_acc"] = iso
    families = iso.get("per_family", {})
    real_entry = families.get("real", {})
    random_values = [entry["asr_at_cost"] for name, entry in families.items()
                     if name in RANDOM_FAMILIES
                     and np.isfinite(entry.get("asr_at_cost", np.nan))]

    if not real_entry or not np.isfinite(real_entry.get("asr_at_cost",
                                                        np.nan)):
        result["verdict"] = (
            f"未能确定：`real` 族在 ACC 代价 {acc_guard} 处没有可读的点"
            f"（实测的 ACC 代价区间没覆盖到它，本模块不外推）。"
            f"补一个更小的 multiplier 再跑。")
        return result
    if not random_values:
        result["verdict"] = (
            f"未能确定：没有任何随机方向族在 ACC 代价 {acc_guard} 处可读，"
            f"缺对照。随机族的 ACC 代价普遍偏小时要往上加 multiplier。")
        return result

    asr_real = float(real_entry["asr_at_cost"])
    asr_random = float(np.mean(random_values))
    real_retained = asr_real / asr0 if asr0 > 0 else float("nan")
    random_retained = asr_random / asr0 if asr0 > 0 else float("nan")
    result.update({
        "iso_acc_cost": float(acc_guard),
        "asr_real_at_iso_acc": asr_real,
        "asr_random_at_iso_acc": asr_random,
        "real_retained_fraction": real_retained,
        "random_retained_fraction": random_retained,
    })

    anchor_note = ("（锚点是**漂移之后**的模型：`zero` 格就是那 200 轮良性漂移的"
                   "终点，`real` 格是沿同一方向**再外推**一段，不是复现那 200 轮。"
                   "口径：ASR 为 benign 个性化模型上的 target-排除口径。）")

    if not (np.isfinite(real_retained) and np.isfinite(random_retained)):
        result["verdict"] = "未能确定：基线 ASR 为 0，保留比例无定义。"
    elif random_retained >= 0.8 and real_retained <= 0.5:
        result["verdict"] = (
            f"**方向特殊，不是幅度**：在**同样的 ACC 代价** {acc_guard:g} 上，"
            f"沿良性方向继续走把 ASR 打到 {asr_real:.4g}（剩 {real_retained:.0%}），"
            f"而幅度等价的随机方向只到 {asr_random:.4g}（剩 {random_retained:.0%}）。"
            f"→ 后门的残留**不由位移幅度决定**；良性优化方向本身携带了任务信息，"
            f"对后门有不成比例的破坏力。这**证伪了「任意同幅度扰动都出不了盆」"
            f"的宽盆读法**，"
            f"但也**不等于**证明了后门可被普通训练洗掉（`zero` 格自己就是 200 轮"
            f"良性训练之后的结果，ASR 仍有 {asr0:.4g}）。" + anchor_note)
    elif random_retained < 0.8 and real_retained <= 0.5:
        result["verdict"] = (
            f"两者都有效：等 ACC 代价下 real 到 {asr_real:.4g}"
            f"（剩 {real_retained:.0%}）、随机到 {asr_random:.4g}"
            f"（剩 {random_retained:.0%}）。方向仍然更强，但随机方向也压得动，"
            f"说明幅度本身有贡献。**不要**只报 real。" + anchor_note)
    elif random_retained >= 0.8 and real_retained > 0.5:
        result["verdict"] = (
            f"**宽盆**：等 ACC 代价 {acc_guard:g} 上，real 与随机方向都打不掉后门"
            f"（剩 {real_retained:.0%} / {random_retained:.0%}）。"
            f"地板不挂在特定参数值上 —— (c) 函数平坦的正面证据，"
            f"且意味着参数空间的定点操作（剪枝/掩码/占位）原理上够不到它。"
            + anchor_note)
    else:
        result["verdict"] = (
            f"中间情形：等 ACC 代价下 real 剩 {real_retained:.0%}、"
            f"随机剩 {random_retained:.0%}，两个阈值都没跨过。"
            f"如实报数，**不要**据此选边。" + anchor_note)
    return result


# ---------------------------------------------------------------------------
# CPU：标定
# ---------------------------------------------------------------------------
def recipe_key(recipe: Dict[str, Any]) -> str:
    """配方的稳定标识：``族|幅度|seed``。用于断点续跑时比对已完成的格子。"""
    return (f"{recipe['family']}|{float(recipe['multiplier']):g}"
            f"|{int(recipe['seed'])}")


def load_drift(ckpt_dir, drift_from: int, drift_to: int
               ) -> Tuple[np.ndarray, np.ndarray, ps.ParamIndex,
                          Dict[str, Any]]:
    """读两端全局快照，返回 ``aggregated`` 子空间上的 (θ, Δθ_real, 索引, 读数)。

    `calibrate` 与 `evaluate_recipes` 共用这一段 —— 两边必须用**同一个**子空间
    与同一条漂移，否则清单里标定过的幅度和真正加到模型上的幅度会悄悄不一致。
    """
    ckpt_dir = Path(ckpt_dir)
    paths = {}
    for round_index in (drift_from, drift_to):
        path = global_path(ckpt_dir, round_index)
        if path is None:
            raise FileNotFoundError(
                f"{ckpt_dir} 下缺 round_{round_index:04d} 的全局快照 —— "
                f"T3 的幅度基准来自 T0 的漂移，两端都要有")
        paths[round_index] = path

    theta_all, delta_all, index_all = ps.displacement(
        load_state(paths[drift_from]), load_state(paths[drift_to]))
    mask = index_all.kind_mask(AGGREGATED_KINDS)
    index = ps.subset_index(index_all, mask)
    theta, delta_real = theta_all[mask], delta_all[mask]

    drift = {
        "drift_window": f"{drift_from}->{drift_to}",
        "n_params_aggregated": int(index.n_params),
        "drift_relative_displacement": ps.relative_displacement(delta_real,
                                                                theta),
        "drift_l2": ps.l2(delta_real),
        "theta_l2": ps.l2(theta),
    }
    return theta, delta_real, index, drift


def calibrate(ckpt_dir, drift_from: int, drift_to: int,
              recipes: Sequence[Dict[str, Any]]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按配方造一遍扰动并测量它们，返回 ``(清单行, 漂移本身的读数)``。

    这一步的全部意义是**在花机时之前**确认标定对不对：achieved 与 target 应当
    一致到浮点精度，`gaussian_layer_matched` 的逐层剖面偏差应当接近 0，
    `real` 与真实漂移的 cos 应当恰好是 1。
    """
    theta, delta_real, index, drift = load_drift(ckpt_dir, drift_from, drift_to)
    rows: List[Dict[str, Any]] = []
    for recipe in recipes:
        direction = make_direction(recipe["family"], delta_real, theta, index,
                                   recipe["seed"])
        vector = scale_to_relative(direction, theta, recipe["target_relative"])
        rows.append({**recipe,
                     **perturbation_profile(vector, theta, delta_real, index)})
    return rows, drift


# ---------------------------------------------------------------------------
# GPU：评估
# ---------------------------------------------------------------------------
def _client_accuracy(model: Any, loader: Any, device: Any) -> Dict[str, float]:
    """干净准确率（MTA 口径：全部样本）。没有它，ASR 掉了也说明不了任何事。"""
    import torch

    was_training = model.training
    model.eval()
    correct = total = 0
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                preds = model(images).argmax(dim=1)
                correct += int((preds == labels).sum())
                total += int(labels.numel())
    finally:
        if was_training:
            model.train()
    return {"acc": (correct / total if total else float("nan")),
            "n_acc_samples": int(total)}


def client_checkpoint(ckpt_dir, client_id: int,
                      anchor_round: Optional[int] = None) -> Optional[Path]:
    """要往上加扰动的那个客户端模型的路径；不存在返回 ``None``。

    ``anchor_round=None`` 用 run **根目录**的 ``client_<cid>.pt``
    （`hooks.save_run` 写的最终模型，全部 40 个都有）。

    给了 ``anchor_round`` 就改用 ``round_XXXX/client_<cid>.pt``
    —— 那是 `SnapshotRecorder` 写的，**只覆盖被选中的快照客户端**
    （`config.yaml` 的 `snapshot_n_benign: 10` + `snapshot_n_malicious: 2`），
    而且带 staleness（存的是"该网格点之后该客户端首次参与后的模型"）。
    所以换 anchor 会把可评客户端数从 36 降到约 10，这是数据本身的限制，
    不是可以绕开的参数 —— `eval_clients` 会如实少返回，不补齐。
    """
    ckpt_dir = Path(ckpt_dir)
    if anchor_round is None:
        path = ckpt_dir / f"client_{int(client_id)}.pt"
    else:
        path = (ckpt_dir / f"round_{int(anchor_round):04d}"
                / f"client_{int(client_id)}.pt")
    return path if path.exists() else None


def client_staleness(ckpt_dir, anchor_round: Optional[int]
                     ) -> Dict[int, int]:
    """从 ``snapshot_manifest.json`` 取各客户端在该网格点的 staleness。

    staleness = 实际保存轮次 − 网格轮次 ≥ 0。它是**协变量，不是噪声**：
    anchor 名义上是 r=200，实际可能是该客户端 r=203 才参与后的模型。
    manifest 缺失时返回空字典 —— 缺就缺，不编。
    """
    if anchor_round is None:
        return {}
    path = Path(ckpt_dir) / "snapshot_manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["client_id"]): int(record["staleness"])
            for record in manifest.get("records", [])
            if record.get("kind") == "client"
            and int(record.get("grid_round", -1)) == int(anchor_round)}


def eval_clients(ckpt_dir, benign_only: bool = True,
                 max_clients: Optional[int] = None,
                 anchor_round: Optional[int] = None) -> List[Dict[str, Any]]:
    """要评的客户端记录。``max_clients`` 取**编号最小的前 N 个**（确定性）。

    默认只评良性客户端：恶意客户端的 ASR≈1.0，混进均值里会把地板抬起来，
    而地板讲的是**受害者**模型上还剩多少。

    ``anchor_round`` 非空时**只保留该轮次真有快照的客户端**，并把 staleness
    附在记录上（`anchor_staleness`）。少了就少了，如实返回。
    """
    meta = json.loads((Path(ckpt_dir) / "meta.json").read_text())
    clients = sorted((record for record in meta["clients"]
                      if record.get("test_indices")
                      and not (benign_only and record["is_malicious"])),
                     key=lambda record: int(record["client_id"]))
    if anchor_round is not None:
        staleness = client_staleness(ckpt_dir, anchor_round)
        kept = []
        for record in clients:
            cid = int(record["client_id"])
            if client_checkpoint(ckpt_dir, cid, anchor_round) is not None:
                kept.append({**record,
                             "anchor_staleness": staleness.get(cid, None)})
        clients = kept
    if not clients:
        raise ValueError(
            f"{ckpt_dir} 在 anchor_round={anchor_round} 上没有可评的客户端"
            f"（benign_only={benign_only}）。按轮次的客户端快照只覆盖 "
            f"snapshot_n_benign + snapshot_n_malicious 个被选中的客户端；"
            f"要全部 40 个只能用 anchor_round=None 的最终模型。")
    if max_clients is not None and max_clients > 0:
        clients = clients[:int(max_clients)]
    return clients


def missing_eval_inputs(ckpt_dir, drift_from: int = 200, drift_to: int = 400,
                        benign_only: bool = True,
                        max_clients: Optional[int] = None,
                        anchor_round: Optional[int] = None) -> List[str]:
    """``--mode eval`` 需要、但这个 run 目录里缺掉的东西。

    dry-run 会先跑这一遍：**一次把缺的全报齐**，而不是让 GPU 作业排队两小时后
    死在第一个缺失的文件上。检查四类：

    - `meta.json`（客户端划分、target_class、test_indices）
    - `generator.pt`（触发器生成器 —— ASR 口径要用原始触发器）
    - 每个要评的客户端的 `client_<cid>.pt`（在 run 根目录，不是 round_XXXX/ 下）
    - 漂移两端的全局快照 `round_XXXX/global.pt`（幅度基准从这里来）
    """
    ckpt_dir = Path(ckpt_dir)
    missing: List[str] = []
    for name in ("meta.json", "generator.pt"):
        if not (ckpt_dir / name).exists():
            missing.append(str(ckpt_dir / name))
    for round_index in (drift_from, drift_to):
        if global_path(ckpt_dir, round_index) is None:
            missing.append(str(ckpt_dir / f"round_{round_index:04d}"
                               / "global.pt"))
    if (ckpt_dir / "meta.json").exists():
        try:
            clients = eval_clients(ckpt_dir, benign_only, max_clients,
                                   anchor_round)
        except ValueError as error:
            missing.append(str(error))
            return missing
        for record in clients:
            if client_checkpoint(ckpt_dir, int(record["client_id"]),
                                 anchor_round) is None:
                missing.append(str(ckpt_dir / f"client_"
                                   f"{int(record['client_id'])}.pt"))
    return missing


def append_row(row: Dict[str, Any], path) -> Path:
    """把一行追加进 CSV（文件不存在时先写表头）。

    逐格写盘是为了让作业被抢占时**已经算过的不白算**。表头与行的键集合不一致时
    直接报错 —— 静默错位会把一张看似正常的表变成垃圾。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    if path.exists():
        with open(path, encoding="utf-8", newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing is not None and existing != fields:
            raise ValueError(
                f"{path} 已有的表头与本次要写的列不一致，拒绝追加：\n"
                f"  已有：{existing}\n  本次：{fields}\n"
                f"（换了 --families/--multipliers 就换个 --raw-out，别混写一个文件）")
        with open(path, "a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
    else:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    return path


def load_raw_rows(raw_path) -> List[Dict[str, Any]]:
    """读回逐格原始行（续跑后要把旧行与新行一起聚合）。"""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return []
    with open(raw_path, encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_done_keys(raw_path) -> set:
    """已完成的 ``(配方, 客户端)`` 格子 —— 断点续跑用。

    多小时的 GPU 作业被抢占是常态，重头再来一遍是纯浪费。逐格写盘 + 这个集合
    就够了：**不猜、不合并**，只跳过 raw CSV 里已经有的那些格子。
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return set()
    with open(raw_path, encoding="utf-8") as handle:
        return {(row["recipe_key"], int(row["client_id"]))
                for row in csv.DictReader(handle)
                if row.get("recipe_key") and row.get("client_id")}


def evaluate_recipes(ckpt_dir, recipes: Sequence[Dict[str, Any]],
                     drift_from: int, drift_to: int, test_dataset: Any, *,
                     model_size: int, device: Any, batch_size: int = 128,
                     benign_only: bool = True,
                     max_clients: Optional[int] = None,
                     anchor_round: Optional[int] = None,
                     raw_path=None, resume: bool = False
                     ) -> List[Dict[str, Any]]:
    """把扰动加到**每个客户端自己的最终个性化模型**上，评 ASR + ACC，返回逐格的原始行。

    - 扰动只加在 ``aggregated`` 参数上，客户端自己的 BN 原样保留（FedBN）。
    - ASR 用 `recompute_asr_final.client_asr` 的**原始触发器**口径，
      与 B2 那条 0.4133 的曲线可比；ACC 是同一 loader 上的干净准确率。

    **循环是"客户端在外、配方在内"**：每个 checkpoint 只从盘上读一次。反过来写
    （配方在外）会让 53 个配方 × 36 个客户端 = 1908 次读盘、约 86GB 的 I/O，
    在集群文件系统上这比 GPU 计算本身还贵。代价是扰动向量要按 (客户端, 配方)
    重新生成 —— 固定 seed 保证每次生成的完全一样，一次约 0.2 秒，总共几分钟，
    比 86GB 的读便宜得多。

    ``raw_path`` 给了就**逐格追加写盘**，配合 ``resume`` 可以断点续跑。
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from resnet import get_resnet
    from generator import Autoencoder

    from .hooks import load_client_model
    from .recompute_asr_final import client_asr

    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    target_class = int(meta["target_class"])
    num_classes = int(meta["num_classes"])

    generator = Autoencoder().to(device)
    generator.load_state_dict(torch.load(ckpt_dir / "generator.pt",
                                         map_location=device))
    generator.device = device

    # 扰动由「全局快照的漂移」定义，与客户端无关；索引/子空间与 calibrate 共用
    theta, delta_real, index, _ = load_drift(ckpt_dir, drift_from, drift_to)
    clients = eval_clients(ckpt_dir, benign_only, max_clients, anchor_round)
    done = load_done_keys(raw_path) if (resume and raw_path) else set()
    if done:
        print(f"[exp_t3] 续跑：raw CSV 里已有 {len(done)} 个格子，跳过它们")

    rows: List[Dict[str, Any]] = []
    total = len(clients) * len(recipes)
    for position, record in enumerate(clients, start=1):
        cid = int(record["client_id"])
        pending = [r for r in recipes if (recipe_key(r), cid) not in done]
        if not pending:
            print(f"[exp_t3] client {cid} 全部格子已完成，跳过")
            continue

        anchor_path = client_checkpoint(ckpt_dir, cid, anchor_round)
        if anchor_path is None:                 # eval_clients 已过滤，这里兜底
            raise FileNotFoundError(
                f"client {cid} 在 anchor_round={anchor_round} 上没有快照")
        model = load_client_model(
            anchor_path,
            lambda: get_resnet(size=int(model_size), num_classes=num_classes),
            device)
        base_state = {key: value.detach().cpu().numpy().copy()
                      for key, value in model.state_dict().items()}
        loader = DataLoader(Subset(test_dataset, list(record["test_indices"])),
                            batch_size=batch_size)

        for recipe in pending:
            direction = make_direction(recipe["family"], delta_real, theta,
                                       index, recipe["seed"])
            vector = scale_to_relative(direction, theta,
                                       recipe["target_relative"])
            perturbed = apply_perturbation(base_state, vector, index)
            model.load_state_dict({key: torch.as_tensor(value)
                                   for key, value in perturbed.items()})
            model.to(device)

            asr = client_asr(model, generator, loader, target_class, device)
            accuracy = _client_accuracy(model, loader, device)
            row = {"run_id": ckpt_dir.name, "recipe_key": recipe_key(recipe),
                   **recipe, "client_id": cid,
                   "anchor_round": ("" if anchor_round is None
                                    else int(anchor_round)),
                   "anchor_staleness": record.get("anchor_staleness", ""),
                   "is_malicious": bool(record["is_malicious"]), **asr,
                   **accuracy}
            rows.append(row)
            if raw_path is not None:
                append_row(row, raw_path)
        del model
        print(f"[exp_t3] client {cid} 完成 {len(pending)} 个格子 "
              f"（{position}/{len(clients)} 个客户端，共 {total} 格）")
    return rows


def aggregate_results(raw_rows: Sequence[Dict[str, Any]],
                      profiles: Dict[str, Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """把逐 (配方, 客户端) 的原始行聚合成逐配方一行，并带上标定读数。

    只对**有限值**求均值；某个配方一个有效客户端都没有时记 nan（不填 0，
    遵"无定义留空"铁律）。
    """
    by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        by_recipe.setdefault(str(row["recipe_key"]), []).append(row)

    def _mean(block: Sequence[Dict[str, Any]], key: str) -> float:
        values = [float(row[key]) for row in block
                  if row.get(key) not in (None, "")
                  and np.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    out: List[Dict[str, Any]] = []
    for key, block in by_recipe.items():
        first = block[0]
        out.append({
            "run_id": first.get("run_id", ""), "recipe_key": key,
            "family": first["family"],
            "multiplier": float(first["multiplier"]),
            "seed": int(first["seed"]),
            "target_relative": float(first["target_relative"]),
            **profiles.get(key, {}),
            "n_clients": len(block),
            "asr": _mean(block, "asr_std_filtered"),
            "asr_unfiltered": _mean(block, "asr_unfiltered"),
            "acc": _mean(block, "acc"),
        })
    return sorted(out, key=lambda row: (row["multiplier"], row["family"],
                                        row["seed"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="T3：地板是宽盆，还是真实漂移方向特殊？")
    parser.add_argument("--mode", choices=("build", "eval"), default="build",
                        help="build = CPU 标定与清单；eval = GPU 真评")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--drift-from", type=int, default=200)
    parser.add_argument("--drift-to", type=int, default=400)
    parser.add_argument("--base-relative", type=float, default=0.156,
                        help="幅度基准 = T0 实测的漂移相对位移（默认 0.156）")
    parser.add_argument("--multipliers", default="0.5,1,2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--out-dir", default="results/t3")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--model-size", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--include-malicious", action="store_true",
                        help="也评恶意客户端（默认只评良性 —— 地板讲的是受害者）")
    parser.add_argument("--anchor-round", type=int, default=None,
                        help="把扰动加到哪一轮的客户端模型上。缺省 = run 根目录的"
                             "最终模型（r=total_round，全部客户端）；给 200 就用 "
                             "round_0200/client_*.pt —— 但那只覆盖被选中的快照"
                             "客户端（约 10 个良性），且带 staleness")
    parser.add_argument("--max-clients", type=int, default=None,
                        help="只评编号最小的前 N 个客户端（先跑小规模试水用）")
    parser.add_argument("--raw-out", default="",
                        help="逐 (配方, 客户端) 的原始行 CSV，"
                             "缺省 <out-dir>/t3_raw.csv")
    parser.add_argument("--resume", action="store_true",
                        help="跳过 raw CSV 里已完成的格子（作业被抢占后续跑）")
    parser.add_argument("--execute", action="store_true",
                        help="eval 模式真跑；缺省只打印格子数与预估")
    args = parser.parse_args(argv)

    recipes = build_recipes(
        families=[f.strip() for f in args.families.split(",") if f.strip()],
        multipliers=[float(m) for m in args.multipliers.split(",") if m.strip()],
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        base_relative=args.base_relative)
    out_dir = Path(args.out_dir)

    if args.mode == "build":
        rows, drift = calibrate(args.ckpt_dir, args.drift_from, args.drift_to,
                                recipes)
        path = write_rows(rows, out_dir / "t3_manifest.csv")
        print(f"[exp_t3] {len(rows)} 个配方 -> {path}")
        print(f"\n=== 漂移基准（{drift['drift_window']}，aggregated 口径）===")
        print(f"  实测相对位移 {drift['drift_relative_displacement']:.6g}"
              f"（命令行给的基准 {args.base_relative:g}）")
        if abs(drift["drift_relative_displacement"]
               - args.base_relative) > 0.01:
            print(f"  ⚠️ 命令行基准与实测差得不小 —— 确认 --base-relative "
                  f"是不是该改成 {drift['drift_relative_displacement']:.4g}")
        print(f"\n=== 标定自检 ===")
        print(f"  {'family':<24}{'mult':>6}{'seed':>5}{'target':>9}"
              f"{'achieved':>10}{'cos(real)':>11}{'layer dev':>11}")
        for row in rows:
            print(f"  {row['family']:<24}{row['multiplier']:>6g}"
                  f"{row['seed']:>5}{row['target_relative']:>9.4f}"
                  f"{row['achieved_relative_displacement']:>10.4f}"
                  f"{row['cos_with_real_drift']:>11.4f}"
                  f"{row['layer_profile_max_rel_dev']:>11.4f}")
        print("\n  自检要点：achieved 必须等于 target；`real` 的 cos(real) 必须是 1；"
              "\n  `gaussian_layer_matched` 的 layer dev 必须 ≈ 0，"
              "而 `gaussian_global` 不会 —— 那正是两者的区别。")
        with open(out_dir / "t3_drift.json", "w", encoding="utf-8") as handle:
            json.dump(drift, handle, indent=2, ensure_ascii=False)
        return 0

    raw_path = Path(args.raw_out) if args.raw_out else out_dir / "t3_raw.csv"

    missing = missing_eval_inputs(args.ckpt_dir, args.drift_from, args.drift_to,
                                  not args.include_malicious, args.max_clients,
                                  args.anchor_round)
    if missing:
        print(f"[exp_t3] ✗ 这个 run 目录缺 {len(missing)} 个 eval 需要的文件：")
        for path in missing[:20]:
            print(f"    {path}")
        if len(missing) > 20:
            print(f"    …… 还有 {len(missing) - 20} 个")
        print("  client_*.pt / meta.json / generator.pt 由 hooks.save_run 在 run "
              "结束时写在**根目录**（不是 round_XXXX/ 下）；\n"
              "  round_XXXX/global.pt 由 snapshots.SnapshotRecorder 写。缺哪类就补哪类。")
        return 1

    if not args.execute:
        clients = eval_clients(args.ckpt_dir, not args.include_malicious,
                               args.max_clients, args.anchor_round)
        done = load_done_keys(raw_path) if args.resume else set()
        cells = len(recipes) * len(clients) - len(done)
        samples = sum(len(record["test_indices"]) for record in clients)
        print(f"[dry-run] {len(recipes)} 个配方 × {len(clients)} 个客户端"
              f"{'（只良性）' if not args.include_malicious else ''} = "
              f"{len(recipes) * len(clients)} 格"
              + (f"，其中 {len(done)} 格已在 {raw_path} 里，还剩 {cells} 格"
                 if done else ""))
        print(f"  每一格 = 该客户端 test 分区上的一趟 ASR（PGD num_iter=1 + "
              f"生成器 + 前向）加一趟干净准确率；"
              f"全体客户端每轮合计 {samples} 个样本。")
        print(f"  读盘：客户端在外层循环 -> 每个 checkpoint **只读一次**"
              f"（{len(clients)} 次），不是 {len(recipes) * len(clients)} 次。")
        print(f"  逐格写 {raw_path}；被抢占后加 --resume 续跑。")
        for recipe in recipes:
            print(f"  {recipe['family']:<24} m={recipe['multiplier']:<5g}"
                  f" seed={recipe['seed']} target_rel="
                  f"{recipe['target_relative']:.4f}")
        print(f"\n  加 --execute 真跑。")
        return 0

    import torch
    import torchvision

    device = torch.device("cpu" if args.device == "cpu"
                          else f"cuda:{args.device}")
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(args.data_root, train=False,
                                                download=False,
                                                transform=transform)
    evaluate_recipes(args.ckpt_dir, recipes, args.drift_from, args.drift_to,
                     test_dataset, model_size=args.model_size, device=device,
                     batch_size=args.batch_size,
                     benign_only=not args.include_malicious,
                     max_clients=args.max_clients,
                     anchor_round=args.anchor_round, raw_path=raw_path,
                     resume=args.resume)

    # 从 raw CSV 聚合（而不是只用本次跑出来的行）—— 续跑时旧格子也要算进去
    manifest, _ = calibrate(args.ckpt_dir, args.drift_from, args.drift_to,
                            recipes)
    profiles = {recipe_key(row): {key: row[key] for key in
                                  ("achieved_relative_displacement",
                                   "cos_with_real_drift", "cos_with_theta",
                                   "layer_profile_max_rel_dev")}
                for row in manifest}
    rows = aggregate_results(load_raw_rows(raw_path), profiles)
    path = write_rows(rows, out_dir / "t3_results.csv")
    print(f"\n[exp_t3] {len(rows)} 个配方（逐格原始行在 {raw_path}）-> {path}")
    print(f"  {'family':<24}{'mult':>6}{'seed':>5}{'ASR':>9}{'ACC':>9}{'n':>5}")
    for row in rows:
        print(f"  {row['family']:<24}{row['multiplier']:>6g}{row['seed']:>5}"
              f"{row['asr']:>9.4f}{row['acc']:>9.4f}{row['n_clients']:>5}")

    verdict = flatness_verdict(rows)
    print("\n=== 判词 ===")
    for key, value in verdict.items():
        if key in ("verdict", "per_multiplier"):
            continue
        print(f"  {key:<34} {value:.6g}" if isinstance(value, float)
              else f"  {key:<34} {value}")
    for multiplier, entry in verdict.get("per_multiplier", {}).items():
        print(f"  m={multiplier:<5} ACC 最多掉 {entry['worst_acc_drop']:.4f}"
              f"{'（可读）' if entry['acc_ok'] else '（ACC 已坏，ASR 不可读）'}"
              f"  随机方向 ASR 均值 {entry['asr_random_mean']:.4f}")
    print(f"\n  {verdict['verdict']}")
    with open(out_dir / "t3_verdict.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
