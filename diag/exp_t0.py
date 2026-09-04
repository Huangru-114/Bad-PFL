"""T0：全局模型的**逐坐标位移剖面**（Stage 0，纯 CPU，不排队）。

# 它回答的问题（以及它**不**回答的问题）

`PLAN_T0T4.md §2` 的谜题是"θ₄₀₀ 里那个让冻结触发器还能拿到 0.41 的东西是什么"，
三个候选机制是 (a) 休眠容量 / (b) 对齐共址 / (c) 函数平坦。

T0 在 CPU 上先回答**前置问题**：**攻击停止后的 200 轮纯干净训练，到底把参数
改写了多少？** 这个数本身就约束了 (a) 的可能性 —— 一个被大幅改写却仍然工作的
参数集不叫"休眠"。

**但 T0 单独判不了 (a) 的生死**：它测的是**全体坐标**的位移，不是后门载体 P 的
位移。(a) 完全可以以"载体是一个位移远低于全局的小子集"的形式存活。要把 W / P
分开需要 ``L_bd`` 的梯度（模型 + 数据 + 生成器），那是 S1，在集群上做。
所以本模块的判词一律以"**这给 (a) 施加了什么约束**"的形式写，
不写"(a) 死了"。

# 窗口怎么切（`PLAN_T0T4.md §1.1 结论 1`）

B2 臂 B 的曲线不是匀速衰减：``r0 0.626 → [0,30] 0.518 → [30,60] 0.429 →
[60,130] 0.419 → [130,200] 0.404``。直接拿 θ₂₀₀ vs θ₄₀₀ 会把"前 30 轮蒸发
0.23 的快速衰减段"和"之后的地板段"混成一个数。因此默认切三段：

    attack   [140, 200)   植入窗口 —— 位移的**参照尺度**
    decay    [200, 230]   快速衰减段（ASR 掉 0.23）
    floor    [230, 400]   地板段（ASR 几乎不动）

外加锚定窗口 ``200 → r``（r 取遍所有快照轮次），给出累计位移曲线。
默认轮次列表按 ``snapshot_every=50`` 的网格给（`PLAN_T0T4.md §6`），
``--rounds`` 可覆盖；网格上没有 230 时，decay 段会退化成 ``[200, 250]``，
本模块**如实按实际可用的轮次命名窗口**，不假装取到了 230。

# 两条必须随结果一起报的口径声明

1. **模型口径错位**（`PLAN_T0T4.md` 坑 3）：B2 的 ASR 是 **benign 个性化模型**
   口径，而这里的位移在 **global** 上测。两者不是同一个模型。
2. **BN 不聚合**（坑 6）：FedBN 一直开着，全局模型的 BN buffer 停在初始化值。
   主分析只用可训练参数，BN buffer 的位移**单独一行**报，不混进去。

# 用法

    python -m diag.exp_t0 \\
        --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \\
        --out-dir results/t0

    # 噪声底（Stage 1 的两条干净 run 就位后）：
    python -m diag.exp_t0 --ckpt-dir ... --noise-floor-dir checkpoints/clean_s1 ...

产出四张 CSV（``--out-dir`` 下）：``t0_windows.csv`` / ``t0_layers.csv`` /
``t0_energy.csv`` / ``t0_bins.csv``，外加一段判词打印到 stdout。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paramspace as ps

__all__ = ["load_state", "global_path", "available_global_rounds",
           "default_rounds", "build_windows", "window_report", "analyze",
           "displacement_verdict", "attack_reference", "random_walk_check",
           "load_window_rows", "write_rows", "main"]

#: 判词里用到的两个阈值。**先写死在这里再看数据**，事后不动
#: （`PLAN_T0T4.md` 坑 8）。两个都不是"显著性"，只是叙述的分界。
RATIO_COMPARABLE = 0.5      # 干净段位移 / 植入段位移 >= 此值 -> 称"相当"
RATIO_NEGLIGIBLE = 0.05     # <= 此值 -> 称"几乎没动"，此时必须有噪声底才可读

#: **实际参与联邦聚合**的参数种类。FedBN 下 BN 从不聚合，全局模型的
#: ``bn.weight`` / ``bn.bias`` 与 buffer 一样停在初始化值（首跑实测：10 个窗口
#: 的 bn_affine 位移**恰好为 0**，‖θ_bn‖ 恒为 sqrt(4800)=69.282）。把它们算进
#: 分母只会把 ‖θ‖ 从 43.3 抬到 81.7，凭空稀释相对位移约 1.9 倍，而分子一点不变。
#: 所以除了 ``trainable_*``（含 bn_affine）之外**另外**报一份 ``aggregated_*``。
#: 两份都出、都进 CSV —— 换分母不是为了让某个数好看，比值本身几乎不受影响
#: （首跑：1.35 vs 1.33）。
AGGREGATED_KINDS: Tuple[str, ...] = (ps.KIND_WEIGHT, ps.KIND_BIAS)


# ---------------------------------------------------------------------------
# 读取侧（唯一 import torch 的地方，且是延迟导入）
# ---------------------------------------------------------------------------
def load_state(path) -> Dict[str, np.ndarray]:
    """把一个 checkpoint 读成 ``Dict[str, np.ndarray]``。

    支持 ``.pt``（torch state_dict）与 ``.npz``（测试 fixture 用，不需要 torch）。
    torch 在**函数内部**导入 —— 本模块的其余部分在没有 torch 的机器上照样能跑。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    if path.suffix == ".npz":
        with np.load(path) as handle:
            return {key: np.asarray(handle[key]) for key in handle.files}

    from .hooks import load_checkpoint     # noqa: WPS433 —— 见 docstring

    state = load_checkpoint(path)
    out: Dict[str, np.ndarray] = {}
    for key, value in state.items():
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else \
            np.asarray(value)
        out[key] = array
    return out


def global_path(ckpt_dir, round_index: int) -> Optional[Path]:
    """``round_XXXX/`` 下的全局快照路径；不存在返回 ``None``。

    ``.pt`` 优先，其次 ``.npz``。后者不是集群上会出现的格式，它的用途是让
    **没有 torch 的机器也能跑通整条 CLI**（本机就是这种情况）—— 有了它，
    "CSV 列名对不对、判词分支走没走到"这类问题不必等上集群才发现。
    """
    directory = Path(ckpt_dir) / f"round_{int(round_index):04d}"
    for name in ("global.pt", "global.npz"):
        if (directory / name).exists():
            return directory / name
    return None


def available_global_rounds(ckpt_dir) -> List[int]:
    """磁盘上实际存在全局快照的轮次。

    以磁盘为准而不是读 manifest —— 任务被抢占 / 文件被清理时两者会不一致
    （与 ``snapshots.available_rounds`` 同样的理由）。
    """
    ckpt_dir = Path(ckpt_dir)
    rounds: List[int] = []
    for directory in sorted(ckpt_dir.glob("round_*")):
        try:
            round_index = int(directory.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if global_path(ckpt_dir, round_index) is not None:
            rounds.append(round_index)
    return sorted(rounds)


def default_rounds(ckpt_dir) -> List[int]:
    """没给 ``--rounds`` 时用磁盘上的全部全局快照。"""
    rounds = available_global_rounds(ckpt_dir)
    if not rounds:
        raise FileNotFoundError(
            f"{ckpt_dir} 下没有任何 round_*/global.pt。B2 的 run 应当有 "
            f"snapshot_every=50 的网格（PLAN_T0T4.md §6）；若确实没有，"
            f"T0 无法在 CPU 上做，需要带快照重跑。")
    return rounds


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------
def build_windows(rounds: Sequence[int], attack_start: int, attack_stop: int,
                  anchor: Optional[int] = None) -> List[Dict[str, Any]]:
    """构造要计算的 (from, to) 窗口列表，每个带一个 ``phase`` 标签。

    三类窗口：

    - ``segment``：相邻快照之间。切出"快速衰减段"与"地板段"的分辨率就来自它。
    - ``anchor``：``anchor -> r``，给累计位移曲线。默认 anchor = 攻击停止轮。
    - ``attack``：``attack_start -> attack_stop``，位移的参照尺度。

    ``phase`` 按窗口**整体**落在攻击停止轮的哪一侧定：``attack`` / ``clean`` /
    ``mixed``（跨越停止轮，判词里不使用）。

    同一对 ``(from, to)`` 只算一次，``kind`` 按 ``attack > anchor > segment``
    的优先级定 —— 否则 ``[140, 200)`` 会先被当成 segment 记下，判词就找不到
    参照尺度了；重复计算同一个窗口在千万量级的坐标上也不便宜。
    """
    rounds = sorted({int(r) for r in rounds})
    attack_start, attack_stop = int(attack_start), int(attack_stop)
    anchor = attack_stop if anchor is None else int(anchor)

    def phase_of(start: int, end: int) -> str:
        if end <= attack_stop and start >= attack_start:
            return "attack"
        if start >= attack_stop:
            return "clean"
        return "mixed"

    windows: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add(start: int, end: int, kind: str) -> None:
        if start >= end or (start, end) in windows:
            return
        windows[(start, end)] = {
            "round_from": start, "round_to": end, "kind": kind,
            "phase": phase_of(start, end), "span_rounds": end - start}

    if attack_start in rounds and attack_stop in rounds:
        add(attack_start, attack_stop, "attack")
    if anchor in rounds:
        for end in rounds:
            if end > anchor:
                add(anchor, end, "anchor")
    for start, end in zip(rounds[:-1], rounds[1:]):
        add(start, end, "segment")
    return [windows[key] for key in sorted(windows)]


# ---------------------------------------------------------------------------
# 单窗口的度量
# ---------------------------------------------------------------------------
def window_report(state_from: Dict[str, np.ndarray],
                  state_to: Dict[str, np.ndarray],
                  topk_fractions: Sequence[float] = (0.0001, 0.001, 0.01, 0.1),
                  n_bins: int = 10
                  ) -> Dict[str, Any]:
    """一个窗口的完整位移剖面。纯 numpy，可用手工 fixture 精确验证。

    返回四部分：``summary``（一行）、``layers``、``energy``、``bins``。
    ``summary`` 里 ``trainable_*`` 是主口径（排除 BN buffer），``bn_buffer_*``
    单独给 —— 坑 6 要求 BN 显式声明，不是"顺便算了"。
    """
    theta, delta, index = ps.displacement(state_from, state_to)
    trainable = index.kind_mask(ps.TRAINABLE_KINDS)
    bn_buffer = index.kind_mask((ps.KIND_BN_BUFFER,))
    bn_affine = index.kind_mask((ps.KIND_BN_AFFINE,))
    aggregated = index.kind_mask(AGGREGATED_KINDS)

    theta_t, delta_t = theta[trainable], delta[trainable]
    theta_a, delta_a = theta[aggregated], delta[aggregated]
    summary: Dict[str, Any] = {
        "n_params_total": int(theta.size),
        "n_params_trainable": int(trainable.sum()),
        "n_params_aggregated": int(aggregated.sum()),
        "n_params_bn_buffer": int(bn_buffer.sum()),
        "excluded_keys": ";".join(index.excluded),
        "trainable_l2_base": ps.l2(theta_t),
        "trainable_l2_delta": ps.l2(delta_t),
        "trainable_relative_displacement": ps.relative_displacement(delta_t,
                                                                    theta_t),
        "trainable_cos_theta_delta": ps.cosine(theta_t, delta_t),
        "trainable_cos_from_to": ps.cosine(theta_t, theta_t + delta_t),
        "trainable_mean_abs_delta": (float(np.abs(delta_t).mean())
                                     if delta_t.size else float("nan")),
        "trainable_median_abs_delta": (float(np.median(np.abs(delta_t)))
                                       if delta_t.size else float("nan")),
        "trainable_max_abs_delta": (float(np.abs(delta_t).max())
                                    if delta_t.size else float("nan")),
        # 实际参与聚合的那部分（不含 BN 仿射 —— FedBN 下它从不更新）
        "aggregated_l2_base": ps.l2(theta_a),
        "aggregated_l2_delta": ps.l2(delta_a),
        "aggregated_relative_displacement": ps.relative_displacement(delta_a,
                                                                     theta_a),
        "aggregated_cos_theta_delta": ps.cosine(theta_a, delta_a),
        "bn_affine_l2_base": ps.l2(theta[bn_affine]),
        "bn_affine_l2_delta": ps.l2(delta[bn_affine]),
        "bn_buffer_l2_base": ps.l2(theta[bn_buffer]),
        "bn_buffer_l2_delta": ps.l2(delta[bn_buffer]),
        "bn_buffer_relative_displacement": ps.relative_displacement(
            delta[bn_buffer], theta[bn_buffer]),
    }
    # |Δ| 与 |θ| 的秩相关：位移是"按比例缩放"还是"与当前值无关的改写"
    summary["spearman_absdelta_abstheta"] = ps.spearman(np.abs(delta_t),
                                                        np.abs(theta_t))
    agreement = ps.sign_agreement(theta_t, delta_t)
    summary["sign_agreement_theta_delta"] = agreement["rate"]
    summary["sign_agreement_n_compared"] = agreement["n_compared"]

    # 逐 kind 的一行行（weight / bias / bn_affine / bn_buffer 各自的位移）
    kind_rows = ps.layer_table(theta, delta, index, by="kind")

    return {
        "summary": summary,
        "layers": ps.layer_table(theta_t, delta_t,
                                 _masked_index(index, trainable), by="layer"),
        "kinds": kind_rows,
        "energy": ps.topk_energy(delta_t, topk_fractions),
        "bins": ps.binned_curve(np.abs(theta_t), np.abs(delta_t),
                                n_bins=n_bins, mode="quantile"),
    }


def _masked_index(index: ps.ParamIndex, mask: np.ndarray) -> ps.ParamIndex:
    """从整体索引里裁出"只含掩码为 True 的整张量"的子索引。

    掩码总是按**整张量**取的（``kind_mask`` / ``layer_mask`` 都是），所以
    这里只需按张量筛，不会出现半张量被切开的情况；真被切开时直接报错，
    因为那样得到的子索引与子向量对不上，静默通过会让逐层表整体错位。
    """
    keys, sizes, kinds, shapes = [], [], [], []
    for i, key in enumerate(index.keys):
        block = mask[index.offsets[i]:index.offsets[i + 1]]
        if block.all():
            keys.append(key)
            sizes.append(index.sizes[i])
            kinds.append(index.kinds[i])
            shapes.append(index.shapes[i])
        elif block.any():
            raise ValueError(f"掩码把张量 '{key}' 切开了 —— 子索引与子向量会错位")
    return ps.ParamIndex(keys, sizes, kinds, shapes, index.excluded)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------
def analyze(states: Dict[int, Dict[str, np.ndarray]],
            windows: Sequence[Dict[str, Any]],
            topk_fractions: Sequence[float] = (0.0001, 0.001, 0.01, 0.1),
            n_bins: int = 10, run_id: str = ""
            ) -> Dict[str, List[Dict[str, Any]]]:
    """对每个窗口跑 ``window_report``，摊平成四张表。

    ``states`` 的键是轮次。缺的轮次**跳过并如实记录**在返回的 ``skipped`` 里，
    不插值、不用邻近轮次顶替。
    """
    window_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for window in windows:
        start, end = int(window["round_from"]), int(window["round_to"])
        if start not in states or end not in states:
            skipped.append({**window, "reason": "缺少该轮次的 global.pt"})
            continue
        report = window_report(states[start], states[end],
                               topk_fractions=topk_fractions, n_bins=n_bins)
        tag = {"run_id": run_id, "round_from": start, "round_to": end,
               "kind": window["kind"], "phase": window["phase"],
               "span_rounds": window["span_rounds"]}
        window_rows.append({**tag, **report["summary"]})
        for row in report["layers"]:
            layer_rows.append({**tag, "scope": "layer", **row})
        for row in report["kinds"]:
            layer_rows.append({**tag, "scope": "kind", **row})
        for row in report["energy"]:
            energy_rows.append({**tag, **row})
        for row in report["bins"]:
            bin_rows.append({**tag, **row})

    return {"windows": window_rows, "layers": layer_rows,
            "energy": energy_rows, "bins": bin_rows, "skipped": skipped}


def _pick(rows: Sequence[Dict[str, Any]], kind: str, phase: str = "",
          longest: bool = True) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if r["kind"] == kind
                  and (not phase or r["phase"] == phase)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["span_rounds"]) if longest else \
        min(candidates, key=lambda r: r["span_rounds"])


def attack_reference(window_rows: Sequence[Dict[str, Any]]
                     ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """挑一个植入阶段的窗口做参照尺度，并说明它是否**完整覆盖**植入窗口。

    返回 ``(row, is_full)``。首选 ``kind == "attack"``（即恰好
    ``[attack_start, attack_stop)`` 那一格）；快照网格上没有 ``attack_start``
    时 —— 首跑就是这样，``snapshot_every=50`` 的网格是 [50,…,400]，140 不在上面
    —— 退而取**落在植入阶段内最长的那个窗口**（首跑是 150→200，覆盖 60 轮里的 50 轮）。

    退化取的是**更短**的窗口，因此参照尺度偏小、比值偏大。这个方向必须随判词
    一起说清楚，不能让"比值大"看起来像是攻击窗口位移小。
    """
    exact = _pick(window_rows, "attack")
    if exact is not None:
        return exact, True
    partial = [r for r in window_rows if r["phase"] == "attack"]
    if not partial:
        return None, False
    return max(partial, key=lambda r: r["span_rounds"]), False


def _displacement_of(row: Dict[str, Any]) -> Tuple[float, str]:
    """取一行的相对位移，优先用**实际参与聚合**的那份，并返回用的是哪一份。

    旧 CSV（首跑）没有 ``aggregated_*`` 列，回退到 ``trainable_*`` 并如实标注 ——
    两者差一个 FedBN 冻结 BN 仿射造成的常数稀释因子，比值几乎不受影响，
    但**绝对值差约 1.9 倍**，混着读会错。
    """
    value = row.get("aggregated_relative_displacement")
    if value is not None and str(value) != "" and np.isfinite(float(value)):
        return float(value), "aggregated"
    return float(row["trainable_relative_displacement"]), "trainable"


def displacement_verdict(window_rows: Sequence[Dict[str, Any]],
                         noise_floor: Optional[float] = None) -> Dict[str, Any]:
    """把位移剖面翻译成"这给 (a) 施加了什么约束"。

    刻意**不**输出 "(a) 死 / 活"：T0 测的是全体坐标，(a) 讲的是载体子集。
    能给的最强结论是一个**条件**：若 (a) 要成立，载体的位移必须显著低于全局
    平均 —— 这正是 S1 要去证伪的。

    ``noise_floor`` 是两条不同 seed 的干净 run 之间的相对位移（Stage 1 产出）。
    没有它时，"位移很小"无法与"所有参数都动得小"区分，判词会如实说 **未能确定**。
    """
    result: Dict[str, Any] = {"n_windows": len(window_rows)}
    attack, attack_is_full = attack_reference(window_rows)
    clean = _pick(window_rows, "anchor", phase="clean")
    if clean is None:
        clean = _pick(window_rows, "segment", phase="clean")

    if clean is None:
        result["verdict"] = ("未能确定：没有任何完整落在干净阶段的窗口。"
                             "T0 需要攻击停止轮之后的至少两个全局快照。")
        return result

    rel_clean, scope = _displacement_of(clean)
    result.update({
        "displacement_scope": scope,
        "clean_window": f"{clean['round_from']}->{clean['round_to']}",
        "clean_span_rounds": int(clean["span_rounds"]),
        "clean_relative_displacement": rel_clean,
        "clean_cos_from_to": float(clean["trainable_cos_from_to"]),
    })
    if noise_floor is not None and np.isfinite(noise_floor) and noise_floor > 0:
        result["noise_floor_relative_displacement"] = float(noise_floor)
        result["clean_over_noise_floor"] = rel_clean / float(noise_floor)

    if attack is None:
        result["verdict"] = (
            f"部分结论：干净阶段 {result['clean_window']} 的相对位移 "
            f"{rel_clean:.4g}（cos(θ_from, θ_to)="
            f"{result['clean_cos_from_to']:.4f}）。**缺植入阶段的窗口做参照尺度**"
            f"（快照网格上要有至少两个落在 [attack_start, attack_stop] 内的轮次），"
            f"因此无法说它是大是小 —— 未能确定。")
        return result

    rel_attack, _ = _displacement_of(attack)
    ratio = rel_clean / rel_attack if rel_attack > 0 else float("nan")
    result.update({
        "attack_window": f"{attack['round_from']}->{attack['round_to']}",
        "attack_span_rounds": int(attack["span_rounds"]),
        "attack_window_is_full": attack_is_full,
        "attack_relative_displacement": rel_attack,
        "ratio_clean_over_attack": ratio,
    })

    # 位移随窗口长度增长，所以"整个干净阶段 vs 一段植入窗口"这个比值一半是
    # 时长差造成的。再给一个**同时长**的比值：干净阶段里跨度与参照窗口相同的
    # 那一格。两个都报，不挑一个。
    matched = [r for r in window_rows if r["phase"] == "clean"
               and int(r["span_rounds"]) == int(attack["span_rounds"])]
    if matched:
        earliest = min(matched, key=lambda r: r["round_from"])
        rel_matched, _ = _displacement_of(earliest)
        result["clean_window_span_matched"] = (
            f"{earliest['round_from']}->{earliest['round_to']}")
        result["clean_relative_displacement_span_matched"] = rel_matched
        result["ratio_span_matched"] = (rel_matched / rel_attack
                                        if rel_attack > 0 else float("nan"))

    scope_note = ("已扣掉 FedBN 冻结的 BN 仿射参数"
                  if scope == "aggregated" else
                  "**含** BN 仿射参数，FedBN 下它恒为 0 位移，会稀释相对位移")
    partial_note = ("" if attack_is_full else
                    f"⚠️ 参照窗口只覆盖植入阶段的一部分"
                    f"（{attack['span_rounds']} 轮，快照网格上没有 attack_start），"
                    f"参照尺度因此偏小、比值偏大。")
    declaration = (f"（口径：位移在 **global** 模型上测，而 B2 的 ASR 是 "
                   f"**benign 个性化模型**口径 —— 两者不是同一个模型；"
                   f"分母 = `{scope}` 参数，{scope_note}；BN buffer 单独报。"
                   f"{partial_note}）")

    if not np.isfinite(ratio):
        result["verdict"] = ("未能确定：植入窗口的位移为 0，比值无定义。" +
                             declaration)
    elif ratio >= RATIO_COMPARABLE:
        matched_text = ""
        if "ratio_span_matched" in result:
            matched_text = (f"同时长比较（{result['clean_window_span_matched']} "
                            f"vs {result['attack_window']}，各 "
                            f"{attack['span_rounds']} 轮）也给 "
                            f"{result['ratio_span_matched']:.3g}。")
        result["verdict"] = (
            f"干净阶段的参数改写与植入阶段**相当**（比值 {ratio:.3g} ≥ "
            f"{RATIO_COMPARABLE}）：{clean['span_rounds']} 轮干净训练把参数移动了 "
            f"{rel_clean:.4g}（相对），而 ASR 只落到 0.41 的地板不再下去。"
            f"{matched_text}"
            f"这对 (a) 休眠容量施加了一个**强约束** —— (a) 若要成立，后门载体 P "
            f"必须是一个位移显著低于全局平均的小子集。**T0 判不了它的生死**"
            f"（T0 测全体坐标，不分 W/P），这正是 S1 载体分离要去证伪的。"
            + declaration)
    elif ratio <= RATIO_NEGLIGIBLE:
        if noise_floor is None:
            result["verdict"] = (
                f"未能确定：干净阶段位移很小（比值 {ratio:.3g} ≤ "
                f"{RATIO_NEGLIGIBLE}），但**没有噪声底**就无法把"
                f"「后门载体没被动」与「所有参数都动得小」区分开。"
                f"需要 Stage 1 的两条不同 seed 干净 run。" + declaration)
        else:
            result["verdict"] = (
                f"干净阶段位移很小（比值 {ratio:.3g}），且为噪声底的 "
                f"{result['clean_over_noise_floor']:.3g} 倍。"
                f"这与 (a) **相容**，但相容不是证实：位移小也可能是"
                f"「被用到但符号来回抵消」，T0 分不开这两种情形，T1 的 Σ|g| 才能。"
                + declaration)
    else:
        result["verdict"] = (
            f"中间情形：比值 {ratio:.3g} 落在 {RATIO_NEGLIGIBLE} 与 "
            f"{RATIO_COMPARABLE} 之间，两个方向都不成立。如实报比值本身，"
            f"**不要**据此选边。" + declaration)
    return result


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def random_walk_check(window_rows: Sequence[Dict[str, Any]]
                      ) -> Optional[Dict[str, Any]]:
    """干净阶段的漂移是**有方向的**还是**随机游走**？

    做法：拿最长的干净锚定窗口，找出正好把它铺满的连续 segment，然后比三个数：

    - ``observed``   = 锚定窗口实测的 ‖Δθ‖
    - ``quadrature`` = 各段 ‖Δθ‖ 的平方和开根 —— 各段两两正交时的期望值
    - ``linear``     = 各段 ‖Δθ‖ 直接相加 —— 各段完全同向时的值

    ``observed / quadrature ≈ 1`` 说明相邻段的位移彼此近似正交，良性训练**不是**
    在朝某个固定方向持续推进，而更像在参数空间里随机游走；越接近
    ``linear / quadrature`` 则越是定向漂移。

    这对判 (a)/(c) 有用：定向漂移意味着存在一个"良性任务要去的地方"，后门被
    挤出去只是时间问题；随机游走则意味着 200 轮之后还留着的东西，再跑 200 轮
    大概率还在 —— 与 B2 观察到的地板一致。

    铺不满（有缺口或重叠）时返回 ``None``，**不用可用的段硬凑**。
    """
    def norm_of(row: Dict[str, Any]) -> Optional[float]:
        for key in ("aggregated_l2_delta", "trainable_l2_delta"):
            value = row.get(key)
            if value not in (None, ""):
                return float(value)
        return None

    anchors = [r for r in window_rows
               if r["kind"] == "anchor" and r["phase"] == "clean"]
    if not anchors:
        return None
    whole = max(anchors, key=lambda r: r["span_rounds"])

    # 按**轮次**贪心地铺，不按 kind —— 锚定起点后的第一格在 build_windows 里被
    # 标成 anchor（优先级 anchor > segment），只收 kind=="segment" 会漏掉它，
    # 于是永远铺不满。每步取从 cursor 出发**最短**的那一格，得到最细的分解。
    starts: Dict[int, List[Dict[str, Any]]] = {}
    for row in window_rows:
        if row is whole or row["phase"] != "clean":
            continue
        if (row["round_from"] < whole["round_from"]
                or row["round_to"] > whole["round_to"]):
            continue
        starts.setdefault(int(row["round_from"]), []).append(row)

    segments: List[Dict[str, Any]] = []
    cursor = int(whole["round_from"])
    while cursor < int(whole["round_to"]):
        candidates = starts.get(cursor)
        if not candidates:
            return None                     # 有缺口 -> 不比，也不用别的段硬凑
        step = min(candidates, key=lambda r: r["round_to"])
        segments.append(step)
        cursor = int(step["round_to"])
    if len(segments) < 2:
        return None

    observed = norm_of(whole)
    parts = [norm_of(segment) for segment in segments]
    if observed is None or any(part is None for part in parts):
        return None
    quadrature = float(np.sqrt(sum(part ** 2 for part in parts)))
    linear = float(sum(parts))
    if quadrature <= 0:
        return None
    return {
        "window": f"{whole['round_from']}->{whole['round_to']}",
        "n_segments": len(segments),
        "observed_l2_delta": float(observed),
        "quadrature_l2_delta": quadrature,
        "linear_l2_delta": linear,
        "observed_over_quadrature": float(observed) / quadrature,
        "linear_over_quadrature": linear / quadrature,
    }


def load_window_rows(windows_csv, layers_csv=None) -> List[Dict[str, Any]]:
    """从已有的 ``t0_windows.csv`` 读回窗口行，供**只重算判词**用。

    用途：checkpoint 在集群、CSV 已经带回本地时，改了判词逻辑不必重跑一遍
    T0（那要重新读十几个 400MB 的 state_dict）。

    ``layers_csv`` 可选。首跑的 CSV 没有 ``aggregated_*`` 列，但
    ``t0_layers.csv`` 的 ``scope=kind`` 行里有逐 kind 的 ``l2_base`` /
    ``l2_delta``，足以把 ``aggregated_*`` **精确**重建出来（不是估计）：
    对 ``AGGREGATED_KINDS`` 的各 kind 做平方和即可。给了 layers 却缺某个窗口的
    kind 行时，那个窗口就保持没有 ``aggregated_*``，判词会自己回退到 trainable
    并标注 —— 不猜、不补。
    """
    with open(Path(windows_csv), encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for key in ("round_from", "round_to", "span_rounds"):
            row[key] = int(row[key])
        for key, value in list(row.items()):
            if key in ("run_id", "kind", "phase", "excluded_keys"):
                continue
            if isinstance(value, str) and value != "":
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        row["round_from"] = int(row["round_from"])
        row["round_to"] = int(row["round_to"])
        row["span_rounds"] = int(row["span_rounds"])

    if layers_csv is None:
        return rows

    with open(Path(layers_csv), encoding="utf-8") as handle:
        kind_rows = [row for row in csv.DictReader(handle)
                     if row.get("scope") == "kind"]
    by_window: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    for row in kind_rows:
        key = (int(row["round_from"]), int(row["round_to"]))
        by_window.setdefault(key, {})[row["group"]] = {
            "l2_base": float(row["l2_base"]),
            "l2_delta": float(row["l2_delta"]),
            "n_params": float(row["n_params"]),
        }
    for row in rows:
        groups = by_window.get((row["round_from"], row["round_to"]), {})
        if not all(kind in groups for kind in AGGREGATED_KINDS):
            continue
        base = np.sqrt(sum(groups[k]["l2_base"] ** 2 for k in AGGREGATED_KINDS))
        delta = np.sqrt(sum(groups[k]["l2_delta"] ** 2
                            for k in AGGREGATED_KINDS))
        row["n_params_aggregated"] = int(sum(groups[k]["n_params"]
                                             for k in AGGREGATED_KINDS))
        row["aggregated_l2_base"] = float(base)
        row["aggregated_l2_delta"] = float(delta)
        row["aggregated_relative_displacement"] = (float(delta / base)
                                                   if base > 0
                                                   else float("nan"))
    return rows


def write_rows(rows: Sequence[Dict[str, Any]], path) -> Path:
    """写 CSV。列取所有行键的并集，缺的留**空**（不是 0，不是 "N/A"）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _noise_floor_from(ckpt_dirs: Sequence[str], round_index: int
                      ) -> Optional[float]:
    """两条不同 seed 的干净 run 在同一轮次上的相对位移 = 噪声底。

    少于两条时返回 ``None``（判词会据此说"未能确定"），不用一条 run 硬凑。
    """
    dirs = [Path(d) for d in ckpt_dirs if d]
    if len(dirs) < 2:
        return None
    paths = [global_path(d, round_index) for d in dirs[:2]]
    if any(path is None for path in paths):
        raise FileNotFoundError(
            f"噪声底需要两条干净 run 在 r={round_index} 都有全局快照，"
            f"实际：{[str(p) if p else '缺' for p in paths]}")
    states = [load_state(path) for path in paths]
    theta, delta, index = ps.displacement(states[0], states[1])
    trainable = index.kind_mask(ps.TRAINABLE_KINDS)
    return ps.relative_displacement(delta[trainable], theta[trainable])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="T0：全局模型的逐坐标位移剖面（Stage 0，纯 CPU）")
    parser.add_argument("--ckpt-dir", default="",
                        help="B2 的 A/B run 目录，例如 "
                             "checkpoints/attack_a0.5_s0_e1b_persist_s0")
    parser.add_argument("--from-windows", default="",
                        help="只重算判词：从已有的 t0_windows.csv 读，"
                             "不碰 checkpoint（改判词逻辑时用，省去重读快照）")
    parser.add_argument("--from-layers", default="",
                        help="配合 --from-windows：给 t0_layers.csv 时，"
                             "从 scope=kind 行精确重建 aggregated_* 列")
    parser.add_argument("--rounds", default="",
                        help="逗号分隔的轮次；默认用磁盘上全部 global 快照")
    parser.add_argument("--attack-start", type=int, default=140)
    parser.add_argument("--attack-stop", type=int, default=200)
    parser.add_argument("--anchor", type=int, default=None,
                        help="锚定窗口的起点，默认 = attack-stop")
    parser.add_argument("--noise-floor-dir", action="append", default=[],
                        help="干净 run 目录，给两次（两个不同 seed）才算噪声底")
    parser.add_argument("--noise-floor-round", type=int, default=None,
                        help="噪声底在哪一轮比较，默认 = 最后一个快照轮次")
    parser.add_argument("--out-dir", default="results/t0")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    noise_floor = None

    if args.from_windows:
        window_rows = load_window_rows(args.from_windows,
                                       args.from_layers or None)
        if not window_rows:
            raise SystemExit(f"{args.from_windows} 里一行都没有")
        run_id = str(window_rows[0].get("run_id", ""))
        rounds = sorted({r["round_from"] for r in window_rows}
                        | {r["round_to"] for r in window_rows})
        windows = [{key: row[key] for key in
                    ("round_from", "round_to", "kind", "phase", "span_rounds")}
                   for row in window_rows]
        print(f"[exp_t0] 只重算判词：{len(window_rows)} 个窗口读自 "
              f"{args.from_windows}"
              + (f"（aggregated_* 由 {args.from_layers} 重建）"
                 if args.from_layers else ""))
    else:
        if not args.ckpt_dir:
            raise SystemExit("要么给 --ckpt-dir，要么给 --from-windows")
        ckpt_dir = Path(args.ckpt_dir)
        rounds = ([int(r) for r in args.rounds.split(",") if r.strip()]
                  if args.rounds else default_rounds(ckpt_dir))
        windows = build_windows(rounds, args.attack_start, args.attack_stop,
                                args.anchor)

        states: Dict[int, Dict[str, np.ndarray]] = {}
        needed = sorted({w["round_from"] for w in windows}
                        | {w["round_to"] for w in windows})
        for round_index in needed:
            path = global_path(ckpt_dir, round_index)
            if path is None:
                print(f"[exp_t0] ⚠️ 缺 {ckpt_dir}/round_{round_index:04d}/"
                      f"global.pt，相关窗口会被跳过（不插值）")
                continue
            states[round_index] = load_state(path)

        run_id = ckpt_dir.name
        tables = analyze(states, windows, n_bins=args.n_bins, run_id=run_id)
        window_rows = tables["windows"]

        for name in ("windows", "layers", "energy", "bins"):
            if tables[name]:
                path = write_rows(tables[name], out_dir / f"t0_{name}.csv")
                print(f"[exp_t0] {len(tables[name]):>6} 行 -> {path}")
        if tables["skipped"]:
            print(f"[exp_t0] ⚠️ 跳过 {len(tables['skipped'])} 个窗口："
                  f"{[(w['round_from'], w['round_to']) for w in tables['skipped']]}")

        noise_round = (args.noise_floor_round
                       if args.noise_floor_round is not None
                       else (rounds[-1] if rounds else None))
        if len(args.noise_floor_dir) >= 2 and noise_round is not None:
            noise_floor = _noise_floor_from(args.noise_floor_dir, noise_round)
            print(f"[exp_t0] 噪声底（两条干净 run 在 r={noise_round} 的相对位移）="
                  f"{noise_floor:.6g}")
        elif args.noise_floor_dir:
            print("[exp_t0] ⚠️ 噪声底需要**两条不同 seed** 的干净 run，"
                  f"只给了 {len(args.noise_floor_dir)} 条 —— 不计算。")

    has_aggregated = any(row.get("aggregated_relative_displacement") not in
                         (None, "") for row in window_rows)
    print("\n=== 位移剖面（BN buffer 已排除）===")
    print(f"  {'window':>14}{'kind':>9}{'phase':>7}"
          f"{'rel(trainable)':>16}{'rel(aggregated)':>17}"
          f"{'cos(θ,Δθ)':>12}{'cos(from,to)':>14}")
    for row in window_rows:
        label = f"{row['round_from']}->{row['round_to']}"
        aggregated = row.get("aggregated_relative_displacement")
        cell = (f"{float(aggregated):>17.6g}"
                if has_aggregated and aggregated not in (None, "")
                else f"{'':>17}")
        print(f"  {label:>14}{row['kind']:>9}{row['phase']:>7}"
              f"{float(row['trainable_relative_displacement']):>16.6g}{cell}"
              f"{float(row['trainable_cos_theta_delta']):>12.4f}"
              f"{float(row['trainable_cos_from_to']):>14.6f}")
    if not has_aggregated:
        print("  （没有 aggregated_* 列：判词会用 trainable，"
              "分母含 FedBN 下恒不动的 BN 仿射参数，相对位移被稀释）")

    verdict = displacement_verdict(window_rows, noise_floor)
    print("\n=== 判词 ===")
    for key, value in verdict.items():
        if key == "verdict":
            continue
        print(f"  {key:<38} {value:.6g}" if isinstance(value, float)
              else f"  {key:<38} {value}")
    print(f"\n  {verdict['verdict']}")

    walk = random_walk_check(window_rows)
    if walk is not None:
        print(f"\n=== 干净阶段的漂移：定向还是随机游走 ===")
        print(f"  窗口 {walk['window']}，由 {walk['n_segments']} 段铺满")
        print(f"  实测 ‖Δθ‖ {walk['observed_l2_delta']:.6g}  vs  "
              f"各段正交时 {walk['quadrature_l2_delta']:.6g}  vs  "
              f"完全同向时 {walk['linear_l2_delta']:.6g}")
        print(f"  实测/正交 = {walk['observed_over_quadrature']:.4g}"
              f"（同向/正交 = {walk['linear_over_quadrature']:.4g}）")

    summary_path = out_dir / "t0_verdict.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "rounds": rounds, "windows": windows,
                   # 判词是从 checkpoint 现算的，还是从已有 CSV 重算的 ——
                   # 两者数值一致，但来源要能查
                   "source": (f"recomputed from {args.from_windows}"
                              + (f" + {args.from_layers}"
                                 if args.from_layers else "")
                              if args.from_windows else "checkpoints"),
                   "verdict": verdict, "random_walk": walk,
                   "noise_floor_relative_displacement": noise_floor},
                  handle, indent=2, ensure_ascii=False)
    print(f"\n[exp_t0] 判词 -> {summary_path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
