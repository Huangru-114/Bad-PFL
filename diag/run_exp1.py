"""实验 1 / 1B 的命令生成器（**默认 dry-run**）。

# 为什么是单轴扫描而不是全因子

全因子是 ``|N_m| × |ρ_p| × |seed| = 4 × 4 × 2 = 32`` 个 run。按 40 客户端 /
ResNet-10 / 200 轮估算，单个 run 在一张 GPU 上是小时量级 —— 32 个 run 会把
第一阶段拖成几天，而第一阶段要回答的只是"弱 / 过渡 / 强攻击各自在哪一段"。

所以默认是**十字扫描**：固定 ``ρ_p`` 扫 ``N_m``，固定 ``N_m`` 扫 ``ρ_p``,
交叉点共用同一个 run。4 + 4 − 1 = 7 个点 × 2 seed = 14 个 run。
确认了过渡区在哪里，再用 ``--grid`` 在那一小块上补全因子。

# 三件容易做错的事

1. **每个 run 必须有独立的 ``--run-tag``。** 否则 checkpoint 目录与植入 CSV
   会互相覆盖，而且覆盖是**静默**的。tag 由剂量参数拼出来，一定唯一。
2. **``ρ_p`` 是逐样本的伯努利概率，不是精确比例**（``fba.py:49`` 的
   ``torch.rand(...) <= poison_ratio``）。batch=32 时 ρ_p=0.1 有约 3.4% 的批次
   一个毒样本都没有。报告里写"投毒比例"时要说明这一点。
3. **``--eval-every`` 必须够密。** 实验 1 的核心图是 ASR vs MTA，
   每个 run 只有 ``total_round / eval_every`` 个 (MTA, ASR) 点。200 轮 / 每 5 轮
   = 40 个点；调到 20 就只剩 10 个点，散点图上看不出任何转折。

用法::

    python -m diag.run_exp1                 # 打印命令，不执行
    python -m diag.run_exp1 --stage 1b      # 只看 1B 的调度
    python -m diag.run_exp1 --execute       # 真跑
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from pathlib import Path

from .config import Cfg, load_config
from .schedule import AttackSchedule

__all__ = ["dose_points", "build_commands", "estimate_cost",
           "implantation_csv", "csv_has_paper_column", "main"]


def dose_points(bad_nums: Sequence[int], poison_rates: Sequence[float],
                bad_fixed: int, rate_fixed: float,
                full_grid: bool = False) -> List[Dict[str, Any]]:
    """要跑的 (N_m, ρ_p) 组合。

    默认十字扫描；``full_grid=True`` 才是全因子。交叉点只出现一次 ——
    重复跑同一个配置除了浪费机时，还会在并表时变成两个"独立"观测。
    """
    if full_grid:
        return [{"bad_num": int(b), "poison_rate": float(p)}
                for b in bad_nums for p in poison_rates]

    points: List[Dict[str, Any]] = []
    seen = set()
    for bad in bad_nums:
        key = (int(bad), float(rate_fixed))
        if key not in seen:
            seen.add(key)
            points.append({"bad_num": key[0], "poison_rate": key[1]})
    for rate in poison_rates:
        key = (int(bad_fixed), float(rate))
        if key not in seen:
            seen.add(key)
            points.append({"bad_num": key[0], "poison_rate": key[1]})
    return points


def _arm_prefix(prefix: str, pfl: str) -> str:
    """把 PFL 臂编进 run tag。

    **不编进去会静默覆盖数据**：`implantation_csv()` 的路径是
    ``exp_ij_implantation_fedavg_attack_a{alpha}_s{seed}_{tag}.csv``，
    只由 (alpha, seed, tag) 决定。同一个剂量格在 fedbn 与 fedrep 两条臂下
    tag 相同 → 第二条臂把第一条的 CSV 覆盖掉，而且 `--skip-existing`
    还会因为「文件已存在且有 asr_paper_all 列」直接跳过不跑。

    fedbn 保持原样（不加后缀），这样此前跑出来的文件名不变、仍可续跑。
    """
    return prefix if str(pfl) == "fedbn" else f"{prefix}_{pfl}"


def _tag(prefix: str, **parts: Any) -> str:
    pieces = [prefix]
    for name, value in parts.items():
        text = str(value).replace(".", "p").replace("-", "m")
        pieces.append(f"{name}{text}")
    return "_".join(pieces)


def _base_command(cfg_exp1: Cfg, seed: int, alpha: float,
                  instrument_root: str, results_dir: str,
                  ckpt_root: str, total_round: Optional[int] = None,
                  pfl: str = "fedbn", layer_metrics: bool = False,
                  local_steps: Optional[int] = None
                  ) -> List[str]:
    """一条 run_fl 命令。

    ``pfl`` 显式传下去而不是靠 config 默认值：``diag/config.yaml`` 是
    **gitignore** 的，靠它承载「这一批跑的是哪条 PFL 臂」等于没有记录。

    ``layer_metrics`` 默认**关**。此前这里无条件传 ``--layer-metrics``，
    而它逐层扫全部 key（``track._record_round`` → ``layer_signals`` +
    ``gram_matrix``），每轮一次。产物只落进 instrumentation 的逐轮 npz
    （``layer_update_norm`` / ``layer_cos_centroid`` / ``global_update_norm``），
    **Exp 1 的分析一列都不读** —— ``diag/analysis_exp1.py`` 里
    ``layer`` / ``update_norm`` / ``cos_centroid`` 零命中，它只读 implantation CSV，
    而这些量根本不进 CSV。检测类实验（I/J）才读 npz，那时显式开。
    """
    rounds = int(cfg_exp1.total_round if total_round is None else total_round)
    steps = int(cfg_exp1.local_steps if local_steps is None else local_steps)
    cmd = [
        "python", "-m", "diag.run_fl",
        "--mode", "attack",
        "--alpha", str(alpha),
        "--seed", str(int(seed)),
        "--defense", "fedavg",
        "--client-num", str(int(cfg_exp1.client_num)),
        "--select-per-round", str(int(cfg_exp1.select_per_round)),
        "--local-steps", str(steps),
        "--model-size", str(int(cfg_exp1.model_size)),
        "--total-round", str(rounds),
        "--eval-every", str(int(cfg_exp1.eval_every)),
        "--eval-include-malicious",
        "--pfl", str(pfl),
        "--instrument-dir", instrument_root,
        "--results-dir", results_dir,
        "--ckpt-root", ckpt_root,
    ]
    if layer_metrics:
        cmd.append("--layer-metrics")
    return cmd


def implantation_csv(results_dir: str, alpha: float, seed: int, tag: str) -> str:
    """run_fl 会写出的植入 CSV 路径。

    与 run_fl 的命名严格一致：
    ``exp_ij_implantation_<defense>_<mode>_a<alpha>_s<seed>_<run-tag>.csv``。
    exp1 里 defense 恒为 fedavg、mode 恒为 attack（见 _base_command）。
    """
    base = f"attack_a{alpha}_s{int(seed)}_{tag}"
    return str(Path(results_dir) / f"exp_ij_implantation_fedavg_{base}.csv")


def csv_has_paper_column(path: str) -> bool:
    """CSV 是否已带论文口径列 ``asr_paper_all``（= 新代码跑出来的，可复用）。

    只读表头，不依赖 pandas。文件不存在 / 读不到都算"没有"，从而会被重跑。
    旧口径的 CSV（无此列）一律判为需重跑，避免把 perturb 口径当新结果并进来。
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline()
    except OSError:
        return False
    return "asr_paper_all" in [c.strip() for c in header.split(",")]


def build_commands(cfg: Cfg, stage: str = "all", *,
                   instrument_root: str = "instrumentation",
                   results_dir: str = "results/raw",
                   ckpt_root: str = "./checkpoints",
                   full_grid: bool = False,
                   seeds: Optional[Sequence[int]] = None,
                   pfl: str = "fedbn",
                   layer_metrics: bool = False
                   ) -> List[Dict[str, Any]]:
    """返回 ``[{"tag", "stage", "cmd"}]``。``stage`` ∈ {1, 1b, all}。

    ``pfl`` ∈ {fedbn, fedrep}。它既进命令行（``--pfl``）**也进 run tag** ——
    两条臂的 CSV 路径只由 (alpha, seed, tag) 决定，不区分就会互相覆盖，
    而 ``--skip-existing`` 还会把第二条臂整个跳过。见 ``_arm_prefix``。
    """
    if pfl not in ("fedbn", "fedrep"):
        raise ValueError(f"未知的 pfl={pfl!r}；可选 fedbn / fedrep")
    exp1 = cfg.exp1
    alpha = float(exp1.alpha)
    seeds = [int(s) for s in (seeds if seeds is not None else exp1.seeds)]
    jobs: List[Dict[str, Any]] = []

    if stage in ("1", "all"):
        points = dose_points(exp1.bad_nums, exp1.poison_rates,
                             int(exp1.bad_num_fixed),
                             float(exp1.poison_rate_fixed), full_grid)
        for point in points:
            for seed in seeds:
                tag = _tag(_arm_prefix("e1", pfl), bad=point["bad_num"],
                           rho=point["poison_rate"], s=seed)
                cmd = _base_command(exp1, seed, alpha, instrument_root,
                                    results_dir, ckpt_root, pfl=pfl, layer_metrics=layer_metrics)
                cmd += ["--bad-client-num", str(point["bad_num"]),
                        "--poison-rate", str(point["poison_rate"]),
                        "--run-tag", tag]
                jobs.append({"tag": tag, "stage": "1", "cmd": cmd,
                             "csv": implantation_csv(results_dir, alpha, seed,
                                                     tag)})

    if stage in ("1b", "all"):
        # 1B 固定剂量，只变时间结构 —— 剂量和时间同时变就分不清是谁的作用
        bad = int(exp1.bad_num_fixed)
        rate = float(exp1.poison_rate_fixed)
        for spec in exp1.schedules:
            spec = dict(spec)
            kind = str(spec.pop("kind"))
            # 先构造一次，参数不合法就地报错，而不是等集群上跑到才发现
            schedule = AttackSchedule(kind=kind, **spec)
            for seed in seeds:
                tag = _tag(_arm_prefix("e1b", pfl), sched=kind, s=seed)
                cmd = _base_command(exp1, seed, alpha, instrument_root,
                                    results_dir, ckpt_root, pfl=pfl, layer_metrics=layer_metrics)
                cmd += ["--bad-client-num", str(bad),
                        "--poison-rate", str(rate),
                        "--attack-schedule", kind,
                        "--run-tag", tag]
                for name, value in spec.items():
                    cmd += [f"--attack-{name.replace('_', '-')}", str(value)]
                jobs.append({"tag": tag, "stage": "1b", "cmd": cmd,
                             "describe": schedule.describe(),
                             "csv": implantation_csv(results_dir, alpha, seed,
                                                     tag)})

    if stage == "calib":
        # ── Stage B 标定（导师意见 #5）──────────────────────────────────
        # 唯一自变量 = local_steps。剂量固定在十字扫描的交叉点
        # (bad_num_fixed, poison_rate_fixed)，因为标定要问的是「训练多久够」，
        # 不是「打多重」—— 两者同时变就分不清是谁让曲线提前平的。
        #
        # 与 tf-dpfl 侧 (local_epochs ∈ {1,3,5}) 同构：那边一个 epoch 是整个
        # 本地数据集一遍，这边 15 steps × batch 32 ≈ 0.38 epoch。**两库的
        # "local budget" 单位不同**，所以标定各做各的，只共享读数口径。
        #
        # 顺带更正报告 §2 的「1 local epoch」——Exp 1 从来就不是 1 个 epoch。
        # 用 `in` + 属性访问，不用 .get —— Cfg.__getattr__ 会把嵌套 dict 包成
        # Cfg（于是 calib.total_round 可用），而继承来的 dict.get 不会包，
        # 拿回来的是裸 dict、点号访问直接 AttributeError。persist 分支同此写法。
        if "calibration" not in exp1:
            raise ValueError(
                "stage=calib 需要 config.yaml 的 exp1.calibration 段"
                "（local_steps / total_round / eval_every）")
        calib = exp1.calibration
        bad = int(exp1.bad_num_fixed)
        rate = float(exp1.poison_rate_fixed)
        total = int(calib.total_round)
        every = int(calib.eval_every)
        for steps in calib.local_steps:
            for seed in seeds:
                # **local_steps 必须进 tag**：CSV 路径只由 (alpha, seed, tag) 决定，
                # 三格共用 tag 就是第二格覆盖第一格，而 --skip-existing 还会因为
                # 「文件已存在且有 asr_paper_all 列」把后两格整个跳过。
                tag = _tag(_arm_prefix("e1calib", pfl), steps=int(steps), s=seed)
                cmd = _base_command(exp1, seed, alpha, instrument_root,
                                    results_dir, ckpt_root, pfl=pfl,
                                    total_round=total, layer_metrics=layer_metrics,
                                    local_steps=int(steps))
                # eval_every 也要覆盖：标定要密的轨迹才看得出拐点。
                # _base_command 里已经放了一个 --eval-every，这里直接改那一项，
                # 而不是再 append 一个（argparse 取最后一个，但两个值并存的命令
                # 事后没法判读跑的是哪个）。
                cmd[cmd.index("--eval-every") + 1] = str(every)
                cmd += ["--bad-client-num", str(bad),
                        "--poison-rate", str(rate),
                        "--run-tag", tag]
                jobs.append({"tag": tag, "stage": "calib", "cmd": cmd,
                             "describe": (f"calibration: local_steps={steps} "
                                          f"@ Nm={bad}, rho={rate}, "
                                          f"{total} rounds, eval every {every}"),
                             "csv": implantation_csv(results_dir, alpha, seed, tag)})

    if stage in ("persist", "all") and "persistence" in exp1:
        # B2 专用长跑：攻击窗口 [start, end) 把 ASR 顶到高位，再干净训练到 total。
        # 用 burst(start, end-start) 表达；burst 之后自动是干净轮次。
        pers = exp1.persistence
        bad = int(exp1.bad_num_fixed)
        rate = float(exp1.poison_rate_fixed)
        total = int(pers.total_round)
        a_start = int(pers.attack_start)
        a_len = int(pers.attack_end) - a_start
        # 就地校验：窗口非法（如 end<=start 或超出总轮数）立刻报错
        AttackSchedule(kind="burst", start=a_start, length=a_len)
        if a_start + a_len > total:
            raise ValueError(
                f"persistence 攻击窗口 [{a_start},{a_start + a_len}) 超出 "
                f"total_round={total} —— 停攻后就没有干净轮次可观察衰减了")
        for seed in seeds:
            # A/B 线：δ 停攻即冻结（默认门控），并开 --freeze-trigger-eval 另存
            # 冻结触发器 ASR（asr_paper_frozen_*）。一个 run 同时给出 A 和 B。
            tag = _tag(_arm_prefix("e1b_persist", pfl), s=seed)
            cmd = _base_command(exp1, seed, alpha, instrument_root,
                                results_dir, ckpt_root, pfl=pfl,
                                total_round=total, layer_metrics=layer_metrics)
            cmd += ["--bad-client-num", str(bad),
                    "--poison-rate", str(rate),
                    "--attack-schedule", "burst",
                    "--attack-start", str(a_start),
                    "--attack-length", str(a_len),
                    "--freeze-trigger-eval",
                    "--run-tag", tag]
            jobs.append({"tag": tag, "stage": "persist", "cmd": cmd,
                         "describe": (f"A/B: attack [{a_start},{a_start + a_len})"
                                      f" of {total} rounds, delta frozen after"),
                         "csv": implantation_csv(results_dir, alpha, seed, tag)})

            # C 线（上界对照）：投毒仍只在 [a_start, a_start+a_len)，但生成器从
            # a_start 起一直在线更新 —— 与 A 只在衰减段不同。
            tag_c = _tag(_arm_prefix("e1b_persist_online", pfl), s=seed)
            cmd_c = _base_command(exp1, seed, alpha, instrument_root,
                                  results_dir, ckpt_root, pfl=pfl,
                                total_round=total, layer_metrics=layer_metrics)
            cmd_c += ["--bad-client-num", str(bad),
                      "--poison-rate", str(rate),
                      "--attack-schedule", "burst",
                      "--attack-start", str(a_start),
                      "--attack-length", str(a_len),
                      "--generator-online-from", str(a_start),
                      "--run-tag", tag_c]
            jobs.append({"tag": tag_c, "stage": "persist", "cmd": cmd_c,
                         "describe": (f"C: same but generator stays online from "
                                      f"round {a_start} (upper bound)"),
                         "csv": implantation_csv(results_dir, alpha, seed,
                                                 tag_c)})
    return jobs


def _cmd_int(cmd: Sequence[str], flag: str, default: int) -> int:
    """从生成好的命令里读一个整数参数。"""
    try:
        return int(cmd[list(cmd).index(flag) + 1])
    except (ValueError, IndexError):
        return int(default)


def estimate_cost(jobs: Sequence[Dict[str, Any]], cfg: Cfg) -> Dict[str, Any]:
    """把机时说清楚 —— 这是决定要不要缩规模的唯一依据。

    **逐 job 从命令里读**，不从 config 读。calib / persist 这些 stage 会覆盖
    ``--total-round`` / ``--local-steps`` / ``--eval-every``，而旧实现直接读
    ``cfg.exp1`` 的值 → ``--stage calib`` 会打印 200 轮 × 15 steps，
    实际跑的是 80 轮 × {15,45,75}。dry-run 打印的机时是决定要不要缩规模的
    唯一依据，它错了整件事就白算。

    每个 run 的轮数/步数不一致时（calib 正是如此），``rounds_per_run`` 等
    单 run 字段报 **min–max 区间字符串**而不是一个数 —— 报一个数就是在
    六个不同的 run 之间随便挑了一个。
    """
    exp1 = cfg.exp1
    sel = int(exp1.select_per_round)
    per_job = []
    for job in jobs:
        cmd = job["cmd"]
        rounds = _cmd_int(cmd, "--total-round", exp1.total_round)
        steps = _cmd_int(cmd, "--local-steps", exp1.local_steps)
        every = max(_cmd_int(cmd, "--eval-every", exp1.eval_every), 1)
        per_job.append({"rounds": rounds, "steps": steps, "every": every,
                        "batches": rounds * sel * steps,
                        "evals": rounds // every})

    def _span(key):
        vals = sorted({j[key] for j in per_job})
        if not vals:
            return 0
        return vals[0] if len(vals) == 1 else f"{vals[0]}-{vals[-1]}"

    return {
        "n_runs": len(jobs),
        "rounds_per_run": _span("rounds"),
        "local_steps_per_run": _span("steps"),
        "local_batches_per_run": _span("batches"),
        "evaluations_per_run": _span("evals"),
        "total_local_batches": sum(j["batches"] for j in per_job),
        # 散点图点数的**下界** —— 警告要按最差的那个 run 发，不是按平均
        "asr_mta_points_per_run": (min(j["evals"] for j in per_job)
                                   if per_job else 0),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="实验 1 / 1B 的命令生成器（默认 dry-run）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pfl", default="fedbn", choices=["fedbn", "fedrep"],
                        help="PFL 臂。fedbn=上游实现（BN 私有、分类头共享）；"
                             "fedrep=diag 旁路实现（分类头私有、BN 聚合，与 tf-dpfl "
                             "的 hier_fedrep 对齐）。**它会进 run tag**，两条臂的"
                             "产物互不覆盖；fedbn 保持原命名以便续跑。")
    parser.add_argument("--stage", default="all",
                        choices=["1", "1b", "persist", "calib", "all"],
                        help="calib = Stage B 收敛标定（local_steps 扫描）。"
                             "**不含在 all 里** —— 它用的是缩短的预算与加密的"
                             "评估点，和主力格子不可比，混进去会污染并表。")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--full-grid", action="store_true",
                        help="全因子而不是十字扫描。第一阶段不要用 —— "
                             "4×4×2=32 个 run")
    parser.add_argument("--layer-metrics", action="store_true",
                        help="逐层更新范数 / 逐层余弦 / global_update_norm。"
                             "默认**关**：它每轮扫一遍全部 key，产物只进 "
                             "instrumentation 的逐轮 npz，而 Exp 1 的分析"
                             "（diag/analysis_exp1.py）只读 implantation CSV、"
                             "一列都不碰它。检测类实验（I/J）读 npz，那时才开。")
    parser.add_argument("--instrument-root", default="instrumentation")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--execute", action="store_true",
                        help="真正执行；缺省只打印")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过植入 CSV 已带论文口径列 asr_paper_all 的 run —— "
                             "省机时。旧口径（无该列）的 CSV 仍会重跑。")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    jobs = build_commands(cfg, args.stage,
                          instrument_root=args.instrument_root,
                          results_dir=args.results_dir,
                          ckpt_root=args.ckpt_root,
                          full_grid=args.full_grid, seeds=args.seeds,
                          pfl=args.pfl, layer_metrics=args.layer_metrics)

    cost = estimate_cost(jobs, cfg)
    print(f"=== 实验 1 / 1B：{cost['n_runs']} 个 run ===")
    batches = cost["local_batches_per_run"]
    print(f"  每个 run {cost['rounds_per_run']} 轮 × "
          f"{cfg.exp1.select_per_round}×{cost['local_steps_per_run']} 本地 batch "
          f"= {batches:,} 个 batch" if isinstance(batches, int) else
          f"  每个 run {cost['rounds_per_run']} 轮 × "
          f"{cfg.exp1.select_per_round}×{cost['local_steps_per_run']} 本地 batch "
          f"= {batches} 个 batch（各 run 不同，报区间）")
    print(f"  合计约 {cost['total_local_batches']:,} 个本地 batch")
    print(f"  每个 run 有 {cost['evaluations_per_run']} 个 (MTA, ASR) 配对 —— "
          f"ASR-vs-MTA 散点图的点数")
    if cost["asr_mta_points_per_run"] < 20:
        print(f"  ⚠️ 最少的那个 run 只有 {cost['asr_mta_points_per_run']} 个配对，"
              f"散点图上看不出转折。把 eval_every 调小。")

    # --skip-existing：把已带论文口径列的 run 标记出来，不再重训
    if args.skip_existing:
        for job in jobs:
            job["skip"] = csv_has_paper_column(job.get("csv", ""))
        n_skip = sum(1 for j in jobs if j.get("skip"))
        print(f"  [--skip-existing] {n_skip}/{len(jobs)} 个 run 的 CSV 已带 "
              f"asr_paper_all，将跳过；实跑 {len(jobs) - n_skip} 个。")

    print()
    for job in jobs:
        note = f"   # {job['describe']}" if "describe" in job else ""
        flag = "  [SKIP: 已有 asr_paper_all]" if job.get("skip") else ""
        print(" ".join(job["cmd"]) + note + flag)

    if not args.execute:
        print(f"\n[dry-run] 以上 {len(jobs)} 条命令**未执行**。加 --execute 才真跑。")
        return 0

    todo = [j for j in jobs if not j.get("skip")]
    for index, job in enumerate(todo, 1):
        print(f"\n[{index}/{len(todo)}] {job['tag']}")
        result = subprocess.run(job["cmd"])
        if result.returncode != 0:
            # 不继续跑剩下的：后面的分析会把缺失的 run 当成"没测过"，
            # 而实际是"跑挂了"，两者在图上无法区分
            print(f"[run_exp1] {job['tag']} 失败（returncode="
                  f"{result.returncode}），中止")
            return result.returncode
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
