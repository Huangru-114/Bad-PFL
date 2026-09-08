"""``diag.run_exp1`` 的单元测试（命令生成 + --skip-existing 的判定）。

重点盯两件容易错的事：
1. 重建的 CSV 路径必须与 run_fl 真实写出的名字逐字一致，否则 --skip-existing
   会永远判"不存在"而白跑，或误判"存在"而漏跑。
2. 只有带论文口径列 asr_paper_all 的 CSV 才算"可复用"；旧口径 CSV 必须重跑。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pathlib import Path as _P

ROOT = _P(__file__).resolve().parent.parent.parent

from diag.run_exp1 import csv_has_paper_column, implantation_csv


def test_implantation_csv_matches_run_fl_naming():
    # run_fl: exp_ij_implantation_<defense>_<mode>_a<alpha>_s<seed>_<tag>.csv
    path = implantation_csv("results/raw", 0.5, 0, "e1_bad4_rho0p1_s0")
    assert path == ("results/raw/exp_ij_implantation_fedavg_"
                    "attack_a0.5_s0_e1_bad4_rho0p1_s0.csv")


def test_csv_has_paper_column_true_only_when_column_present():
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "new.csv"
        new.write_text("round,seed,asr_paper_all,mta_personalized\n5,0,0.8,0.6\n")
        old = Path(tmp) / "old.csv"
        old.write_text("round,seed,asr_personalized_targeted\n5,0,0.2\n")
        assert csv_has_paper_column(str(new)) is True
        assert csv_has_paper_column(str(old)) is False


def test_csv_has_paper_column_false_when_missing_file():
    assert csv_has_paper_column("does/not/exist.csv") is False


def test_csv_has_paper_column_ignores_substring_false_positives():
    # 'asr_paper_all' 必须是完整列名，不能被 'asr_paper_benign' 之类误命中
    with tempfile.TemporaryDirectory() as tmp:
        partial = Path(tmp) / "partial.csv"
        partial.write_text("round,asr_paper_benign,asr_paper_malicious\n5,0.2,1.0\n")
        assert csv_has_paper_column(str(partial)) is False

# ══════════════════════════════════════════════════════════════════════════
# PFL 臂（fedbn / fedrep）—— 两条臂的产物必须互不覆盖
# ══════════════════════════════════════════════════════════════════════════

def test_pfl_arm_is_passed_to_run_fl():
    """`--pfl` 必须真的出现在命令里。

    不传就会落到 `diag/config.yaml` 的默认值上，而那个文件是 **gitignore** 的
    —— 靠它承载「这一批跑的是哪条 PFL 臂」等于没有记录。
    """
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    for arm in ("fedbn", "fedrep"):
        cmd = build_commands(cfg, "1", seeds=[0], pfl=arm)[0]["cmd"]
        assert "--pfl" in cmd, f"{arm}: 命令里没有 --pfl"
        assert cmd[cmd.index("--pfl") + 1] == arm


def test_the_two_arms_never_share_a_run_tag():
    """
    **不区分就会静默覆盖数据**：CSV 路径是
    ``exp_ij_implantation_fedavg_attack_a{alpha}_s{seed}_{tag}.csv``，
    只由 (alpha, seed, tag) 决定。两条臂 tag 相同 → 第二条覆盖第一条，
    而且 `--skip-existing` 会因为「文件已存在且有 asr_paper_all 列」
    把第二条臂整个跳过，输出看起来一切正常。
    """
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    tags_bn = {j["tag"] for j in build_commands(cfg, "all", seeds=[0], pfl="fedbn")}
    tags_rep = {j["tag"] for j in build_commands(cfg, "all", seeds=[0], pfl="fedrep")}
    assert tags_bn and tags_rep
    assert not (tags_bn & tags_rep), \
        f"两条臂共用了 {len(tags_bn & tags_rep)} 个 tag：{sorted(tags_bn & tags_rep)[:3]}"


def test_fedbn_tags_are_unchanged_so_old_runs_still_resume():
    """
    fedbn 保持原命名（不加后缀），否则此前跑出来的 CSV 全部认不出来、
    `--skip-existing` 会把已完成的格子重跑一遍。
    """
    from diag.run_exp1 import _arm_prefix, _tag
    assert _arm_prefix("e1", "fedbn") == "e1"
    assert _arm_prefix("e1", "fedrep") == "e1_fedrep"
    assert _tag(_arm_prefix("e1", "fedbn"), bad=4, rho=0.1, s=0) == "e1_bad4_rho0p1_s0"


def test_unknown_pfl_arm_is_rejected_early():
    """拼错臂名要在本地立刻炸，而不是提交到集群之后。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    try:
        build_commands(cfg, "1", seeds=[0], pfl="fedavg")
    except ValueError:
        return
    raise AssertionError("未知的 pfl 臂名没有被拒绝")


# ══════════════════════════════════════════════════════════════════════════
# --layer-metrics：默认关（标定跑不需要它，而它每轮扫全部 key）
# ══════════════════════════════════════════════════════════════════════════

def test_layer_metrics_is_off_by_default():
    """此前这里**无条件**传 ``--layer-metrics``，于是 Exp 1 的每个 run 都在付
    逐层扫描的钱，而 Exp 1 的分析一列都不读（见下一条测试给的证据）。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    for stage in ("1", "1b", "all"):
        for job in build_commands(cfg, stage, seeds=[0]):
            assert "--layer-metrics" not in job["cmd"], (
                f"{stage}/{job['tag']}：默认不该开 --layer-metrics")


def test_layer_metrics_can_still_be_turned_on():
    """检测类实验（I/J）读 instrumentation 的逐轮 npz，需要它 —— 是开关不是删除。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    for stage in ("1", "1b", "all"):
        for job in build_commands(cfg, stage, seeds=[0], layer_metrics=True):
            assert "--layer-metrics" in job["cmd"], \
                f"{stage}/{job['tag']}：显式打开时必须传下去"


def test_exp1_analysis_reads_no_layer_column():
    """关掉它不丢 Exp 1 需要的任何一列 —— 这是「默认关」的依据，不是感觉。

    ``analysis_exp1.py`` 只读 implantation CSV，而逐层量根本不进 CSV
    （它们进 ``instrumentation/<run>/round_XXXX.npz``，见 RoundRecorder.write）。
    这条测试直接扫源码，新增图表若开始读逐层列会立刻变红，提醒改默认值。
    """
    from pathlib import Path
    src = Path("diag/analysis_exp1.py").read_text(encoding="utf-8")
    for column in ("layer_update_norm", "layer_cos_centroid",
                   "layer_global_update_norm", "global_update_norm"):
        assert column not in src, (
            f"analysis_exp1.py 现在读 {column} 了 —— "
            "那 --layer-metrics 就不能默认关，请同步改 _base_command")


def test_layer_metrics_flag_does_not_change_run_tag():
    """开关不进 tag：同一格开与不开产物同名，避免把「同一个格子」拆成两份。

    （与 ``--pfl`` 相反 —— pfl 改变的是**跑什么**，必须分家；layer-metrics
    只改**记什么**，CSV 逐列相同。）
    """
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    off = {j["tag"]: j["csv"] for j in build_commands(cfg, "all", seeds=[0])}
    on = {j["tag"]: j["csv"]
          for j in build_commands(cfg, "all", seeds=[0], layer_metrics=True)}
    assert off == on


# ══════════════════════════════════════════════════════════════════════════
# Stage B 收敛标定（--stage calib）—— 导师意见 #5
# ══════════════════════════════════════════════════════════════════════════

def _calib_jobs(seeds=(0, 1)):
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    return build_commands(load_config("diag/config.yaml"), "calib", seeds=list(seeds))


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_calib_covers_the_declared_local_steps_x_seeds():
    from diag.config import load_config
    calib = load_config("diag/config.yaml").exp1.calibration
    jobs = _calib_jobs()
    want = {(int(s), seed) for s in calib.local_steps for seed in (0, 1)}
    got = {(int(_arg(j["cmd"], "--local-steps")), int(_arg(j["cmd"], "--seed")))
           for j in jobs}
    assert got == want, f"格子不齐：多 {got - want}，少 {want - got}"


def test_local_steps_is_the_only_axis_that_varies():
    """标定必须是单变量。剂量跟着一起变就分不清是谁让曲线提前平的。"""
    jobs = _calib_jobs()
    for flag in ("--bad-client-num", "--poison-rate", "--total-round",
                 "--eval-every", "--model-size", "--client-num",
                 "--select-per-round", "--alpha", "--pfl"):
        vals = {_arg(j["cmd"], flag) for j in jobs}
        assert len(vals) == 1, f"{flag} 在标定批里不恒定：{vals}"
    # 档数从 config 读，不写死 —— 网格一加档这条就该跟着走
    n_levels = len(_cfg().exp1.calibration.local_steps)
    assert len({_arg(j["cmd"], "--local-steps") for j in jobs}) == n_levels


def test_calib_uses_the_shortened_budget_and_denser_eval():
    """标定用缩短的预算 + 加密的评估点；不这么设就既贵又看不出拐点。"""
    from diag.config import load_config
    exp1 = load_config("diag/config.yaml").exp1
    jobs = _calib_jobs()
    for j in jobs:
        assert _arg(j["cmd"], "--total-round") == str(int(exp1.calibration.total_round))
        assert _arg(j["cmd"], "--eval-every") == str(int(exp1.calibration.eval_every))
    assert int(exp1.calibration.total_round) < int(exp1.total_round)
    assert int(exp1.calibration.eval_every) < int(exp1.eval_every)


def test_eval_every_appears_exactly_once():
    """`--eval-every` 被就地改写，不是再 append 一个。

    argparse 取最后一个值，所以两个并存**不会报错**，但那条命令事后没法判读
    到底跑的是哪个密度 —— 与陷阱 #7（`--config` 被静默忽略）同一类失败。
    """
    for j in _calib_jobs():
        assert j["cmd"].count("--eval-every") == 1, j["cmd"]
        assert j["cmd"].count("--local-steps") == 1, j["cmd"]
        assert j["cmd"].count("--total-round") == 1, j["cmd"]


def test_calib_tags_and_csvs_are_unique():
    """local_steps 必须进 tag。CSV 路径只由 (alpha, seed, tag) 决定 ——
    三格共用 tag = 后两格覆盖前一格，而 `--skip-existing` 还会因为
    「文件已存在且有 asr_paper_all 列」把它们整个跳过，输出看起来一切正常。"""
    jobs = _calib_jobs()
    tags = [j["tag"] for j in jobs]
    csvs = [j["csv"] for j in jobs]
    assert len(set(tags)) == len(jobs), f"tag 撞了：{sorted(tags)}"
    assert len(set(csvs)) == len(jobs), f"CSV 撞了：{sorted(csvs)}"
    for j in jobs:
        assert f"steps{_arg(j['cmd'], '--local-steps')}" in j["tag"]


def test_calib_is_not_included_in_stage_all():
    """标定的预算与主力不同，混进并表就是把两批不可比的数字画进一张图。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    assert not any(j["stage"] == "calib"
                   for j in build_commands(cfg, "all", seeds=[0]))


def test_calib_does_not_leak_into_the_main_grid():
    """反向锚点：主力格子的 local_steps / total_round 不受标定段影响。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    cfg = load_config("diag/config.yaml")
    for stage in ("1", "1b"):
        for j in build_commands(cfg, stage, seeds=[0]):
            assert _arg(j["cmd"], "--local-steps") == str(int(cfg.exp1.local_steps))
            assert _arg(j["cmd"], "--eval-every") == str(int(cfg.exp1.eval_every))


def test_calib_arms_do_not_share_tags_with_fedbn():
    """两条 PFL 臂的标定产物也不能互相覆盖（与主力格子同一条约束）。"""
    bn = {j["tag"] for j in _calib_jobs()}
    from diag.config import load_config
    from diag.run_exp1 import build_commands
    rep = {j["tag"] for j in build_commands(load_config("diag/config.yaml"),
                                            "calib", seeds=[0, 1], pfl="fedrep")}
    assert bn and rep and not (bn & rep)


def test_cost_estimate_reads_the_actual_commands_not_the_config():
    """dry-run 打印的机时是决定要不要缩规模的唯一依据，读错就整件事白算。

    旧实现直接读 `cfg.exp1.total_round / local_steps / eval_every`，
    于是 `--stage calib`（覆盖了这三个）会打印 200 轮 × 15 steps，
    而实际跑的是 80 轮 × {15,45,75}。
    """
    from diag.config import load_config
    from diag.run_exp1 import build_commands, estimate_cost
    cfg = load_config("diag/config.yaml")
    jobs = build_commands(cfg, "calib", seeds=[0, 1])
    cost = estimate_cost(jobs, cfg)
    calib = cfg.exp1.calibration
    assert cost["rounds_per_run"] == int(calib.total_round), \
        f"报的是 {cost['rounds_per_run']}，实际跑 {calib.total_round}"
    assert cost["rounds_per_run"] != int(cfg.exp1.total_round), \
        "还在读主力的 total_round"
    # 每个 run 有 80/2 = 40 个评估点
    assert cost["evaluations_per_run"] == int(calib.total_round) // int(calib.eval_every)


def test_cost_reports_a_span_when_runs_differ():
    """六个 run 的 local_steps 不同 → 单 run 字段必须报区间，不能挑一个数。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands, estimate_cost
    cfg = load_config("diag/config.yaml")
    cost = estimate_cost(build_commands(cfg, "calib", seeds=[0]), cfg)
    steps = [int(x) for x in cfg.exp1.calibration.local_steps]
    want = f"{min(steps)}-{max(steps)}"      # 区间端点从 config 推，不写死
    assert cost["local_steps_per_run"] == want, cost["local_steps_per_run"]
    assert isinstance(cost["local_batches_per_run"], str)


def test_cost_stays_a_single_number_for_the_uniform_main_grid():
    """反向锚点：主力格子的轮数/步数本来就一致，不该被改成区间。"""
    from diag.config import load_config
    from diag.run_exp1 import build_commands, estimate_cost
    cfg = load_config("diag/config.yaml")
    cost = estimate_cost(build_commands(cfg, "1", seeds=[0, 1]), cfg)
    assert cost["rounds_per_run"] == int(cfg.exp1.total_round)
    assert cost["local_steps_per_run"] == int(cfg.exp1.local_steps)
    assert isinstance(cost["local_batches_per_run"], int)


# ══════════════════════════════════════════════════════════════════════════
# Stage D 主力重跑：corner / arm / --list-only
# ══════════════════════════════════════════════════════════════════════════

def _cfg():
    from diag.config import load_config
    return load_config("diag/config.yaml")


def test_rho_grid_covers_the_points_the_supervisor_asked_for():
    """导师第 4 条点名要 0 / 0.3 / 0.7 / 0.9。ρ=0 是**无攻击对照** ——
    报告一直引用「no-attack baseline ASR」却从没进过 grid。"""
    rates = [float(r) for r in _cfg().exp1.poison_rates]
    for want in (0.0, 0.3, 0.7, 0.9):
        assert want in rates, f"ρ grid 缺 {want}：{rates}"
    assert 0.0 in rates and 1.0 in rates, "两个端点都要有"


def test_cross_sweep_size_is_exact():
    """8 个 ρ + 6 个 Nm − 1 个共用交叉点 = 13 格。
    交叉点重复会在并表时变成两个"独立"观测。"""
    from diag.run_exp1 import build_commands
    cfg = _cfg()
    jobs = build_commands(cfg, "1", seeds=[0])
    assert len(jobs) == 13, f"十字扫描应为 13 格，实际 {len(jobs)}"
    assert len({j["tag"] for j in jobs}) == 13, "有重复格子"


def test_corner_skips_cells_already_in_the_cross():
    """塌陷角与十字在 Nm=bad_num_fixed 那一列重叠，必须跳过。"""
    from diag.run_exp1 import build_commands
    cfg = _cfg()
    corner = build_commands(cfg, "corner", seeds=[0])
    cross = {j["tag"] for j in build_commands(cfg, "1", seeds=[0])}
    assert not ({j["tag"] for j in corner} & cross), "塌陷角与十字有重叠格子"
    n_bad = len([b for b in cfg.exp1.main.collapse_corner.bad_nums
                 if int(b) != int(cfg.exp1.bad_num_fixed)])
    n_rate = len(cfg.exp1.main.collapse_corner.poison_rates)
    assert len(corner) == n_bad * n_rate, f"塌陷角格数不对：{len(corner)}"


def test_control_arm_is_a_single_cell_per_arm():
    """对照臂只在十字交叉点上 —— 换的是 PFL 框架，不是剂量。"""
    from diag.run_exp1 import build_commands
    cfg = _cfg()
    for arm in ("fedbn", "fedrep"):
        jobs = build_commands(cfg, "arm", seeds=[0, 1, 2], pfl=arm)
        assert len(jobs) == 3, f"{arm}: 应为 3 个 seed × 1 格"
        doses = {(_arg(j["cmd"], "--bad-client-num"),
                  _arg(j["cmd"], "--poison-rate")) for j in jobs}
        assert len(doses) == 1, f"{arm}: 剂量不唯一 {doses}"


def test_the_two_arms_do_not_share_tags():
    from diag.run_exp1 import build_commands
    cfg = _cfg()
    bn = {j["tag"] for j in build_commands(cfg, "arm", seeds=[0, 1, 2])}
    rep = {j["tag"] for j in build_commands(cfg, "arm", seeds=[0, 1, 2],
                                            pfl="fedrep")}
    assert bn and rep and not (bn & rep)


def test_corner_and_arm_are_not_in_stage_all():
    """它们是按需单独提交的补丁，混进 all 会让「跑一遍 all」变成 100+ 个 run。"""
    from diag.run_exp1 import build_commands
    stages = {j["stage"] for j in build_commands(_cfg(), "all", seeds=[0])}
    assert "corner" not in stages and "arm" not in stages


def test_list_only_prints_one_clean_command_per_line():
    """job array 靠 `sed -n "${SLURM_ARRAY_TASK_ID}p"` 取行 ——
    stdout 里混进一个字（表头、注释、SKIP 标记）就整体错位。"""
    import contextlib, io
    from diag.run_exp1 import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(["--stage", "corner", "--seeds", "0", "--list-only"])
    assert rc == 0
    lines = [ln for ln in out.getvalue().splitlines() if ln]
    assert lines, "stdout 一行都没有"
    for ln in lines:
        assert ln.startswith("python -m diag.run_fl "), f"不是干净的命令行：{ln!r}"
        assert "#" not in ln and "SKIP" not in ln, f"混进了注释/标记：{ln!r}"
    # 说明性输出必须去 stderr
    assert "实验 1" in err.getvalue(), "表头没有走 stderr"
    assert "实验 1" not in out.getvalue(), "表头污染了 stdout"


def test_list_only_and_execute_are_mutually_exclusive():
    from diag.run_exp1 import main
    try:
        main(["--stage", "corner", "--seeds", "0", "--list-only", "--execute"])
    except SystemExit as e:          # argparse.error -> SystemExit(2)
        assert e.code == 2
        return
    raise AssertionError("--list-only 与 --execute 同时给应当报错")


# ══════════════════════════════════════════════════════════════════════════
# 标定网格必须覆盖导师要的 3–5 local epoch
# ══════════════════════════════════════════════════════════════════════════
CIFAR10_TRAIN_N = 50000     # main.py:59 torchvision.datasets.CIFAR10(train=True)


def _steps_per_epoch(exp1) -> int:
    """1 个 local epoch = 多少个 local_steps。

    `client.py:44` 的 `local_fine_tuning(iter_nums)` 逐 batch 循环，
    `fetch_data` 用完自动重开一轮 —— 所以 local_steps **就是 batch 数**。
    每客户端训练样本 = 50000 / client_num（main.py:65 的整除），
    DataLoader 是 batch_size=32 + drop_last=True（main.py:78-80）。
    """
    per_client = CIFAR10_TRAIN_N // int(exp1.client_num)
    return per_client // 32


def test_steps_per_epoch_matches_the_partition_formula():
    """换算的分母跟着 config 走 —— client_num 一改，下面那条断言的
    epoch 数就得跟着变，不能写死。"""
    exp1 = _cfg().exp1
    assert _steps_per_epoch(exp1) == 39, (
        f"client_num={exp1.client_num} 下每 epoch 是 "
        f"{_steps_per_epoch(exp1)} steps，注释里的换算要同步更新")


def test_calibration_grid_reaches_the_supervisor_range():
    """导师意见 #5 要的是 **3–5 local epoch**。

    早先网格写成 [15, 45, 75]，上限才 1.9 epoch —— **根本没进那个区间**，
    标定跑完也回答不了导师的问题。这条挡住它重演。
    """
    exp1 = _cfg().exp1
    spe = _steps_per_epoch(exp1)
    epochs = sorted(float(s) / spe for s in exp1.calibration.local_steps)
    assert max(epochs) >= 5.0 - 0.05, \
        f"网格上限只有 {max(epochs):.2f} epoch，导师要到 5"
    assert any(2.95 <= e <= 3.05 for e in epochs), \
        f"网格里没有 3 epoch 那一档：{[round(e, 2) for e in epochs]}"
    assert min(epochs) < 1.0, \
        f"缺少现状基线（<1 epoch）那一档：{[round(e, 2) for e in epochs]}"


def test_calibration_grid_is_sorted_and_unique():
    steps = [int(s) for s in _cfg().exp1.calibration.local_steps]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)


# ══════════════════════════════════════════════════════════════════════════
# Exp 1 的主力臂是 fedrep（与 Exp 3 对齐），fedbn 只作对照
# ══════════════════════════════════════════════════════════════════════════
def test_submitters_default_to_the_fedrep_arm():
    """用户决策：Exp 1 改成和 Exp 3 一样的 FedRep。

    `run_exp1.py` 自己的默认值仍是 fedbn（为了让 2026-08 之前的 fedbn 产物
    在 --skip-existing 下仍认得出，那些 tag 不带后缀），所以**提交器必须
    显式传 --pfl**，不能依赖那个默认值。
    """
    from pathlib import Path
    for name in ("submit_exp1.sh", "run_calib.sbatch"):
        src = (ROOT / "diag" / name).read_text(encoding="utf-8")
        assert 'PFL="${PFL:-fedrep}"' in src, f"{name} 的默认臂不是 fedrep"
        assert "--pfl" in src, f"{name} 没有显式传 --pfl"
