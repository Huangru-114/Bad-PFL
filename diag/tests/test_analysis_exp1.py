"""``diag.analysis_exp1`` 的单元测试。

这个模块最大的风险是**把相关当成因果**、以及**只统计成功的 run**。
两者都会给出一张很干净的图和一个很强的结论，而结论是错的。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_exp1 import (crossing_table, dose_response, load_runs,
                                onset_analysis, persistence_table, plot_e1_1,
                                plot_e1_2, plot_e1_4, plot_e1b_1, plot_e1b_2,
                                threshold_verdict)


def _run(run_id, *, bad=4, rho=0.5, seed=0, schedule="continuous",
         asr=None, mta=None, active=None, rounds=None):
    rounds = rounds if rounds is not None else list(range(5, 105, 5))
    n = len(rounds)
    asr = asr if asr is not None else np.linspace(0.0, 0.9, n)
    mta = mta if mta is not None else np.linspace(0.2, 0.8, n)
    active = active if active is not None else [True] * n
    return pd.DataFrame({
        "round": rounds, "seed": seed, "bad_client_num": bad,
        "poison_rate": rho, "schedule": schedule,
        "attack_active_this_round": active,
        "asr_personalized_targeted": asr,
        "mta_personalized": mta,
        "clean_loss_personalized": np.linspace(2.3, 0.5, n),
        "run_id": run_id,
    })


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------
def test_load_rejects_old_csvs_missing_the_dose_columns():
    """缺 poison_rate 的旧 CSV 混进来会把所有剂量点挤成一个。"""
    with tempfile.TemporaryDirectory() as tmp:
        frame = _run("old").drop(columns=["poison_rate"])
        path = Path(tmp) / "exp_ij_implantation_old.csv"
        frame.to_csv(path, index=False)
        raised = None
        try:
            load_runs(str(Path(tmp) / "*.csv"))
        except KeyError as error:
            raised = str(error)
        assert raised is not None and "poison_rate" in raised


def test_load_derives_run_id_from_the_filename():
    with tempfile.TemporaryDirectory() as tmp:
        _run("x").drop(columns=["run_id"]).to_csv(
            Path(tmp) / "exp_ij_implantation_e1_bad4_rho0p5_s0.csv", index=False)
        frame = load_runs(str(Path(tmp) / "*.csv"))
        assert set(frame["run_id"]) == {"e1_bad4_rho0p5_s0"}


# ---------------------------------------------------------------------------
# 越过点：不能只统计成功的 run
# ---------------------------------------------------------------------------
def test_runs_that_never_cross_stay_in_the_table():
    """丢掉没越过的 run 会让阈值看起来比实际更普遍。"""
    frame = pd.concat([
        _run("a", asr=np.linspace(0.0, 0.9, 20)),
        _run("b", seed=1, asr=np.full(20, 0.02)),      # 从未越过
    ], ignore_index=True)
    table = crossing_table(frame, level=0.5)
    assert len(table) == 2
    never = table[~table["crossed"]].iloc[0]
    assert never["run_id"] == "b"
    # 没越过就是 nan，不是 0 —— 0 会被当成"第 0 轮就越过了"
    assert not np.isfinite(never["round_at_cross"])
    assert not np.isfinite(never["mta_at_cross"])


def test_crossing_picks_the_first_crossing():
    asr = np.array([0.1, 0.2, 0.6, 0.3, 0.8])
    frame = _run("a", rounds=[10, 20, 30, 40, 50], asr=asr,
                 mta=np.array([0.3, 0.4, 0.5, 0.6, 0.7]))
    table = crossing_table(frame, level=0.5)
    assert table["round_at_cross"].iloc[0] == 30
    assert np.isclose(table["mta_at_cross"].iloc[0], 0.5)


def test_verdict_is_undetermined_with_too_few_crossings():
    frame = pd.concat([_run("a", asr=np.full(20, 0.01)),
                       _run("b", seed=1, asr=np.full(20, 0.01))],
                      ignore_index=True)
    verdict = threshold_verdict(crossing_table(frame, 0.5), frame)
    assert "未能确定" in verdict["verdict"]
    assert verdict["n_never_crossed"] == 2


def test_verdict_detects_a_tight_mta_threshold():
    """各 run 在很不同的**轮次**、但很接近的 MTA 处起飞 -> MTA 更集中。"""
    runs = []
    # 三条 MTA 轨迹终点相同、**上升快慢差很多**，于是越过 MTA=0.6 的轮次
    # 差得很远，而越过时的 MTA 都是 0.6 左右。
    for index, curvature in enumerate([0.5, 1.0, 2.0]):
        n = 40
        rounds = list(range(5, 5 * n + 1, 5))
        mta = 0.9 * (np.arange(n) / (n - 1)) ** curvature
        # ASR 在 MTA 越过 0.6 之后才起飞 —— 与轮次无关
        asr = np.where(mta >= 0.6, 0.9, 0.02)
        runs.append(_run(f"r{index}", seed=index, rounds=rounds, asr=asr,
                         mta=mta))
    frame = pd.concat(runs, ignore_index=True)
    verdict = threshold_verdict(crossing_table(frame, 0.5), frame)
    # 必须比**零假设**更紧，而不只是比轮次更紧
    assert verdict["relative_spread_mta"] < verdict["null_relative_spread_mta"]
    assert "更紧的预测量" in verdict["verdict"]
    # 措辞必须自己声明不是因果
    assert "不是因果" in verdict["verdict"]


def test_verdict_detects_a_time_driven_pattern():
    """各 run 在同一**轮次**、但很不同的 MTA 处起飞 -> 时间驱动。"""
    runs = []
    for index, speed in enumerate([1.0, 0.5, 2.0]):
        n = 40
        rounds = list(range(5, 5 * n + 1, 5))
        mta = np.clip(np.linspace(0.1, 0.9, n) * speed, 0, 0.95)
        asr = np.where(np.array(rounds) >= 100, 0.9, 0.02)
        runs.append(_run(f"r{index}", seed=index, rounds=rounds, asr=asr,
                         mta=mta))
    frame = pd.concat(runs, ignore_index=True)
    verdict = threshold_verdict(crossing_table(frame, 0.5), frame)
    assert verdict["relative_spread_mta"] > verdict["null_relative_spread_mta"]
    assert "不成立" in verdict["verdict"]


# ---------------------------------------------------------------------------
# 1B
# ---------------------------------------------------------------------------
def test_onset_reports_the_mta_at_the_first_poisoned_round():
    n = 20
    active = [False] * 10 + [True] * 10
    mta = np.linspace(0.1, 0.9, n)
    frame = _run("late", schedule="late", active=active, mta=mta)
    table = onset_analysis(frame)
    assert np.isclose(table["mta_at_onset"].iloc[0], mta[10])
    assert table["n_active_eval_rounds"].iloc[0] == 10


def test_persistence_excludes_runs_that_never_stopped():
    """continuous 永远不停，放进来会在 k=0 处堆一堆与衰减无关的点。"""
    frame = pd.concat([
        _run("cont", schedule="continuous"),
        _run("early", schedule="early",
             active=[True] * 5 + [False] * 15),
    ], ignore_index=True)
    table = persistence_table(frame)
    assert set(table["schedule"]) == {"early"}
    assert table["rounds_since_attack_stopped"].min() == 0


def test_persistence_aligns_at_the_last_poisoned_round():
    active = [True] * 4 + [False] * 6
    rounds = list(range(10, 101, 10))
    frame = _run("early", schedule="early", rounds=rounds, active=active,
                 asr=np.array([.1, .3, .6, .9, .8, .7, .6, .5, .4, .3]),
                 mta=np.linspace(0.2, 0.8, 10))
    table = persistence_table(frame)
    # 最后一个投毒轮是 index 3 -> round 40
    assert table["rounds_since_attack_stopped"].tolist() == [0, 10, 20, 30,
                                                             40, 50, 60]
    assert np.isclose(table["asr"].iloc[0], 0.9)


def test_persistence_skips_runs_that_never_attacked():
    frame = _run("none", schedule="late", active=[False] * 20)
    assert persistence_table(frame).empty


# ---------------------------------------------------------------------------
# 剂量-响应
# ---------------------------------------------------------------------------
def test_dose_response_uses_the_tail():
    frame = _run("a", rounds=[10, 20, 30], asr=np.array([0.0, 0.0, 0.9]),
                 mta=np.array([0.1, 0.2, 0.3]))
    table = dose_response(frame, tail=1)
    assert np.isclose(table["asr"].iloc[0], 0.9)
    table3 = dose_response(frame, tail=3)
    assert np.isclose(table3["asr"].iloc[0], 0.3)


def test_dose_response_skips_runs_shorter_than_the_tail():
    frame = _run("a", rounds=[10], asr=np.array([0.5]), mta=np.array([0.5]),
                 active=[True])
    assert dose_response(frame, tail=5).empty


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _mixed_frame():
    return pd.concat([
        _run("a", bad=1, rho=0.5, seed=0),
        _run("b", bad=4, rho=0.5, seed=0),
        _run("c", bad=4, rho=0.5, seed=1),
        _run("d", bad=4, rho=0.1, seed=0, asr=np.full(20, 0.05)),
    ], ignore_index=True)


def test_all_figures_render_and_pass_the_cjk_scan():
    frame = _mixed_frame()
    verdict = threshold_verdict(crossing_table(frame, 0.5), frame)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert plot_e1_1(frame, out / "e1.png").exists()
        assert plot_e1_2(frame, out / "e2.png", verdict).exists()
        assert plot_e1_4(dose_response(frame, tail=3),
                         out / "e4.png").exists()


def test_schedule_figures_handle_the_degenerate_cases():
    """没有任何 run 停过攻击时，B2 要给出说明而不是一张空白图。"""
    frame = pd.concat([
        _run("cont", schedule="continuous"),
        _run("early", schedule="early", active=[True] * 5 + [False] * 15),
    ], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert plot_e1b_1(frame, out / "b1.png").exists()
        assert plot_e1b_2(persistence_table(frame), out / "b2.png").exists()
        # 空表也要出图（图上写明原因），不能抛异常
        assert plot_e1b_2(persistence_table(_run("cont")),
                          out / "b2empty.png").exists()


def test_saturating_mta_alone_does_not_count_as_a_threshold():
    """回归测试：MTA(t) 饱和会机械地压低 mta_at_cross 的离散度。

    这份数据里 ASR 起飞**纯粹由剂量决定**，各 run 共用同一条 MTA(t) 曲线,
    所以 MTA 不可能带来轮次之外的信息。朴素地比较
    ``relative_spread_mta < relative_spread_round`` 会在这里报出假阳性
    （实测就是这样发现的），必须靠零假设挡住。
    """
    n = 40
    rounds = list(range(5, 5 * n + 1, 5))
    # 共用的饱和 MTA 曲线
    mta = 0.85 * (1 - np.exp(-np.arange(n) / 9.0))
    runs = []
    # 增益都很小，于是四个 run 都在 MTA 已经**饱和**的后段才越过 0.5 ——
    # 轮次差得远（约 130 / 140 / 160 / 180），而那一段的 MTA 几乎不动。
    for index, gain in enumerate([0.0200, 0.0180, 0.0160, 0.0140]):
        asr = np.clip(np.cumsum(np.full(n, gain)), 0, 1.0)
        runs.append(_run(f"r{index}", bad=2 ** index, seed=0, rounds=rounds,
                         asr=asr, mta=mta))
    frame = pd.concat(runs, ignore_index=True)
    verdict = threshold_verdict(crossing_table(frame, 0.5), frame)
    # 朴素比较会说"MTA 更集中"——零假设必须把它挡回"未能确定"
    assert verdict["relative_spread_mta"] < verdict["relative_spread_round"]
    assert "未能确定" in verdict["verdict"]
    assert "饱和" in verdict["verdict"]


def test_1b_figures_hold_the_dose_fixed():
    """实验 1 的剂量扫描全是 continuous，混进来会让 continuous 那一栏
    塞满不同剂量的曲线 —— 看起来像方差大，其实是剂量在变。"""
    from diag.analysis_exp1 import restrict_to_common_dose
    frame = pd.concat([
        # 剂量扫描（都是 continuous，各种剂量）
        _run("e1_a", bad=1, rho=0.5), _run("e1_b", bad=8, rho=0.5),
        _run("e1_c", bad=4, rho=0.1),
        # 1B：固定剂量 bad=4 rho=0.5
        _run("e1b_cont", bad=4, rho=0.5, schedule="continuous"),
        _run("e1b_early", bad=4, rho=0.5, schedule="early",
             active=[True] * 5 + [False] * 15),
    ], ignore_index=True)
    kept, note = restrict_to_common_dose(frame)
    assert set(kept["run_id"]) == {"e1b_cont", "e1b_early"}
    assert "Nm=4" in note and "rho=0.5" in note
    assert "3" in note                      # 报出被排除的 run 数


def test_restrict_is_a_noop_without_other_schedules():
    from diag.analysis_exp1 import restrict_to_common_dose
    frame = pd.concat([_run("a", bad=1), _run("b", bad=8)], ignore_index=True)
    kept, note = restrict_to_common_dose(frame)
    assert len(kept) == len(frame) and note == ""
