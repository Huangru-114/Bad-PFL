"""`diag.read_calibration` 的判据 —— 与 tf-dpfl 侧同一套，必须给出同样的答案。

**这个模块能在本地真跑**（纯 stdlib + csv 夹具），不像 test_wall_clock 只能扫源码。

盯的是同一件事：**判不出来就报 None**。给一个「看着像」的平台轮，等于把
「80 轮还没收敛」这个结论偷偷换成一个数字，而那正是 Stage B 要回答的问题。
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from diag.read_calibration import (TAG_RE, _f, _plateau_scan,  # noqa: E402
                                   cost_model, eval_spacing, load_cells,
                                   plateau_round, project_run)


def _rows(col, values, start=2, every=2, **extra):
    out = []
    for i, v in enumerate(values):
        r = {"round": str(start + i * every), col: "" if v is None else str(v)}
        for k, seq in extra.items():
            r[k] = "" if seq[i] is None else str(seq[i])
        out.append(r)
    return out


# ── 平台判据 ───────────────────────────────────────────────────────────────
def test_plateau_found_when_curve_flattens():
    rows = _rows("mta_personalized", [0.10, 0.30, 0.50, 0.70, 0.70, 0.70, 0.70])
    # round = 2,4,6,8,10,12,14；从第 4 个点（round 8）起都在带内
    assert plateau_round(rows, "mta_personalized", tol=0.01) == 8


def test_plateau_is_none_when_still_climbing():
    rows = _rows("mta_personalized", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert plateau_round(rows, "mta_personalized", tol=0.01) is None


def test_plateau_requires_staying_inside_the_band():
    """中途掉出带外 → 平台不能算在第一个点上。

    （夹具从 5 点加长到 10 点：新的尾窗排除要求平台之后 ≥ tail+1 个点，
    5 个点时"平台在第 3 点"与"最后 3 点碰巧平"本来就无法区分。
    加长不改变本测试的意图 —— 它测的是**中途的坑会不会被忽略**。）
    """
    rows = _rows("mta_personalized", [0.70, 0.40] + [0.70] * 8)
    assert plateau_round(rows, "mta_personalized", tol=0.01) == 6
    assert plateau_round(rows, "mta_personalized", tol=0.01) != 2


def test_plateau_is_none_with_too_few_points():
    assert plateau_round(_rows("mta_personalized", [0.7, 0.7, 0.7]),
                         "mta_personalized", 0.01) is None
    assert plateau_round([], "mta_personalized", 0.01) is None


def test_nan_and_blank_are_skipped_not_read_as_zero():
    """`track.py` 把没量到的值写成 nan，pandas 写 CSV 时也可能留空。
    两者都必须跳过 —— 当成 0 会把平台推到最后，或伪造一段假的爬升。"""
    assert _f("nan") is None and _f("") is None and _f(None) is None
    assert _f("0.0") == 0.0, "真正的 0 要保留 —— 只有 nan/空才是无定义"
    rows = _rows("asr_paper_filtered_benign", [0.70, None, 0.70, 0.70, 0.70, 0.70])
    assert plateau_round(rows, "asr_paper_filtered_benign", tol=0.01) == 2


# ── 代价：单位成本，而不是求和 ─────────────────────────────────────────────
def test_cost_reports_per_unit_costs_not_sums():
    """**这是 2026-09 的实测教训。**

    植入行只在**评估轮**写（track.py 的 _on_round_end 每轮算 _t_train_s，
    但行是 _evaluate_now 写的），每行携带**它自己那一轮**的训练秒数。
    于是 sum(train_wall_s) 覆盖的是行数那么多轮，**不是全部轮数** ——
    80 轮 / eval_every=2 的 run 会把训练时间少算一半，而 T_main 正是从它算的。
    正确的单位量是 mean，再乘轮数外推。
    """
    rows = _rows("mta_personalized", [0.7] * 5,
                 train_wall_s=[10.0] * 5, eval_wall_s=[2.0] * 5)
    c = cost_model(rows)                       # round = 2,4,6,8,10
    assert c["s_per_round"] == 10.0
    assert c["s_per_eval"] == 2.0
    assert c["eval_every"] == 2
    assert c["last_round"] == 10
    # 外推到 10 轮 = 100 s，而不是「5 行 × 10 = 50 s」
    assert c["train_s_est"] == 100.0
    assert c["eval_s"] == 10.0
    assert c["run_s_est"] == 110.0


def test_summing_train_would_have_undercounted_by_eval_every():
    """反向锚点：明确记下旧算法给出的（错的）数，防止有人改回去。"""
    rows = _rows("mta_personalized", [0.7] * 5,
                 train_wall_s=[10.0] * 5, eval_wall_s=[2.0] * 5)
    c = cost_model(rows)
    naive_sum = 5 * 10.0                       # 旧实现：sum(train_wall_s)
    assert naive_sum == 50.0
    assert c["train_s_est"] == naive_sum * c["eval_every"], \
        "外推倍数就是 eval_every —— 这正是旧实现漏掉的那一倍"


def test_cost_is_none_not_zero_when_columns_absent():
    """旧 CSV 没有时间列 → None。0 会被读成「不花时间」，
    而墙钟正是 Stage B 唯一要读的东西。"""
    rows = _rows("mta_personalized", [0.7] * 4)
    c = cost_model(rows)
    assert c["s_per_round"] is None and c["s_per_eval"] is None
    assert c["train_s_est"] is None and c["eval_s"] is None and c["run_s_est"] is None


def test_cost_is_partial_when_only_train_is_recorded():
    rows = _rows("mta_personalized", [0.7] * 3, train_wall_s=[10.0] * 3)
    c = cost_model(rows)
    assert c["s_per_round"] == 10.0 and c["train_s_est"] == 60.0
    assert c["eval_s"] is None
    assert c["run_s_est"] is None, "缺一半就不该给总数 —— 会被当成完整墙钟读"


def test_eval_spacing_is_read_from_the_data_not_the_config():
    assert eval_spacing(_rows("m", [0.1] * 5, start=5, every=5)) == 5
    assert eval_spacing(_rows("m", [0.1])) is None      # 一个点推不出间隔


def test_project_run_scales_train_with_rounds_and_eval_with_points():
    """T_main：标定跑 80 轮/ev=2，主力跑 200 轮/ev=5 —— 评估是固定成本、
    训练随轮数走，两者比例完全不同，不能直接把标定的总数当 T_main。"""
    c = cost_model(_rows("m", [0.7] * 5, train_wall_s=[10.0] * 5,
                         eval_wall_s=[2.0] * 5))
    # 200 轮 × 10 s + (200//5=40) 次评估 × 2 s = 2000 + 80
    assert project_run(c, 200, 5) == 2080.0
    assert project_run(cost_model(_rows("m", [0.7] * 3)), 200, 5) is None


# ── 文件名解析 ─────────────────────────────────────────────────────────────
def test_tag_regex_reads_steps_seed_and_arm():
    m = TAG_RE.search("exp_ij_implantation_fedavg_attack_a0.5_s1_e1calib_steps45_s1.csv")
    assert m and m.group("steps") == "45" and m.group("seed") == "1"
    m2 = TAG_RE.search("exp_ij_implantation_fedavg_attack_a0.5_s0_e1calib_fedrep_steps75_s0.csv")
    assert m2 and m2.group("arm") == "fedrep" and m2.group("steps") == "75"


def test_tag_regex_ignores_main_grid_csvs():
    """主力格子（e1_bad4_rho0p1_s0）绝不能被当成标定格并进这张表 ——
    它们的预算不同，混进来直接把「谁最快到平台」的结论搞错。"""
    assert not TAG_RE.search("exp_ij_implantation_fedavg_attack_a0.5_s0_e1_bad4_rho0p1_s0.csv")
    assert not TAG_RE.search("exp_ij_implantation_fedavg_attack_a0.5_s0_e1b_sched_burst_s0.csv")


def test_load_cells_reads_only_calibration_csvs():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        def _write(name, rows):
            with open(d / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
        body = [{"round": "2", "mta_personalized": "0.5"}]
        _write("exp_ij_implantation_fedavg_attack_a0.5_s0_e1calib_steps15_s0.csv", body)
        _write("exp_ij_implantation_fedavg_attack_a0.5_s0_e1_bad4_rho0p1_s0.csv", body)
        cells = load_cells(d)
        assert len(cells) == 1
        assert cells[0]["local_steps"] == 15 and cells[0]["seed"] == 0
        assert cells[0]["arm"] == "fedbn"


# ── 平台的两道排除（2026-09 实测教训）───────────────────────────────────────
def test_plateau_in_the_tail_window_is_rejected():
    """`final` 就是最后 3 个点算出来的 —— 平台落在那里等于自证。

    实测：第一批标定四档全部报出 72–80 的"平台"，而 MTA 还在 0.29 → 0.65 一路爬。
    """
    # 一路爬，只有最后 3 个点碰巧在 ±0.01 内
    rows = _rows("mta_personalized",
                 [0.10, 0.20, 0.30, 0.40, 0.50, 0.600, 0.605, 0.607])
    r, why = _plateau_scan(rows, "mta_personalized", tol=0.01)
    assert r is None
    assert "只剩" in why or "覆盖" in why


def test_plateau_must_cover_a_meaningful_share_of_the_run():
    """「±0.01 内待了几轮」是走得慢，不是收敛：平坦段要覆盖最后 ≥25% 的轮数。"""
    vals = [0.05 * i for i in range(1, 17)] + [0.80] * 4   # 20 点，平台在第 17 点
    rows = _rows("mta_personalized", vals)                 # round 2..40
    r, why = _plateau_scan(rows, "mta_personalized", tol=0.01)
    assert r is None and "覆盖" in why
    # 放宽到 5% 就该判出来 —— 证明拦住它的确实是这条规则，而不是别的
    r2, _ = _plateau_scan(rows, "mta_personalized", tol=0.01, frac=0.05)
    assert r2 == 32


def test_a_real_plateau_is_still_reported():
    """反向锚点：真平台不能被这两道排除误杀。"""
    rows = _rows("mta_personalized", [0.1, 0.3, 0.5] + [0.70] * 9)   # round 2..24
    r, why = _plateau_scan(rows, "mta_personalized", tol=0.01)
    assert r == 8 and why is None


def test_reason_is_given_when_never_flat():
    rows = _rows("mta_personalized", [0.1 * i for i in range(1, 9)])
    r, why = _plateau_scan(rows, "mta_personalized", tol=0.01)
    assert r is None and "仍未进入" in why


# ── glob 只认 implantation（run_fl 每个 run 还写一份逐 edge CSV）─────────────
def test_load_cells_ignores_the_per_edge_csv_of_the_same_run():
    """`run_fl.py:670` 另写 `exp_ij_edge_<run_id>.csv`，**尾巴与 implantation 同名**。

    实测：`*e1calib*.csv` 把它一起收进来 → 6 个格子印成 12 行，
    一半因为没有 mta/asr/时间列而全是 n/a。
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        def _write(name, rows):
            with open(d / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)

        tag = "attack_a0.5_s0_e1calib_fedrep_steps117_s0.csv"
        _write(f"exp_ij_implantation_fedavg_{tag}",
               [{"round": "2", "mta_personalized": "0.5"}])
        _write(f"exp_ij_edge_fedavg_{tag}",
               [{"round": "2", "client_id": "0", "asr_targeted": "0.1"}])
        cells = load_cells(d)
        assert len(cells) == 1, f"逐 edge 文件被当成了一个格子：{[c['path'].name for c in cells]}"
        assert cells[0]["arm"] == "fedrep" and cells[0]["local_steps"] == 117
