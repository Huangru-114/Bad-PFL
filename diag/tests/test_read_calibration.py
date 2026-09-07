"""`diag.read_calibration` 的判据 —— 与 tf-dpfl 侧同一套，必须给出同样的答案。

**这个模块能在本地真跑**（纯 stdlib + csv 夹具），不像 test_wall_clock 只能扫源码。

盯的是同一件事：**判不出来就报 None**。给一个「看着像」的平台轮，等于把
「80 轮还没收敛」这个结论偷偷换成一个数字，而那正是 Stage B 要回答的问题。
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from diag.read_calibration import (TAG_RE, cost_up_to, load_cells,  # noqa: E402
                                   plateau_round, _f)


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
    rows = _rows("mta_personalized", [0.70, 0.40, 0.70, 0.70, 0.70])
    assert plateau_round(rows, "mta_personalized", tol=0.01) == 6


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


# ── 代价 ───────────────────────────────────────────────────────────────────
def test_cost_sums_measured_values():
    rows = _rows("mta_personalized", [0.7] * 5,
                 train_wall_s=[10.0] * 5, eval_wall_s=[2.0] * 5)
    c = cost_up_to(rows, 6)          # round 2,4,6 → 3 行
    assert c["train_s"] == 30.0 and c["eval_s"] == 6.0 and c["total_s"] == 36.0
    assert cost_up_to(rows, None)["total_s"] == 60.0


def test_cost_is_none_not_zero_when_columns_absent():
    """旧 CSV 没有时间列 → None。0 会被读成「不花时间」，
    而墙钟正是 Stage B 唯一要读的东西。"""
    rows = _rows("mta_personalized", [0.7] * 4)
    c = cost_up_to(rows, None)
    assert c["train_s"] is None and c["eval_s"] is None and c["total_s"] is None


def test_cost_is_partial_when_only_train_is_recorded():
    rows = _rows("mta_personalized", [0.7] * 3, train_wall_s=[10.0] * 3)
    c = cost_up_to(rows, None)
    assert c["train_s"] == 30.0
    assert c["eval_s"] is None
    assert c["total_s"] is None, "缺一半就不该给总数 —— 会被当成完整墙钟读"


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
