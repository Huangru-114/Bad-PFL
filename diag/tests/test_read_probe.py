"""`diag.read_probe` —— FedRep 探针的读数器。

盯的两件事与 `test_read_calibration` 相同：**判不出来就留空**，
以及**不要把同一个 run 的第二份 CSV 当成另一个格子**（`run_fl.py:670`
每个 run 还写一份 `exp_ij_edge_<run_id>.csv`，尾巴与 implantation 同名）。
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from diag.read_probe import TAG_RE, _f, load, tail_mean


def _write(d: Path, name, rows):
    with open(d / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_tag_regex_reads_name_arm_and_steps():
    m = TAG_RE.search("exp_ij_implantation_fedavg_attack_a0.5_s0_probe_B2_fedrep_s90.csv")
    assert m and m.group("name") == "B2" and m.group("arm") == "fedrep"
    assert m.group("steps") == "90"


def test_tag_regex_ignores_calibration_and_main_grid():
    """探针与标定/主力是三套不同预算，混进同一张表结论就错了。"""
    for other in ("exp_ij_implantation_fedavg_attack_a0.5_s0_e1calib_fedrep_steps117_s0.csv",
                  "exp_ij_implantation_fedavg_attack_a0.5_s0_e1_bad4_rho0p1_s0.csv"):
        assert not TAG_RE.search(other)


def test_load_ignores_the_per_edge_csv_of_the_same_run():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        body = [{"round": "10", "mta_personalized": "0.5", "asr_paper_benign": "0.9"}]
        tail = "attack_a0.5_s0_probe_A_fedbn_s15.csv"
        _write(d, f"exp_ij_implantation_fedavg_{tail}", body)
        _write(d, f"exp_ij_edge_fedavg_{tail}", body)
        cells = load(d)
        assert len(cells) == 1, [c["path"].name for c in cells]
        assert cells[0]["name"] == "A" and cells[0]["steps"] == 15


def test_tail_mean_uses_the_last_n_present_values():
    rows = [{"m": str(v)} for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert tail_mean(rows, "m", 3) == (0.3 + 0.4 + 0.5) / 3


def test_tail_mean_skips_nan_and_blank_rather_than_reading_them_as_zero():
    rows = [{"m": "0.4"}, {"m": "nan"}, {"m": ""}, {"m": "0.6"}]
    assert tail_mean(rows, "m", 3) == 0.5          # 只有 0.4 与 0.6 有值
    assert tail_mean([{"m": ""}], "m", 3) is None  # 一个都没有 → None，不是 0
    assert _f("0.0") == 0.0, "真正的 0 要保留 —— 只有 nan/空才是无定义"


def test_missing_column_is_none_not_zero():
    rows = [{"round": "10", "mta_personalized": "0.5"}]
    assert tail_mean(rows, "asr_paper_benign", 3) is None
