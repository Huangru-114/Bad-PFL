"""``diag.run_exp1`` 的单元测试（命令生成 + --skip-existing 的判定）。

重点盯两件容易错的事：
1. 重建的 CSV 路径必须与 run_fl 真实写出的名字逐字一致，否则 --skip-existing
   会永远判"不存在"而白跑，或误判"存在"而漏跑。
2. 只有带论文口径列 asr_paper_all 的 CSV 才算"可复用"；旧口径 CSV 必须重跑。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
