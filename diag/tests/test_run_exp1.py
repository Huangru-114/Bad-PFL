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
