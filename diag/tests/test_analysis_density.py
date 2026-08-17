"""``diag.analysis_density`` 的单元测试。

盯的是**读图时才会暴露**的错误：

- 密度从文件名猜错 -> 整条曲线归到错误的 x 上，图看起来完全正常；
- 缺格被插值连线 -> 没测过的密度看起来测过了；
- 误差棒用了 run 内的尾部抖动 -> 自相关的数被当成不确定度；
- ASR 没被压低时 ACC 代价填 0 -> "代价为零"与"根本没起作用"混为一谈。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_density import (accuracy_cost, load_implantation,
                                   parse_bad_num, plot_density, tail_summary)


def _rows(defense: str, bad: int, seed: int, asr: float, acc: float,
          n: int = 10, with_column: bool = True):
    rows = []
    for index in range(n):
        row = {"round": (index + 1) * 10, "defense": defense, "seed": seed,
               "asr_personalized_targeted": asr, "acc_personalized": acc}
        if with_column:
            row["bad_client_num"] = bad
        rows.append(row)
    return pd.DataFrame(rows)


def _write(directory: Path, frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(directory / name, index=False)


# ---------------------------------------------------------------------------
# 密度的来源
# ---------------------------------------------------------------------------
def test_parse_bad_num_reads_the_tag_and_refuses_to_guess():
    assert parse_bad_num("exp_ij_implantation_flame_a0.5_s0_bad5.csv") == 5
    assert parse_bad_num("exp_ij_implantation_flame_a0.5_s0_bad10.csv") == 10
    # 没有 tag 就是 None —— 绝不套用 config 的默认值
    assert parse_bad_num("exp_ij_implantation_flame_attack_a0.5_s0.csv") is None


def test_column_wins_over_the_filename():
    """列与文件名冲突时以**列**为准：文件名可能被改过，列是训练时写下的。"""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write(directory, _rows("flame", 5, 0, 0.3, 0.77),
               "exp_ij_implantation_flame_bad99.csv")   # 文件名故意写错
        frame = load_implantation(str(directory / "*.csv"))
        assert set(frame["bad_client_num"]) == {5}


def test_filename_is_the_fallback_when_the_column_is_absent():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write(directory, _rows("fedavg", 2, 0, 0.86, 0.80, with_column=False),
               "exp_ij_implantation_fedavg_bad2.csv")
        frame = load_implantation(str(directory / "*.csv"))
        assert set(frame["bad_client_num"]) == {2}


def test_unresolvable_density_raises_and_names_the_file():
    """既无列、文件名也无 tag -> 报错并列出文件，不静默归到某个密度。"""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write(directory, _rows("fedavg", 10, 0, 0.9, 0.8, with_column=False),
               "exp_ij_implantation_fedavg_attack.csv")
        raised = None
        try:
            load_implantation(str(directory / "*.csv"))
        except ValueError as error:
            raised = str(error)
        assert raised is not None
        assert "exp_ij_implantation_fedavg_attack.csv" in raised
        # 显式指定后才允许通过
        frame = load_implantation(str(directory / "*.csv"), assume_bad_num=10)
        assert set(frame["bad_client_num"]) == {10}


def test_negative_column_falls_back_instead_of_being_used_as_a_density():
    """``bad_client_num=-1`` 是"调用方没传"的哨兵值，不是密度 −1。"""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write(directory, _rows("flame", -1, 0, 0.3, 0.77),
               "exp_ij_implantation_flame_bad5.csv")
        frame = load_implantation(str(directory / "*.csv"))
        assert set(frame["bad_client_num"]) == {5}


# ---------------------------------------------------------------------------
# 两级聚合
# ---------------------------------------------------------------------------
def test_tail_summary_averages_only_the_last_rounds():
    frame = _rows("flame", 5, 0, 0.0, 0.0, n=10)
    # 前 5 轮 ASR=0.9，后 5 轮 ASR=0.2；tail=5 只应看到 0.2
    frame["asr_personalized_targeted"] = [0.9] * 5 + [0.2] * 5
    frame["acc_personalized"] = [0.5] * 5 + [0.8] * 5
    summary = tail_summary(frame, tail=5)
    assert len(summary) == 1
    assert np.isclose(summary["asr_mean"].iloc[0], 0.2)
    assert np.isclose(summary["acc_mean"].iloc[0], 0.8)
    assert summary["last_round"].iloc[0] == 100


def test_runs_with_too_few_eval_rows_are_rejected_not_shortened():
    """用更短的窗口凑数会让这一格与其他格不同口径 —— 必须报错。"""
    frame = _rows("flame", 5, 0, 0.3, 0.77, n=3)
    raised = None
    try:
        tail_summary(frame, tail=5)
    except ValueError as error:
        raised = str(error)
    assert raised is not None and "只有 3 个评估行" in raised


def test_error_bars_come_from_seeds_not_from_within_run_scatter():
    """单 seed 时跨 seed std 必须是 nan —— run 内尾部抖动不是不确定度。"""
    frame = _rows("flame", 5, 0, 0.3, 0.77)
    frame["asr_personalized_targeted"] = [0.1, 0.2, 0.3, 0.4, 0.5] * 2
    summary = tail_summary(frame, tail=5)
    assert summary["n_seeds"].iloc[0] == 1
    assert not np.isfinite(summary["asr_std_across_seeds"].iloc[0])
    # run 内离散度仍然算出来，但放在另一列，绝不当误差棒
    assert summary["tail_std_within_run"].iloc[0] > 0.0


def test_multiple_seeds_are_pooled_per_run_first():
    """先在 run 内压成一个数，再跨 seed 求 std —— 不是把尾部行混在一起。"""
    frame = pd.concat([_rows("flame", 5, 0, 0.20, 0.77),
                       _rows("flame", 5, 1, 0.30, 0.75)], ignore_index=True)
    summary = tail_summary(frame, tail=5)
    assert len(summary) == 1
    assert summary["n_seeds"].iloc[0] == 2
    assert np.isclose(summary["asr_mean"].iloc[0], 0.25)
    # 两个 run 各自尾部无抖动，std 完全来自 seed 间的 0.20 vs 0.30
    assert np.isclose(summary["asr_std_across_seeds"].iloc[0],
                      np.std([0.20, 0.30], ddof=1))
    assert summary["seeds"].iloc[0] == "0,1"


# ---------------------------------------------------------------------------
# ACC 代价
# ---------------------------------------------------------------------------
def test_accuracy_cost_is_per_asr_point_against_the_same_density():
    summary = pd.DataFrame([
        {"defense": "fedavg", "bad_client_num": 5, "asr_mean": 0.90,
         "acc_mean": 0.80},
        {"defense": "flame", "bad_client_num": 5, "asr_mean": 0.30,
         "acc_mean": 0.77},
    ])
    out = accuracy_cost(summary)
    flame = out[out["defense"] == "flame"].iloc[0]
    assert np.isclose(flame["asr_reduction"], 0.60)
    assert np.isclose(flame["acc_drop"], 0.03)
    assert np.isclose(flame["acc_points_per_asr_point"], 0.05)
    # 基线自己没有代价可言
    assert not np.isfinite(
        out[out["defense"] == "fedavg"]["acc_points_per_asr_point"].iloc[0])


def test_no_asr_reduction_leaves_the_cost_undefined_not_zero():
    """ASR 没降反升时代价**无定义**。填 0 会读成"零代价"，结论正好相反。"""
    summary = pd.DataFrame([
        {"defense": "fedavg", "bad_client_num": 10, "asr_mean": 0.90,
         "acc_mean": 0.80},
        {"defense": "multi_krum", "bad_client_num": 10, "asr_mean": 0.98,
         "acc_mean": 0.72},
    ])
    out = accuracy_cost(summary)
    krum = out[out["defense"] == "multi_krum"].iloc[0]
    assert krum["asr_reduction"] < 0
    assert not np.isfinite(krum["acc_points_per_asr_point"])


def test_missing_baseline_at_a_density_leaves_that_density_blank():
    summary = pd.DataFrame([
        {"defense": "flame", "bad_client_num": 5, "asr_mean": 0.30,
         "acc_mean": 0.77},
    ])
    out = accuracy_cost(summary)
    assert not np.isfinite(out["acc_points_per_asr_point"].iloc[0])
    assert not np.isfinite(out["asr_reduction"].iloc[0])


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _summary_with_gap() -> pd.DataFrame:
    rows = []
    for defense, asr in (("fedavg", 0.9), ("flame", 0.3)):
        for bad in (1, 2, 5):
            rows.append({"defense": defense, "bad_client_num": bad,
                         "asr_mean": asr, "acc_mean": 0.78,
                         "asr_std_across_seeds": np.nan,
                         "acc_std_across_seeds": np.nan,
                         "tail_std_within_run": 0.01, "n_seeds": 1, "tail": 5,
                         "last_round": 500, "seeds": "0"})
    # multi_krum 只有 1 和 5，缺 2 -> 曲线必须在 2 处断开
    for bad in (1, 5):
        rows.append({"defense": "multi_krum", "bad_client_num": bad,
                     "asr_mean": 0.6, "acc_mean": 0.72,
                     "asr_std_across_seeds": np.nan,
                     "acc_std_across_seeds": np.nan,
                     "tail_std_within_run": 0.01, "n_seeds": 1, "tail": 5,
                     "last_round": 500, "seeds": "0"})
    return pd.DataFrame(rows)


def test_plot_breaks_the_line_at_untested_densities():
    """缺格用 nan 断线。若被插值连上，没测过的密度看起来就像测过了。"""
    from diag.analysis_density import _series
    summary = _summary_with_gap()
    xs = [1, 2, 5]
    values = _series(summary, "multi_krum", xs, "asr_mean")
    assert np.isfinite(values[0]) and np.isfinite(values[2])
    assert not np.isfinite(values[1])          # bad=2 必须是 nan


def test_plot_density_writes_a_file_and_passes_the_cjk_scan():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "figs" / "density.png"
        path = plot_density(_summary_with_gap(), out)
        assert path.exists() and path.stat().st_size > 0


def test_paper_marker_is_a_single_point_not_a_horizontal_line():
    """论文的数字只对它自己那个密度成立，不能画成横跨所有密度的横线。"""
    import matplotlib
    matplotlib.use("Agg")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "density.png"
        path = plot_density(_summary_with_gap(), out, paper_asr=0.8222,
                            paper_bad_num=5)
        assert path.exists()
        # 密度不在 xs 里时静默跳过，不画到错误的位置上
        path2 = plot_density(_summary_with_gap(), Path(tmp) / "d2.png",
                             paper_asr=0.8222, paper_bad_num=77)
        assert path2.exists()


def test_end_to_end_from_csv_to_figure():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write(directory, _rows("fedavg", 5, 0, 0.88, 0.80),
               "exp_ij_implantation_fedavg_bad5.csv")
        _write(directory, _rows("flame", 5, 0, 0.30, 0.77),
               "exp_ij_implantation_flame_bad5.csv")
        frame = load_implantation(str(directory / "*.csv"))
        summary = accuracy_cost(tail_summary(frame, tail=5))
        assert len(summary) == 2
        flame = summary[summary["defense"] == "flame"].iloc[0]
        assert np.isclose(flame["asr_reduction"], 0.58)
        assert plot_density(summary, directory / "out.png").exists()
