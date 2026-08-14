"""``diag.analysis_f`` 的单元测试。

重点：
- 上三角以外必须是 ``NaN``，不能变成 0（陷阱 7）；
- 判据的三个分支各自可达，且"未能确定"能区分"没数据"与"落在规则空隙里"；
- 保留率在对角线值接近 0 时返回 nan，而不是一个巨大的假信号；
- 逐 t 的基线行（``gen_round_s`` 为 NaN）不得混进矩阵汇总。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diag.analysis_f import (DECAY_GAP, DECAY_RATIO, METRIC_ABS, METRIC_REL,
                             STABLE_GAP, classify_decay, decay_table,
                             matrix_pivot, plot_f1, plot_f2, plot_f3,
                             summarize_f)


def _raw(pattern, rounds=(200, 400, 600, 800, 1000)) -> pd.DataFrame:
    """构造一份原始 CSV 形态的表；``pattern(s, t) -> asr``。

    ``rounds`` 决定网格，进而决定哪些 gap 存在。这一点在测试里必须显式控制：
    判据要求 ``gap=200`` 与 ``gap=500`` 上的**实测值**，而步长 200 的网格
    根本取不到 gap=500 —— 这不是 bug，是 P2 稀疏网格的真实限制。
    """
    rounds = list(rounds)
    rows = []
    for cid in (0, 1):
        for t in rounds:
            base = 0.05
            rows.append({"gen_round_s": np.nan, "model_round_t": t,
                         "gap": np.nan, "perturb_mode": "none",
                         "eval_target": "personalized", "client_id": cid,
                         METRIC_ABS: base, METRIC_REL: 0.0,
                         "baseline_asr": base, "clean_acc": 0.7,
                         "error_rate": 0.3, "staleness": 2,
                         "alpha": 0.5, "seed": 0})
            for s in rounds:
                if t < s:
                    continue
                value = pattern(s, t)
                rows.append({"gen_round_s": s, "model_round_t": t,
                             "gap": t - s, "perturb_mode": "delta_only",
                             "eval_target": "personalized", "client_id": cid,
                             METRIC_ABS: value + base,
                             METRIC_REL: value,
                             "baseline_asr": base, "clean_acc": 0.7,
                             "error_rate": 0.5, "staleness": 2,
                             "alpha": 0.5, "seed": 0})
    return pd.DataFrame(rows)


def _flat(_s, _t):
    return 0.8


def _decaying(s, t):
    return 0.8 * np.exp(-(t - s) / 200.0)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def test_summarize_drops_per_t_baseline_rows():
    summary = summarize_f(_raw(_flat))
    # 只有 delta_only 的矩阵行进汇总；none 行的 gen_round_s 是 NaN，必须被排除
    assert summary["perturb_mode"].unique().tolist() == ["delta_only"]
    assert np.isfinite(summary["gen_round_s"]).all()
    assert len(summary) == 15          # 5x5 上三角
    assert (summary["n_clients"] == 2).all()


def test_summarize_keeps_staleness_as_a_covariate():
    summary = summarize_f(_raw(_flat))
    assert "staleness_max" in summary.columns
    assert int(summary["staleness_max"].max()) == 2


# ---------------------------------------------------------------------------
# 矩阵
# ---------------------------------------------------------------------------
def test_matrix_pivot_leaves_lower_triangle_nan_not_zero():
    matrix = matrix_pivot(summarize_f(_raw(_flat)), METRIC_ABS)
    array = matrix.to_numpy(dtype=float)
    for row, s in enumerate(matrix.index):
        for column, t in enumerate(matrix.columns):
            if t < s:
                assert np.isnan(array[row, column]), (s, t)
            else:
                assert np.isfinite(array[row, column]), (s, t)


# ---------------------------------------------------------------------------
# 衰减表与判据
# ---------------------------------------------------------------------------
def test_decay_table_retention_math():
    table = decay_table(summarize_f(_raw(_decaying)), METRIC_REL)
    row = table[table["gen_round_s"] == 200].iloc[0]
    assert abs(float(row["diagonal"]) - 0.8) < 1e-9
    expected = np.exp(-DECAY_GAP / 200.0)
    assert abs(float(row[f"retention_gap{DECAY_GAP}"]) - expected) < 1e-6


def test_decay_table_returns_nan_when_diagonal_is_degenerate():
    # 对角线 ~0 时比例没有意义；返回一个巨大的数会在图上造出假信号
    table = decay_table(summarize_f(_raw(lambda s, t: 1e-9)), METRIC_REL)
    assert table[f"retention_gap{DECAY_GAP}"].isna().all()


def test_classify_decay_reports_coadapt():
    table = decay_table(summarize_f(_raw(_decaying)), METRIC_REL)
    verdict = classify_decay(table, mature_s=500)
    assert "H_coadapt" in verdict


def test_classify_decay_reports_stable():
    # 需要 gap=500 存在 -> 网格步长必须能整除 500
    table = decay_table(summarize_f(_raw(_flat, rounds=(500, 750, 1000))),
                        METRIC_REL)
    verdict = classify_decay(table, mature_s=500)
    assert "H_stable" in verdict


def test_sparse_grid_cannot_reach_the_stable_gap():
    """P2 的稀疏网格（步长 200）取不到 gap=500，判据必须如实说"未能确定"。

    这是网格的真实限制，不是缺陷；此处固定住这个行为，防止将来有人为了
    "让判据出结论"而偷偷插值（陷阱 8）。
    """
    summary = summarize_f(_raw(_flat, rounds=(200, 400, 600, 800, 1000)))
    assert 500 not in set(summary["gap"])
    table = decay_table(summary, METRIC_REL)
    assert table[f"retention_gap{STABLE_GAP}"].isna().all()


def test_classify_decay_reports_early_fast_late_flat():
    """第三种情况：早期 s 衰减快、晚期平坦 —— 这**支持论文**，不得淡化。"""

    def pattern(s, t):
        scale = 60.0 if s < 500 else 100000.0   # 早期快衰减，晚期几乎平坦
        return 0.8 * np.exp(-(t - s) / scale)

    table = decay_table(summarize_f(_raw(pattern)), METRIC_REL)
    verdict = classify_decay(table, mature_s=500)
    assert "支持论文" in verdict
    assert "收敛时间尺度" in verdict


def test_classify_decay_distinguishes_no_data_from_inconclusive():
    # (a) 网格根本没有 gap=200 / gap=500 的格子
    sparse = summarize_f(_raw(_flat))
    sparse = sparse[sparse["gap"].isin([0, 400])]
    verdict = classify_decay(decay_table(sparse, METRIC_REL), mature_s=200)
    assert "未能确定" in verdict and "没有 gap" in verdict and "不插值" in verdict

    # (b) 两个 gap 都有数据，但保留率落在判据的空隙里（0.65：既不 <0.5 也不 >0.8）
    #     网格取 (500, 700, 1000)：s=500 时 gap 分别是 200 与 500
    table = decay_table(
        summarize_f(_raw(lambda s, t: 0.8 * (0.65 ** ((t - s) / DECAY_GAP)),
                         rounds=(500, 700, 1000))),
        METRIC_REL)
    row = table[table["gen_round_s"] == 500].iloc[0]
    assert np.isfinite(row[f"retention_gap{DECAY_GAP}"])
    assert np.isfinite(row[f"retention_gap{STABLE_GAP}"])
    verdict = classify_decay(table, mature_s=500)
    assert "未能确定" in verdict and "不向邻近结论靠拢" in verdict


def test_classify_decay_says_so_when_no_mature_generator():
    table = decay_table(summarize_f(_raw(_flat)), METRIC_REL)
    verdict = classify_decay(table, mature_s=99999)
    assert "未能确定" in verdict and "成熟生成器" in verdict


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def test_plots_render_without_cjk():
    """``_finish`` 内部会扫描 CJK，出现中文即抛异常，所以这里跑通即等于通过。"""
    summary = summarize_f(_raw(_decaying))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        produced = [plot_f1(summary, out / "f1.png", METRIC_ABS),
                    plot_f1(summary, out / "f1_rel.png", METRIC_REL),
                    plot_f2(summary, out / "f2.png", METRIC_REL),
                    plot_f3(summary, out / "f3.png", METRIC_ABS)]
        for path in produced:
            assert path.exists() and path.stat().st_size > 0
