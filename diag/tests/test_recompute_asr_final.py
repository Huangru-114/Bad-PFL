"""``diag.recompute_asr_final`` 的单元测试（纯聚合逻辑）。

client_asr / recompute_run 依赖 torch + torchvision + checkpoint，只能在集群上
跑；这里只盯 `aggregate_tiers` 的 benign/all/malicious 划分、nan 处理与"空档留空"。
"""

from __future__ import annotations

import math

from diag.recompute_asr_final import aggregate_tiers


def _rows():
    return [
        {"is_malicious": False, "asr_std_filtered": 0.2, "asr_unfiltered": 0.3},
        {"is_malicious": False, "asr_std_filtered": 0.4, "asr_unfiltered": 0.5},
        {"is_malicious": True, "asr_std_filtered": 0.98, "asr_unfiltered": 0.99},
        {"is_malicious": True, "asr_std_filtered": 1.0, "asr_unfiltered": 1.0},
    ]


def test_aggregate_splits_benign_all_malicious():
    agg = aggregate_tiers(_rows(), "asr_std_filtered")
    assert abs(agg["benign"] - 0.3) < 1e-9          # (0.2+0.4)/2
    assert abs(agg["malicious"] - 0.99) < 1e-9       # (0.98+1.0)/2
    assert abs(agg["all"] - 0.645) < 1e-9            # mean of all four


def test_aggregate_empty_tier_is_nan_not_zero():
    benign_only = [r for r in _rows() if not r["is_malicious"]]
    agg = aggregate_tiers(benign_only, "asr_std_filtered")
    assert math.isnan(agg["malicious"])              # 没有恶意客户端 -> nan，不是 0
    assert not math.isnan(agg["benign"])


def test_aggregate_ignores_non_finite_values():
    rows = _rows() + [{"is_malicious": False,
                       "asr_std_filtered": float("nan"),
                       "asr_unfiltered": float("nan")}]
    agg = aggregate_tiers(rows, "asr_std_filtered")
    assert abs(agg["benign"] - 0.3) < 1e-9           # nan 行被忽略，不拉低均值
