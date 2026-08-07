"""``make_eval_loader`` 的回归测试。

背景：原仓库在**测试集**上也用了 ``drop_last=True``（main.py:83），
这是训练约定泄漏到评估路径。后果是每个客户端静默丢掉最多
``batch_size - 1`` 个样本，且丢弃比例随客户端样本数变化 ——
在高异构下会给指标注入系统性偏差。

这是典型的会复发的 bug，所以单独立一个测试文件盯着它。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from diag.probe import make_eval_loader


def _dataset(n: int) -> TensorDataset:
    return TensorDataset(torch.arange(n, dtype=torch.float32).view(n, 1),
                         torch.zeros(n, dtype=torch.long))


def test_no_samples_dropped_when_not_divisible():
    """核心回归：样本数不能整除 batch_size 时，遍历总数必须等于数据集大小。"""
    for n, batch_size in ((100, 32), (96, 32), (7, 3), (1, 8), (999, 256)):
        loader = make_eval_loader(_dataset(n), batch_size=batch_size)
        total = sum(batch[0].shape[0] for batch in loader)
        assert total == n, (
            f"n={n}, batch_size={batch_size}: 遍历得到 {total} 个样本，"
            f"应为 {n}（drop_last 泄漏？）")


def test_drop_last_is_false():
    loader = make_eval_loader(_dataset(100), batch_size=32)
    assert loader.drop_last is False


def test_shuffle_is_false_for_determinism():
    """kNN 的邻居集合必须可复现，所以评估 loader 不能 shuffle。"""
    loader = make_eval_loader(_dataset(50), batch_size=8)
    first = torch.cat([batch[0] for batch in loader])
    second = torch.cat([batch[0] for batch in loader])
    assert torch.equal(first, second)
    assert torch.equal(first.flatten(), torch.arange(50, dtype=torch.float32))


def test_contrast_with_original_training_loader_behaviour():
    """对照：drop_last=True 确实会丢样本 —— 说明这个测试不是无的放矢。"""
    dataset = _dataset(100)
    dropping = DataLoader(dataset, batch_size=32, drop_last=True)
    assert sum(batch[0].shape[0] for batch in dropping) == 96
    assert sum(batch[0].shape[0]
               for batch in make_eval_loader(dataset, batch_size=32)) == 100
