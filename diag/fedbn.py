"""FedBN 的 key 判定 —— 全仓库唯一的一份。

# 为什么单独成一个模块

``pfl.py::use_fedbn`` 把过滤规则写在两个闭包里（``pfl.py:8`` 与 ``pfl.py:17``），
外部无法 import。诊断侧有三处需要同一条规则：

- ``diag/exp_f.py``：重建"借-BN 全局模型"时决定哪些 key 换成供体的；
- ``diag/exp_f_precheck.py``：证明全局模型的 BN 从未更新；
- ``diag/invariant_agg.py``：Invariant Aggregator 只作用于共享参数。

各写一份必然漂移，而漂移是**静默**的 —— 判错几个 key 不会报错，只会让结果
悄悄变成别的东西。所以这里收敛成一个函数，其余模块一律 import。

# 规则本身

``pfl.py:8`` / ``pfl.py:17`` 的条件逐字是::

    if "bn" in key or "shortcut.1" in key:

**刻意不写成"更正确"的实现**（例如按模块类型判断 ``nn.BatchNorm2d``）——
这里要复现的是仓库**实际使用**的那条规则。任何偏离都会让诊断结论与真实训练
行为对不上号。ResNet 的全部 BN 相关 key 都匹配它：``bn1.*`` / ``layer*.bn*.*``
（含 ``running_mean`` / ``running_var`` / ``num_batches_tracked``）以及
下采样分支的 ``layer*.0.shortcut.1.*``（``resnet.py:22,25,32``）。

# 一个必须知道的后果

``fedbn_update`` 挂在 ``before_update_global``，把这些 key 从 ``server.update``
中 pop 掉；随后 ``server.py:34`` 的 ``load_state_dict(..., strict=False)``
就再也碰不到它们。**因此 ``server.global_model`` 的 BN 参数与 running stats
从初始化起全程不变。** 这是 FedBN 的固有语义（个性化模型 = 共享参数 +
各自私有 BN），不是 bug，但它意味着 ``global.pt`` 不能被当作一个可独立使用的
模型 —— 见 ``diag/README.md`` §4b。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

__all__ = ["is_fedbn_private_key", "split_keys", "FEDBN_KEY_SUBSTRINGS"]

# 与 pfl.py:8 / pfl.py:17 逐字一致
FEDBN_KEY_SUBSTRINGS: Tuple[str, ...] = ("bn", "shortcut.1")


def is_fedbn_private_key(key: str) -> bool:
    """该 key 在 FedBN 下是否为客户端私有（不参与聚合）。"""
    return any(token in key for token in FEDBN_KEY_SUBSTRINGS)


def split_keys(state: Dict[str, torch.Tensor]
               ) -> Tuple[List[str], List[str], List[str]]:
    """把 state_dict 的 key 分成三类。

    Returns
    -------
    ``(shared_float, shared_nonfloat, fedbn_private)``

    - ``shared_float``：参与聚合、且能做 sign / trimmed-mean 的浮点张量。
    - ``shared_nonfloat``：参与聚合但**不是浮点**的张量（整数 buffer 等）。
      符号与顺序统计在它们上面没有意义，调用方应退回简单平均并如实记录
      （陷阱 4）。当前的 ResNet 上这一类应当为空 —— 唯一的整数 buffer
      ``num_batches_tracked`` 的 key 里含 ``bn``，已被归入 fedbn_private。
    - ``fedbn_private``：FedBN 会 pop 掉的 key。
    """
    shared_float: List[str] = []
    shared_nonfloat: List[str] = []
    private: List[str] = []
    for key in state:
        if is_fedbn_private_key(key):
            private.append(key)
            continue
        value = state[key]
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            shared_float.append(key)
        else:
            shared_nonfloat.append(key)
    return sorted(shared_float), sorted(shared_nonfloat), sorted(private)
