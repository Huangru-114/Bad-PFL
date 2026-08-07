"""配置加载与确定性设置。

所有超参集中在 ``diag/config.yaml``；本模块只负责读取、点号访问与随机性初始化。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

__all__ = ["DEFAULT_CONFIG_PATH", "Cfg", "load_config", "set_all_seeds",
           "make_select_rule"]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class Cfg(dict):
    """支持点号访问的 dict（嵌套 dict 自动包装）。

    只读语义：允许赋值以便 CLI 覆盖，但不做深拷贝保护。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - 便于定位配置缺失
            raise AttributeError(
                f"配置项 '{name}' 不存在；可用键: {sorted(self.keys())}") from exc
        return Cfg(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: Optional[os.PathLike] = None) -> Cfg:
    """读取 YAML 配置。``path`` 为 None 时使用 ``diag/config.yaml``。"""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件 {path} 的顶层必须是 mapping")
    return Cfg(raw)


def set_all_seeds(seed: int, cudnn_deterministic: bool = True,
                  cudnn_benchmark: bool = False) -> Dict[str, int]:
    """播种 **torch / numpy / python-random** 三条随机流。

    这是对原仓库的关键修复：``utils.set_random_seed``（utils.py:8-13）只播种了
    torch，而数据划分完全依赖 ``np.random``（main.py:67, utils.py:65/71/74），
    客户端顺序依赖 ``random.shuffle``（main.py:95）。不补这两条种子，
    数据划分不可复现，clean/attack 对照实验也就无从谈起。

    返回实际使用的种子字典，便于写进 meta.json 存档。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    return {"python_random": seed, "numpy": seed, "torch": seed}


def make_select_rule(nums: int, seed: int):
    """构造**与全局 RNG 隔离**的客户端采样规则。

    为什么必须隔离：原实现 ``utils.random_select``（utils.py:23-26）用的是全局
    torch RNG，而攻击 run 在训练过程中会额外消耗 RNG（``Autoencoder`` 初始化、
    每次 ``pgd_attack`` 的 ``uniform_`` 随机起点、每轮 30 次生成器迭代）。
    结果是 clean run 与 attack run 每轮选中的客户端集合**完全不同** ——
    两组的"每个客户端被训练了多少次"对不上，对照实验失效。

    这里给采样一条只由 seed 决定的独立 ``torch.Generator``，
    使采样序列在 clean/attack 两种模式下逐轮一致。
    采样**分布**与原实现相同（均匀无放回），攻击语义不受影响。
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def select(server, clients):  # 签名与 utils.random_select 兼容
        count = int(nums * len(clients)) if isinstance(nums, float) else int(nums)
        count = max(0, min(count, len(clients)))
        return torch.randperm(len(clients), generator=generator)[:count].tolist()

    return select
