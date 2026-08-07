"""Bad-PFL 触发器跨环境不变性诊断工具包。

本包只做**只读埋点与离线测量**，不改变 Bad-PFL 的攻击/训练语义。
对原仓库文件的改动（当前为零）逐条记录在 ``diag/PATCHES.md``。

导入副作用：把仓库根目录加入 ``sys.path``，使 ``import fba`` / ``import resnet``
等原仓库顶层模块可用（原仓库是扁平脚本布局，没有包结构）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPO_ROOT = _REPO_ROOT

__all__ = ["REPO_ROOT"]
