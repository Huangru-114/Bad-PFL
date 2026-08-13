"""P0 排查：验证实验 C / D 的结果是否有效。

本子包是**一次性排查工具**，不是常规测量层。它只做只读诊断（外加一次明确
标注的受控对照重训），产出 ``diag/audit/P0_AUDIT.md``。

设计原则：**判定由写死的规则产生，不由事后措辞决定。**
所有阈值集中在 ``diag/audit/common.py`` 的 ``THRESHOLDS`` 中，报告同时打印
规则与实测值，读者可以自行复核。规则不满足时输出"未能确定"，不猜。
"""

from __future__ import annotations

__all__: list = []
