"""不依赖 pytest 的极简测试运行器。

存在的理由：当前容器没有安装 pytest，而任务要求冒烟测试必须真实跑过。
测试文件本身是标准 pytest 风格，装了 pytest 的环境可以直接
``pytest diag/tests`` —— 这个运行器只是一个后备。

用法::

    python -m diag.tests.run_tests            # 跑全部
    python -m diag.tests.run_tests features   # 只跑名字含 'features' 的模块
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import List

TEST_MODULES = [
    "diag.tests.test_nan_safety",
    "diag.tests.test_features",
    "diag.tests.test_perturb",
    "diag.tests.test_metrics",
    "diag.tests.test_probe",
    "diag.tests.test_eval_loader",
    "diag.tests.test_hooks",
    "diag.tests.test_analysis",
    "diag.tests.test_audit",
    "diag.tests.test_exp_e",
    "diag.tests.test_analysis_e",
    "diag.tests.test_exp_f",
    "diag.tests.test_analysis_f",
]


def run(selector: str = "") -> int:
    passed, failed = 0, []
    started = time.time()

    for module_name in TEST_MODULES:
        if selector and selector not in module_name:
            continue
        module = importlib.import_module(module_name)
        short = module_name.rsplit(".", 1)[1]
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            try:
                func()
                passed += 1
                print(f"  PASS  {short}::{attr}")
            except Exception:
                failed.append((short, attr, traceback.format_exc()))
                print(f"  FAIL  {short}::{attr}")

    elapsed = time.time() - started
    print(f"\n{passed} passed, {len(failed)} failed in {elapsed:.2f}s")
    for short, attr, trace in failed:
        print(f"\n{'=' * 70}\nFAILED {short}::{attr}\n{'-' * 70}\n{trace}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else ""))
