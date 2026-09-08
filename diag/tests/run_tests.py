"""不依赖 pytest 的极简测试运行器。

存在的理由：当前容器没有安装 pytest，而任务要求冒烟测试必须真实跑过。
测试文件本身是标准 pytest 风格，装了 pytest 的环境可以直接
``pytest diag/tests`` —— 这个运行器只是一个后备。

用法::

    python -m diag.tests.run_tests            # 跑全部
    python -m diag.tests.run_tests features   # 只跑名字含 'features' 的模块

跳过约定：用例里 ``raise unittest.SkipTest("原因")``。运行器会**单独计数并
在末尾逐条列出**，绝不与 PASS 混在一起 —— 需要 torch/GPU 的用例在缺依赖的机器上
若被记成 PASS，就是假绿。
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
import unittest
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
    "diag.tests.test_analysis_e_noise",
    "diag.tests.test_exp_f",
    "diag.tests.test_analysis_f",
    "diag.tests.test_invariant_agg",
    "diag.tests.test_instrumentation",
    "diag.tests.test_defenses",
    "diag.tests.test_analysis_ij",
    "diag.tests.test_analysis_density",
    "diag.tests.test_analysis_exposure",
    "diag.tests.test_schedule",
    "diag.tests.test_analysis_exp1",
    "diag.tests.test_run_exp1",
    "diag.tests.test_backbone_alignment",
    "diag.tests.test_wall_clock",
    "diag.tests.test_read_calibration",
    "diag.tests.test_read_probe",
    "diag.tests.test_figstyle",
    "diag.tests.test_metrics_doc",
    "diag.tests.test_viz_trigger",
    "diag.tests.test_cluster_scripts",
    "diag.tests.test_recompute_asr_final",
    "diag.tests.test_represent",
    "diag.tests.test_paramspace",
    "diag.tests.test_exp_t0",
    "diag.tests.test_exp_t3",
    "diag.tests.test_pfl_fedrep",
    "diag.tests.test_paper_asr_filtered",
]


def _assert_registry_is_complete() -> None:
    """目录里有 ``test_*.py`` 却没登记 -> 直接报错。

    这个清单是手写的，漏登记的后果是**测试文件一个 case 都不跑而输出全绿** ——
    比没有测试更糟。所以这里对着磁盘核一遍。
    """
    from pathlib import Path
    on_disk = {f"diag.tests.{p.stem}"
               for p in Path(__file__).parent.glob("test_*.py")}
    missing = sorted(on_disk - set(TEST_MODULES))
    if missing:
        raise SystemExit(
            "以下测试模块在磁盘上但没登记进 TEST_MODULES，"
            f"它们的 case 一个都不会跑：\n  " + "\n  ".join(missing))


def run(selector: str = "") -> int:
    _assert_registry_is_complete()
    passed, failed, skipped = 0, [], []
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
            except unittest.SkipTest as exc:
                # **必须与 PASS 区分开**。曾经的写法是让跳过的用例直接 return，
                # 于是它们和真跑过的一起记 PASS —— 那是假绿，比没有测试更糟
                # （本仓库对「漏登记的测试文件」也是同样的态度，见
                #  _assert_registry_is_complete）。
                skipped.append((short, attr, str(exc)))
                print(f"  SKIP  {short}::{attr}  ({exc})")
            except Exception:
                failed.append((short, attr, traceback.format_exc()))
                print(f"  FAIL  {short}::{attr}")

    elapsed = time.time() - started
    summary = f"{passed} passed, {len(failed)} failed"
    if skipped:
        summary += f", {len(skipped)} skipped"
    print(f"\n{summary} in {elapsed:.2f}s")
    if skipped:
        print("\n跳过的用例（**没有被验证过**，不要当成通过）：")
        for short, attr, why in skipped:
            print(f"  - {short}::{attr}  —— {why}")
    for short, attr, trace in failed:
        print(f"\n{'=' * 70}\nFAILED {short}::{attr}\n{'-' * 70}\n{trace}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else ""))
