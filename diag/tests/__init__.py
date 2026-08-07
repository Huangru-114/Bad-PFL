"""诊断工具的单元与集成冒烟测试。

全部设计为可在 CPU 上、无 GPU、无 torchvision 的环境下运行。
文件是标准 pytest 风格（模块级 ``test_*`` 函数 + 裸 assert）。
若环境没有 pytest，可用 ``python -m diag.tests.run_tests`` 直接跑。
"""
