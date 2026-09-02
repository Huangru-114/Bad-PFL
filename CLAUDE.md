# CLAUDE.md — 长期约定（每次会话自动加载，保持精简）

这是一个基于**因果不变性的联邦后门防御**研究项目，针对 Bad-PFL（ICLR 2025）。
所有诊断代码在 `diag/` 下。**先读 `diag/HANDOFF.md`** 了解当前研究状态与下一步。
**当前在做的实验看 `diag/PLAN_T0T4.md`**（地板 0.41 的机制：休眠容量 / 共址 / 函数平坦 三选一）。

## 铁律（违反会毁掉结果的可比性或可信度）

1. **原仓库文件零 diff。** `main.py` / `client.py` / `server.py` / `fba.py` /
   `fl_process.py` / `pfl.py` / `utils.py` / `resnet.py` / `generator.py` 等一律
   不改。所有行为差异以旁路方式实现在 `diag/` 里。每次提交前跑
   `git diff --stat main -- '*.py' ':(exclude)diag'`，必须为空。

2. **不为了让结果好看而动任何东西** —— 不调指标口径、不改超参范围、不改评估约定。

3. **不确定就报"未能确定"**，不要硬给结论。

4. **图里绝不出现 CJK**（出版要求）。`diag.analysis.assert_no_cjk_in_figure`
   会在保存前扫描整张图并拦截。

5. **无定义的指标留空**，绝不用 0 或 "N/A" 填充后当数值参与统计。特别是：
   不给"不做客户端级决策的防御"（median / invariant）编造 TPR/FPR。

6. **不安装任何包**（torchvision / pytest 等）。测试用 `diag/tests/run_tests.py`。

7. **提交要勤**：每完成一个自洽的单元就 commit + push 到指定分支。

## 关键背景（一句话版，细节见 HANDOFF / README）

- **FedBN 一直开着**，BN 从不聚合，全局模型 BN 停在初始化值（clean acc ≈ 随机）。
  全局模型评估一律**借 BN**，另存 `acc_global_raw` 暴露退化。见 README §4b。
- **本机只有 CPU、没有 torchvision**。正式实验在用户集群上跑；本机只做冒烟 +
  合成 fixture 验证。GPU 相关的 bug 本机测不出 —— 上集群前先跑
  `python -m diag.tests.run_tests test_defenses` 确认 CUDA 测试 PASS 而非跳过。
- **数据在用户集群上**。要"读文件画图"的程序，不是替用户画好的图。

## 分支与提交

- 开发分支：**`claude/bad-pfl-exp-1-nafj1i`**（2026-09-02 起）。它是
  `claude/bad-pfl-trigger-invariance-ucvpcp` 的**严格超集**（merge-base 就是后者的 HEAD），
  B2 持续性的 config 只在这条上。切错分支会看不到 `exp1.persistence`。
- 提交信息不含任何模型标识符。
- 只在用户明确要求时才建 PR。

## 测试

```bash
python -m diag.tests.run_tests            # 全部（约 25 秒）
python -m diag.tests.run_tests test_xxx   # 单模块
```
新增测试文件必须登记进 `run_tests.py` 的 `TEST_MODULES`
（`_assert_registry_is_complete` 会挡住漏登记）。
