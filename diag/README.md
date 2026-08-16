# diag/ — Bad-PFL 触发器跨环境不变性诊断工具

> **交付状态**：代码已实现并通过冒烟测试。**未运行任何正式实验，本目录不含任何
> 科学结论。** 冒烟测试产出的数字（4 客户端 / 2 轮 / 合成数据）没有任何含义。

配套文档：
- [`REPO_MAP.md`](REPO_MAP.md) — Phase 0 仓库侦察报告
- [`PATCHES.md`](PATCHES.md) — 对原仓库的改动记录（**原文件零 diff**）

---

## 1. 新增文件清单

### 基础设施

| 文件 | 说明 | 主要公开接口 | 被谁调用 |
|---|---|---|---|
| `__init__.py` | 把仓库根目录加入 `sys.path`，使 `import fba` / `import resnet` 可用 | `REPO_ROOT` | 所有模块 |
| `config.yaml` | **全部超参的唯一来源** | — | `config.load_config` |
| `config.py` | 配置加载 + 确定性设置 | `load_config`, `Cfg`, `set_all_seeds`, `make_select_rule` | `run_fl`, 所有 exp 脚本 |
| `nanstats.py` | nan-safe 统计（`torch.nanstd` 不存在，自行实现） | `nanmean`, `nanstd`, `nanmedian`, `nanmad`, `nan_cv`, `nan_to_num_count` | `features`, `metrics` |

### 测量层

| 文件 | 说明 | 主要公开接口 | 被谁调用 |
|---|---|---|---|
| `probe.py` | **共享探针集** + 评估 loader 工厂 | `build_probe_set`, `assert_disjoint`, `make_eval_loader`, `get_targets`, `ProbeSet` | `exp_common`, `exp_d` |
| `features.py` | 特征提取与决策几何量 | `extract_penultimate`, `class_prototypes`, `fisher_dispersion`, `margin_matrix`, `knn_overlap`, `find_last_linear` | `metrics`, `exp_c` |
| `perturb.py` | 扰动施加（δ / ξ 的分离与合成） | `apply_perturbation`, `make_xi_fn`, `make_delta_fn`, `delta_from_generator`, `MODES`, `linf` | `metrics`, `exp_common` |
| `metrics.py` | 四个实验的指标计算（纯函数） | `naturalness`, `recovery_score`, `excess_response`, `observable_score`, `dispersion_scores`, `baseline_signals` | 四个 exp 脚本 |

### 训练与埋点

| 文件 | 说明 | 主要公开接口 | 被谁调用 |
|---|---|---|---|
| `hooks.py` | checkpoint 保存 + 对照组校验 + 生成器提取 + 逐位比对 | `save_run`, `attach_save_hook`, `build_client_meta`, `verify_partition_consistency`, `extract_generator`, `flatten_state_dict`, `run_dir_name`, `hash_state`, `state_dict_hash`, `compare_run_checkpoints` | `run_fl`, `exp_c`, `exp_f`, `run_matrix` |
| `snapshots.py` | **按轮次的快照埋点**（实验 F） | `SnapshotRecorder`, `build_grid`, `select_snapshot_clients`, `available_rounds`, `load_manifest` | `run_fl`, `exp_f` |
| `run_fl.py` | **诊断用 FL 训练驱动**（clean / attack） | `run_fl`, `build_datasets`, `SyntheticImageDataset` | CLI, `exp_common`, 冒烟测试 |

### 实验驱动与分析

| 文件 | 说明 | 主要公开接口 | 被谁调用 |
|---|---|---|---|
| `exp_common.py` | 四个实验共用的加载与装配 | `load_bundle`, `RunBundle`, `common_row`, `write_csv`, `COMMON_COLUMNS` | 四个 exp 脚本 |
| `exp_a.py` | 实验 A：δ 的跨客户端自然度离散（⭐ 生死判据） | `run_exp_a` | CLI |
| `exp_b.py` | 实验 B：Table 24 的跨客户端版本 | `run_exp_b` | CLI |
| `exp_c.py` | 实验 C：超额响应与可观测量 | `run_exp_c` | CLI |
| `exp_d.py` | 实验 D：源类表征混乱 | `run_exp_d` | CLI |
| `analysis.py` | 汇总与绘图 | `summarize`, `summarize_c`, `summarize_d`, `plot_a1..a4`, `plot_c1..c3`, `plot_d1..d2`, `auc_score`, `pearson_spearman`, `mannwhitney` | CLI, `run_matrix` |
| `run_matrix.py` | 24 组扫描的命令生成器（**默认 dry-run**） | `build_commands` | CLI |
| `exp_e_precheck.py` | 实验 E 的 Phase 0 校验（干净分支 / 划分一致性 / E1-E2 就绪 / 参数哈希） | `check_clean_branch`, `check_partition_consistency` | CLI |
| `exp_e.py` | 实验 E：对抗地板测定，四个 cell（E1-E4） | `run_cell`, `evaluate_mode`, `build_exp_e_probe`, `train_oracle_generator` | CLI |
| `analysis_e.py` | 实验 E 的汇总与三张图 | `summarize_e`, `decomposition_e`, `classify_floor`, `plot_e1..e3` | CLI |
| `analysis_e_noise.py` | E1 上"随机噪声响应 × 异构度"的良性/恶意分组图（集群侧独立脚本） | `summarize_noise`, `baseline_by_group`, `plot_noise_response` | CLI |
| `exp_f_precheck.py` | 实验 F 的 Phase 0（快照盘点 / **FedBN 全局模型 BN 诊断** / 三方 clean acc / 重跑成本） | `inventory_snapshots`, `diagnose_global_bn`, `measure_clean_acc`, `estimate_rerun_cost` | CLI |
| `exp_f.py` | 实验 F：冻结生成器的时间衰减矩阵（上三角 s × t） | `run_matrix`, `FrozenGenerator`, `compute_delta`, `make_frozen_xi_fn`, `load_snapshot_model` | CLI |
| `analysis_f.py` | 实验 F 的汇总与三张图 | `summarize_f`, `matrix_pivot`, `decay_table`, `classify_decay`, `plot_f1..f3` | CLI |

### 测试

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_features.py` | 形状、eval 模式强制、nan 语义、kNN 的重合/分离极端行为 |
| `tests/test_perturb.py` | 各模式形状与像素范围、L∞ 预算、ξ 可复现性、梯度无副作用、缺组件抛异常 |
| `tests/test_metrics.py` | `observable_score` 在构造矩阵上给高分/随机矩阵给低分、**信息隔离签名检查** |
| `tests/test_nan_safety.py` | 全部统计函数在含 nan 输入下的行为；退化情形返回 nan 而非 inf |
| `tests/test_probe.py` | 四子集不相交、规模、确定性、标签纯度、不扰动全局 RNG |
| `tests/test_eval_loader.py` | **`drop_last` 回归测试** |
| `tests/test_hooks.py` | `verify_partition_consistency` 逐字段的漏报测试、深拷贝、生成器提取 |
| `tests/test_analysis.py` | 汇总数学 + 全部绘图函数在合成数据上跑通 |
| `tests/test_audit.py` | P0 排查工具：泄漏 monkey-patch、数据集重建守卫、Markdown 转义 |
| `tests/test_exp_e.py` | 实验 E：双指标的分子分母、real_target 空分母、E3 不污染 clean 模型 |
| `tests/test_analysis_e.py` | 实验 E 的汇总与绘图 + **「图中禁止中文」的强制检查** |
| `tests/run_tests.py` | 不依赖 pytest 的运行器 |
| `tests/smoke_integration.py` | 端到端集成冒烟测试 |

---

## 2. 对原仓库的改动

**原文件零 diff**（`git diff --stat main -- . ':(exclude)diag'` 为空）。

所有行为差异以旁路方式实现在 `diag/` 中。逐条说明见
[`PATCHES.md`](PATCHES.md)，这里只汇总"是否影响训练动力学"：

| 编号 | 内容 | 影响训练动力学？ |
|---|---|---|
| P1 | 补齐 numpy / random 播种 | ⚠️ **是** —— 改变实际划分出的分片内容（分布不变）。**不修就没有对照实验**，已获确认 |
| P2 | 客户端采样改用独立 RNG | ⚠️ **是** —— 采样序列与原实现不同（分布不变）。**不修则 clean/attack 选中的客户端不一致**，已获确认 |
| P3 | 评估 loader 改 `drop_last=False` | 否，仅评估路径 |
| P4 | 生成 ξ 时用 `eval()` | 否，仅离线诊断路径 |
| P5 | 固定 ξ 的随机起点 | 否，RNG 状态保存/恢复 |
| P6 | `pgd_attack` 后清零梯度 | 否，仅离线诊断路径 |
| 埋点 1–4 | checkpoint / 计数 / 生成器提取 / 自定义属性 | 否，纯只读埋点 |

**P1 与 P2 是仅有的两处会改变训练动力学的差异**，两者都是构造对照实验的前提，
且都已经过你的确认。其余全部是只读埋点或仅作用于离线路径。

---

## 3. 使用方法

### 3.1 环境要求

已具备（当前容器）：`torch 2.13.0`、`numpy 2.4.6`、`pandas 3.0.5`、
`scipy 1.17.1`、`matplotlib 3.11.1`、`pyyaml 6.0.1`、Python 3.11.15。

**正式实验还需要**：
- `torchvision` —— 加载 CIFAR-10（当前**未安装**）
- GPU —— 完整配置约 45,000 次 batch-32 前后向，4 核 CPU 上不现实

### 3.2 跑一次 run 并保存 checkpoint

```bash
# 干净 run
python -m diag.run_fl --mode clean  --alpha 0.5 --seed 0 --ckpt-root ./checkpoints
# 攻击 run（同 seed、同 alpha，唯一变量是是否投毒）
python -m diag.run_fl --mode attack --alpha 0.5 --seed 0 --ckpt-root ./checkpoints
```

产物：

```
checkpoints/{mode}_a{alpha}_s{seed}/
    client_0.pt ... client_99.pt    # 每个客户端的本地模型
    global.pt                       # 全局模型
    generator.pt                    # 生成器（仅 attack run）
    delta.pt                        # δ 存档（仅 attack run，不参与计算，见 §5）
    meta.json                       # 标签分布、是否恶意、投毒比例、划分索引、种子
```

### 3.3 校验对照组（**每次都要做**）

```bash
python -c "
from diag.hooks import verify_partition_consistency
ok, rep = verify_partition_consistency(
    'checkpoints/clean_a0.5_s0/meta.json',
    'checkpoints/attack_a0.5_s0/meta.json')
print(rep); raise SystemExit(0 if ok else 1)"
```

它逐客户端比对 `partition_idx` / 样本数 / 标签直方图 / **训练与测试的绝对索引**。
返回 False 就不要继续 —— 划分错位会让所有跨模式比较**静默**失效。

### 3.4 跑各个实验

```bash
# 实验 A：模型用 clean run，δ 用 attack run 的生成器
python -m diag.exp_a --ckpt-dir checkpoints/clean_a0.5_s0 \
                     --gen-ckpt-dir checkpoints/attack_a0.5_s0 \
                     --out results/raw/expA_a0.5_s0.csv

# 实验 B：同上
python -m diag.exp_b --ckpt-dir checkpoints/clean_a0.5_s0 \
                     --gen-ckpt-dir checkpoints/attack_a0.5_s0 \
                     --out results/raw/expB_a0.5_s0.csv

# 实验 C：需要 clean + attack 两个目录
python -m diag.exp_c --ckpt-dir checkpoints/attack_a0.5_s0 \
                     --clean-ckpt-dir checkpoints/clean_a0.5_s0 \
                     --out results/raw/expC_a0.5_s0.csv

# 实验 D：用 attack run
python -m diag.exp_d --ckpt-dir checkpoints/attack_a0.5_s0 \
                     --out results/raw/expD_a0.5_s0.csv
```

#### E1 的随机噪声响应 × 异构度（分良性/恶意）

`random_eps4` / `random_eps8` 本来只是负对照，但它们确实把 ASR 抬高了
（α=0.5 合并口径：`none` 0.0270 → `eps4` 0.0335 → `eps8` 0.0434）。
这个脚本把抬升按**良性/恶意**拆开，看它是否随 α 变化。

`results/exp_e_summary.csv` **没有 `is_malicious` 这一维**，所以必须读逐客户端的
原始 CSV（`results/raw/` 被 gitignore，只在集群上）：

```bash
python -m diag.analysis_e_noise \
    --raw-glob "results/raw/exp_e_E1_*.csv" \
    --out-dir results/figs \
    --summary-out results/exp_e_noise_response.csv
```

出一张上下两 panel 的图：上=绝对 ASR（含各组 `none` 基线虚线），
下=`ASR(random) − ASR(none)`，其中 lift 是**同一客户端逐个相减后再聚合**，
不是两个组均值相减——这样误差棒反映的是配对差异的离散度。
横轴只有一个 α 时自动退化为纯标记并打印提醒，不画会被误读成趋势的连线。

#### 实验 F：冻结生成器的时间衰减矩阵

**前提**：需要按轮次的快照。attack run 必须带 `--snapshot-every` 跑过，
否则模型侧一个中间轮次都没有（`save_run` 只在训练结束调用一次）。

```bash
# P0：核查快照可用性 + FedBN 全局模型的 BN 诊断
python -m diag.exp_f_precheck --ckpt-root ./checkpoints --out F0_PRECHECK.md

# P1：带快照埋点重跑 attack run（--verify-against 证明埋点没改变训练动力学）
python -m diag.run_fl --mode attack --alpha 0.5 --seed 0 \
       --snapshot-every 50 \
       --verify-against ./checkpoints/attack_a0.5_s0

# P2：稀疏矩阵（5 个点 = 15 格上三角，仅 delta_only、仅个性化模型）
python -m diag.exp_f --ckpt-root ./checkpoints --alpha 0.5 --seed 0 \
       --sparse --modes delta_only --eval-targets personalized

# P3：完整矩阵 + 其余扰动模式 + 借-BN 全局模型
python -m diag.exp_f --ckpt-root ./checkpoints --alpha 0.5 --seed 0 \
       --modes delta_only full_xi_frozen full_xi_fresh

# P4：三张图 + 判据
python -m diag.analysis_f --raw-glob "results/raw/exp_f_matrix_*.csv"
```

### 3.5 汇总与绘图

```bash
python -c "
from diag.analysis import *
sa = summarize('results/raw/expA_*.csv')
sa.to_csv('results/summary/expA_summary.csv', index=False)
plot_a1(sa, 'results/figs/expA_A1.png')
plot_a2(sa, 'results/figs/expA_A2.png')
plot_a3('results/raw/expA_*.csv', 'results/figs/expA_A3.png')
plot_a4('results/raw/expA_*.csv', 'results/figs/expA_A4.png')
"
```

### 3.6 扫描矩阵（**默认 dry-run**）

```bash
python -m diag.run_matrix                # 打印 85 条命令，不执行
python -m diag.run_matrix --stage train  # 只看 24 条训练命令
# python -m diag.run_matrix --execute    # 真正执行（本次任务不要用）
```

### 3.6b 图的语言约定

**所有图中一律使用英文。** `diag/analysis.py::_finish` 在保存前会扫描整张图的
文字（含 figure 级的 `supxlabel` / `suptitle`），发现中日韩字符直接抛
`ValueError`。这是刻意选择：配一个 CJK 字体只会让违规"看起来正常"，
而 matplotlib 默认的 DejaVu Sans 不含 CJK 字形，中文会静默渲染成方框。

### 3.7 跑测试

```bash
python -m diag.tests.run_tests           # 94 个单元测试，约 5 秒
python -m diag.tests.smoke_integration   # 端到端集成冒烟，约 1 分钟
# 装了 pytest 的环境可直接：pytest diag/tests
```

---

## 4. 配置项说明（`config.yaml`）

| 字段 | 含义 | 取值范围 | 影响哪些实验 |
|---|---|---|---|
| `paths.ckpt_root` / `results_root` | checkpoint 与结果的根目录 | 路径 | 全部 |
| `data.target_class` | 目标类，默认 0 (airplane) | 0–9 | 全部 |
| `fl.*` | FL 训练超参，逐项对齐 `main.py` 的 argparse 默认值 | 见文件 | 训练 |
| `fl.poison_rate` | 投毒比例，默认 0.2 | 0–1 | D（稀释效应），写进全部 CSV |
| `sweep.alpha` | Dirichlet 异构度扫描 | `[1.0, 0.5, 0.1, 0.05]` | 全部 |
| `sweep.seed` / `sweep.mode` | 种子与模式 | `[0,1,2]` / `[clean, attack]` | 全部 |
| `smoke.*` | 冒烟测试的极小配置 | 见文件 | 仅冒烟 |
| **`probe.seed`** | 探针集抽样种子，**与实验 seed 解耦** | 固定 12345 | A/B/C/D |
| `probe.n_ref` | 参照分布规模（真实目标类） | 建议 ≥ 10 × `knn_k` | A |
| `probe.n_other_per_class` | 每个非目标类的对照样本数 | 200 | A/D |
| `probe.n_query` | 施加扰动的载体数量 | 1000 | A/C |
| `probe.n_target` | 实验 B 用的目标类样本（与 ref 不相交） | 400 | A(real 组)/B |
| `metrics.knn_k` | kNN 邻居数 | 20 | A |
| `metrics.lambda_std` | `observable_score` 中 std 项权重 | 1.0 | C |
| `metrics.lambda_sweep` | λ 敏感性分析的取值 | `[0,0.5,1,2]` | C（图 C3） |
| `metrics.min_class_count` | **本地**类原型的有效性阈值 | 5 | C |
| `perturb.eps_delta` / `eps_xi` | L∞ 预算，均为 4/255 | 对齐 `fba.py` | A/B/C |
| `perturb.xi_seed` / `noise_seed` | 固定 ξ 与随机噪声 | 任意整数 | A/B/C |
| `exp_e.*` | 实验 E 的探针规模、随机重复次数、E3 现训生成器协议 | 见文件 | E |
| `exp_f.snapshot_every` | 按轮次快照的网格间隔 | 50（1000 轮 → 20 点） | F |
| `exp_f.snapshot_n_benign` / `n_malicious` | 逐轮保存本地模型的客户端数 | 10 / 2 | F |
| `exp_f.grid_sparse` | P2 的稀疏网格 | `[200,…,1000]` | F |
| `exp_f.modes_p2` / `modes_full` / `modes_per_t` | 矩阵模式与只随 t 变的模式 | 见文件 | F |
| `exp_f.eval_targets` | `personalized`（主）/ `global_bn_borrowed`（副） | 见下方 §4b | F |
| `exp_f.mature_s_threshold` | §3.3 判据只看 `s >=` 该值 | 500 | F |
| `determinism.select_rule_seed_offset` | 客户端采样 RNG 的种子偏移 | 10000 | 训练 |

**已删除的字段**：`metrics.min_target_samples`。改用共享探针集后所有客户端都能
算出有效指标，本地目标类样本数改为 CSV 中的协变量 `n_target_samples_local`。

### 确定性

`set_all_seeds(seed)` 播种 `random` / `numpy` / `torch`（含 cuda），并设置
`torch.backends.cudnn.deterministic = True`、`benchmark = False`。
实际使用的种子写进 `meta.json` 的 `seeds` 字段，连同 `torch_version` /
`numpy_version` 一并存档。

探针集使用**独立**的 `np.random.RandomState(probe.seed)`，不触碰全局 numpy RNG
（有测试保证），因此构造探针集不会扰动数据划分。

---

## 4b. ⚠️ FedBN 下全局模型的 BN 永不更新

这条是在做实验 F 的 Phase 0 时从代码里查出来的，**会影响已有结果的解读**。

`pfl.py:5-12` 的 `fedbn_update` 挂在 `before_update_global` 阶段，把所有含
`bn` 或 `shortcut.1` 的键从 `server.update` 里 `pop` 掉；随后 `server.py:34`
的 `load_state_dict(self.update, strict=False)` 就再也碰不到 BN。
ResNet 的 BN 键全部匹配这两个模式（`resnet.py:22,25,32`）。

**结果：`global.pt` 的 BN 权重与 running stats 从初始化起 1000 轮一位没变**
（weight=1, bias=0, running_mean=0, running_var=1, num_batches_tracked=0）。

这**不是 bug**，是 FedBN 的固有语义 —— 个性化模型 = 共享参数 + 各自私有 BN，
全局模型本来就不是一个可以单独拿来用的东西。但它有两个直接后果：

1. **实验 F 的主评估对象不能是 `global.pt`。** 改为：
   - 主：`personalized` —— 抽样良性客户端的本地模型，与论文的 ASR 定义、
     与实验 E 的协议一致；
   - 副：`global_bn_borrowed` —— 全局共享参数 + 固定良性客户端的 BN。
     BN 供体的选择本身是一个自由度，CSV 里如实记录。
2. **实验 E 的 E4 需要重测。** E4 用 `attack_dir/global.pt` 作黑盒 ξ 模型
   （`exp_e.py:406`），得到 ASR=0.0312。这个数字里有多少来自"黑盒迁移困难"、
   多少来自"模型本身失配"，取决于该全局模型的 clean accuracy ——
   `exp_f_precheck.measure_clean_acc` 会把这个数字直接测出来。
   **在拿到那个数字之前，E4 的结论不作数。**

验证方式（不依赖任何推断）：`exp_f_precheck.diagnose_global_bn` 直接读盘断言
全部 BN 条目等于初始化值，并逐条打印实测值。

---

## 5. 已知限制与未验证项 ⭐

**这一节是本文档最重要的部分，请认真读。**

### 5.1 完全没有在真实模型/真实数据上验证过

| 项 | 状态 |
|---|---|
| 全部指标函数 | 只在**合成数据 + 4 客户端 / 2 轮的玩具模型**上跑通过 |
| CIFAR-10 数据通路 | **从未运行**（`torchvision` 未安装） |
| 默认配置的 ACC / ASR 复现 | **从未运行**，无法与论文值对比 |
| Table 24 的 6.70% / 39.10% 复现 | **从未运行**。`exp_b` 已实现该测量并会输出全局模型那一行，但数字对不对完全未知 |
| 完整规模（100 客户端 / 300 轮）下的内存与耗时 | **未测**。粗估 4 核 CPU 上单次 run 数小时，24 组扫描不可行，需要 GPU |

**因此：所有函数的"数值正确性"只有形状级和极端情形级的保证，没有真实数据级的保证。**

### 5.2 基于推测、可能与实际不符的设计

1. **`baseline_signals` 的 `sign_consistency` 定义是我的解释性选择。**
   spec 给的"`|mean_i sign(g_i)|` 逐维求和"是一个**全局**量，无法区分客户端。
   我推广成逐客户端的 `mean_d [sign(u_k[d]) · mean_i sign(u_i[d])]`。
   这**不是** Wang et al. (AISTATS'24) 原文的定义，我也没有查证原文。
   **图 C2 的第三条曲线依赖这个定义，正式使用前需要你确认。**

2. **客户端更新向量取自"训练结束时的本地模型 − 全局模型"。**
   但训练结束时，最后一轮没被选中的客户端，其 `local_model` 停留在它上一次参与
   时的状态（300 轮 × 10/100 下平均参与 30 次，多数客户端会参与，但不同客户端的
   "新鲜度"不同）。真实的防御方看到的是**当轮**上传的更新。
   这是一个已知的口径差异，`meta.json` 的 `n_participations` 可用于事后诊断。

3. **实验 D 的"被投毒源类"分组当前是空的。**
   Bad-PFL 对**全部**非目标类样本施加 ξ，不存在特定源类，所以 `plot_d1` 默认把
   所有非目标类归为"其余类"。若你认为应当区分，需要显式传
   `poisoned_source_classes`。

4. **`exp_d` 的散度在 `probe.ref ∪ probe.other` 上计算。** 这个子集是类别均衡的
   （目标类 500 + 每个非目标类 200），不是均匀的。类别不均衡是否会系统性影响
   `fisher_dispersion` 的跨类比较，**未做敏感性分析**。

### 5.3 δ / ξ 分离的实现方式，以及我对它的信心程度

**分离方式**（对应 `fba.py:53-55`）：

```python
ξ:  fba.pgd_attack(model, x, y) - x           # 无目标 PGD，L∞ = 4/255
δ:  generator(x) * (4/255)                    # ≡ 原实现的 `/255.*4.`
合成: x + ξ + δ                                # L∞ ≤ 8/255
```

**信心程度：对 δ 很高，对 ξ 中等。**

- **δ（高信心）**：`generator(x) / 255. * 4.` 与 `generator(x) * (4/255)` 是
  逐字的数值等价，没有任何解释空间。生成器以 Tanh 收尾，`|δ| ≤ 4/255` 有结构保证
  （有测试）。

- **ξ（中等信心）**：我**直接 import 并调用原 `fba.pgd_attack`**，没有重写，
  所以数学上不会漂移。但有三处我做了决定：
  1. **模式**：诊断时用 `eval()`，原投毒路径是 `train()`（P4）。BN 行为不同，
     算出的 ξ 与训练时刻的 ξ **不完全相同**。我认为 eval 是对的（否则特征被污染），
     但这确实是一处偏离。
  2. **绑定哪个模型**：按你的指示"按代码库原有逻辑"，`xi_fn` 绑定**被评估的那个
     模型**（`exp_a`/`exp_b` 用各客户端自己的模型，`exp_c` 分别用 clean/attacked
     模型）。**这带来一个已知的归因问题**：ξ 本身就跨客户端不同，所以实验 A 测到的
     离散度**无法干净地归因给 δ**。代码保留了把 `xi_fn` 绑到固定参考模型的能力
     （`RunBundle.xi_fn(any_model)`），但四个实验脚本目前都走"自己的模型"这条路。
     **这是 §6 清单里最重要的一条。**
  3. **随机性**：原实现的随机起点不固定，我固定了（P5）。

### 5.4 真实数据上可能出问题的地方

| 风险 | 说明 |
|---|---|
| **`knn_overlap` 的内存** | `torch.cdist(query[1000], bank[2300])` 在 512 维上约 9 MB，安全。但若增大探针集或换更大的特征维度需重新估算。已做 chunk（默认 256）。 |
| **`margin_matrix` 的双层 Python 循环** | 10 类下是 90 次迭代，可忽略。类别数很大时（如 CIFAR-100）会变慢，需要向量化。 |
| **`meta.json` 体积** | 现在包含每客户端的训练/测试绝对索引。100 客户端 × 500 + 100 × 100 ≈ 60,000 个整数，约 400 KB/run，24 个 run 约 10 MB。可接受但不算小。 |
| **checkpoint 体积** | resnet10 约 5 MB，100 客户端 × 24 run ≈ **12 GB**。**磁盘需要提前规划。** |
| **`fisher_dispersion` 的分母** | 若某类原型恰好落在全局原型上，分母趋 0。已保护为返回 nan，但真实数据上多久触发一次未知。 |
| **`observable_score` 的 MAD** | 客户端数少或边距高度一致时 MAD 可能为 0，此时该 `(c,t)` 位置记 nan。极端情况下整个客户端全 nan（有测试覆盖），此时 `A^(k)` 为 nan 而不是 0。**下游 AUC 计算会剔除 nan**，可能导致某些 α 下的有效样本数变少。 |
| **AMP** | 原仓库无条件使用 `torch.cuda.amp`。CPU 上已确认降级为 no-op（`GradScaler.is_enabled() == False`）。GPU 上的数值行为**未测**。 |
| **`client_inner_dirichlet_partition` 在极小 α 下的行为** | 算法中"类别耗尽则随机改派"（`utils.py:72-77`）可能让实际异构度**低于**名义 α。**这直接影响实验 A 的 x 轴含义。** 建议正式实验前先用 `utils.partition_report` 做一次划分统计的 sanity check。 |

### 5.5 关于 t-SNE

**本工具包不实现 t-SNE。** 所有定量结论来自 `knn_overlap` 等可量化指标。
若将来为论文配图需要 t-SNE，必须在图注中标注"该图不参与任何指标计算"。

---

### 5.6 实验 F 特有的限制

| 项 | 状态 |
|---|---|
| 单 (α=0.5, seed=0) 单点 | 无法区分真实效应与单次抽样偶然 |
| 客户端快照的 staleness | FedBN 下客户端只在参与的轮次更新本地模型，网格点的快照取自其后的**首次参与**，典型滞后 0–10 轮。已作为 `staleness` 列如实记录，但它与 `gap` 的效应**无法完全解耦** |
| 借-BN 全局模型的供体选择 | 任选一个良性客户端，这本身是一个自由度 |
| P2 稀疏网格取不到 `gap=500` | 步长 200 的网格只有 200 的倍数，§3.3 的 H_stable 侧判据在 P2 上必然返回"未能确定"。这是网格的真实限制，**绝不用插值补**（陷阱 8），有 `test_sparse_grid_cannot_reach_the_stable_gap` 固定住这个行为 |
| 重跑与旧 run 的一致性 | 已在冒烟规模上实证：开/关快照埋点的两次 run **逐位一致**。GPU + 1000 轮的真实规模上仍需用 `--verify-against` 复核；若不一致，实验 E 与 F 的结果**不可混用**，须在新 run 上重算 E |

---

## 6. 下一步需要人工确认的事项

按重要性排序：

0. **⭐ 实验 E 的 E4 需要重测**（§4b）。它用的 `global.pt` 在 FedBN 下 BN 从未
   更新过。先跑 `exp_f_precheck` 拿到该模型的 clean accuracy，再决定 E4 的
   0.0312 有多少是"黑盒迁移困难"、多少是"模型失配"。

1. **⭐ ξ 的绑定模型是否就此定案。**
   当前按"原有逻辑"绑定被评估的模型，代价是实验 A 的离散度无法干净归因给 δ
   （ξ 本身就在变）。如果实验 A 是生死判据，建议**至少补一组** ξ 固定用全局模型的
   对照。代码已支持，只需在 `exp_a.py` 里改一行绑定。**这个决定影响整个实验 A 的
   可解释性。**

2. **`sign_consistency` 的定义**（§5.2.1）。这是图 C2 的第三条曲线，我的推广方式
   需要你对照 Wang et al. 原文确认。

3. **是否安装 `torchvision`**，以及**是否有 GPU 资源**。没有这两样就无法进入正式实验。
   同时请确认 CIFAR-10 的下载源在出口代理白名单内 —— 这一点**未验证**。

4. **磁盘规划**：24 个 run × 100 客户端 ≈ 12 GB checkpoint。是否接受？
   若不接受，可以改为只保存被选中过的客户端，或只保存最后一轮参与过的客户端。

5. **`exp_d` 是否需要区分"被投毒源类"**（§5.2.3）。当前实现认为不存在特定源类。

6. **正式实验前先做划分 sanity check**（§5.4 最后一行）：确认小 α 下实际的
   标签分布确实足够异构，否则实验 A 的 x 轴不成立。

7. **`run_fl.py` 与 `main.py` 的漂移风险**（`PATCHES.md` §3）。如果你更希望直接改
   `main.py` 而不是复刻，我可以改成那种形式 —— 代价是原文件不再零 diff。

---

## 7. 复现说明

```bash
# 环境
Python 3.11.15 / torch 2.13.0+cu130 (CPU 模式) / numpy 2.4.6
pandas 3.0.5 / scipy 1.17.1 / matplotlib 3.11.1 / pyyaml 6.0.1
# torchvision: 未安装（正式实验必需）
# pytest: 未安装（用 diag/tests/run_tests.py 代替）

# 单元测试（190 个，约 70 秒）
python -m diag.tests.run_tests
python -m diag.tests.run_tests exp_f      # 只跑某个模块

# 集成冒烟测试（约 1 分钟，全 CPU）
python -m diag.tests.smoke_integration --workdir /tmp/diag_smoke --keep

# 确认原仓库文件零改动
git diff --stat main -- . ':(exclude)diag'
```

实验 F 的快照埋点另有一项端到端验证：同一 seed 分别跑
`--snapshot-every 1` 与 `--snapshot-every 0`，用 `--verify-against` 比对，
应当**逐位一致**（已在冒烟规模上通过，6/6 个 checkpoint 哈希相同）。
这是"埋点只做 I/O、不改变训练动力学"这条主张的实证，不是口头保证。

冒烟测试使用 `config.yaml` 的 `smoke` 段：4 客户端 / 1 个恶意 / 2 轮 /
每客户端 64 样本 / batch 16 / CPU / seed 0 / alpha 0.5。
探针集使用 `smoke.probe` 段的极小规模，抽样种子固定 12345。

**再次强调：冒烟测试的所有数值没有科学含义，只证明代码可运行。**
