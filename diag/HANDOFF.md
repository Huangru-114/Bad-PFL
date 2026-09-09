# HANDOFF.md — 研究状态与下一步（新会话开场先读这个）

> 代码/设计细节在 `README.md`（§1 文件清单、§3 用法、§5 已知限制）与
> `PATCHES.md`（旁路实现的每一处差异）。本文件只记**研究状态**：真实数据得到
> 了什么、还没定的决策、下一步做什么。约束见根目录 `CLAUDE.md`。

用户是 KTH 的研究者（zgao@kth.se），在做**基于因果不变性的联邦后门防御**，
对标 Bad-PFL（ICLR 2025）。

---

## 0'. 新增：与 tf-dpfl 对齐的能力（2026-09-07）

导师意见触发的一轮口径/框架对齐。**两件都不改既有列与既有行为**，
但它们改变了「下一批 exp1 该怎么跑」。

### FedRep 臂（`diag/pfl_fedrep.py`，CLI `--pfl fedrep`）

上游 `pfl.py` 只有 FedBN；而 tf-dpfl 的 Experiment 3 跑的是 `hier_fedrep`。
两者的「个性化」几乎是**互补**的：

| | 私有（不聚合） | 共享 |
|---|---|---|
| FedBN（上游） | BN 的 γ/β + running stats | 分类头 |
| FedRep（新增） | 分类头 `linear.*` | BN（含 running stats） |

→ **两组 ASR 在语义上不可直接比大小**，只能比趋势形状。
（报告 §2 把 Exp 1/1B 的框架写成了 FedRep，实际跑的是 FedBN。）

用法：`python -m diag.run_exp1 --pfl fedrep ...`。
臂名会进 run tag（`e1` → `e1_fedrep`），两条臂的 CSV 互不覆盖；
fedbn 保持原命名，旧 run 仍可 `--skip-existing` 续跑。

**一个必须知道的后果**：FedRep 下 BN 参与聚合 → `server.global_model` 的 BN
**会**更新，全局模型是个可用模型，与 FedBN 臂完全相反（README §4b 说的
「BN 从初始化起一位不变」只对 fedbn 臂成立）。所以：
- 两条臂的 `acc_global` **不可同框**；
- `track.py` 的「借 BN」机制（`_borrowed_bn_state`）在 FedRep 臂上是多余的。

### 逐轮 filtered ASR（`asr_paper_filtered_*`）

`asr_paper_*` 一直是 **unfiltered**（分母含目标类，复刻 `main.py:131`），
而 tf-dpfl 是 **filtered**。CIFAR-10 下差约 10 个百分点。
现在两个口径在**同一次前向**里一起算出来（零额外机时），新增三列。
`analysis_exp1 --asr-column asr_paper_filtered_benign` 即可用。
**跨库比较或对外报数一律用 filtered。**
旧 run 没有这列，`resolve_asr_column` 会回退并打印提示。

### ⚠️ 开跑前必做

本轮的测试有一部分**在开发容器里没有 torch/numpy，我没有执行过**：
`test_pfl_fedrep` 的 5 条（逐位冻结不变量等）、`test_paper_asr_filtered` 的 4 条
（手算对拍等）、`test_run_exp1` 的 4 条（两臂 tag 不相交等）。
上机后先跑

```bash
python -m diag.tests.run_tests test_pfl_fedrep
python -m diag.tests.run_tests test_paper_asr_filtered
python -m diag.tests.run_tests test_run_exp1
```

确认它们是 **PASS 而不是 SKIP**，再烧机时。
（运行器本轮加了真 skip 语义——此前跳过的用例会被记成 PASS，是假绿。）

---

## 0. 当前主线（2026-09-02 起）：`diag/PLAN_T0T4.md`

B2 持续性曲线跑出来了。**上游任务书「200 轮不衰减」的前提不成立** —— 实际是
前 ~30 轮掉掉 0.23、之后进 **0.4133 的地板**并停住（尾段斜率 t=−0.86，与 0 不可区分）。

现在要回答的是「**这个地板是什么机制**」，三个互斥解释里选一个：
**(a) 休眠容量 / (b) 对齐共址 / (c) 函数平坦**。结论直接决定防御选型
（占位类 vs 特征层 vs 蒸馏平滑），见 `PLAN_T0T4.md §5`。

计划、已核实的事实、数据位置、以及**八条已知的坑**都在 `PLAN_T0T4.md`。

**T0 已跑完**（2026-09-04，`results/t0/`，结果全文在 `PLAN_T0T4.md §9`）：

| 读数 | 值 | 含义 |
|---|---|---|
| 干净 200 轮 / 植入段 的相对位移比 | **1.33**（同时长 **0.876**） | 干净训练改写得不比植入少，而 ASR 停在 0.41 |
| 逐层比值 min / median / max | 0.976 / 1.265 / 1.698（21 层） | **没有一层是静止的** → 层粒度上没有休眠区 |
| cos(θ, Δθ) | 0.02–0.08 | 是方向改写，不是整体缩放 |
| 实测‖Δθ‖ / 各段正交时 | **1.063** | 干净阶段是近正交增量的**随机游走**，不是定向漂移 |
| bn_affine / bn_buffer 位移 | **恒为 0**，‖θ_bn‖ ≡ √4800 | FedBN「全局 BN 停在初始化值」的逐坐标实证 |

→ **(a) 休眠容量没被杀死，但只剩"坐标粒度的小子集"这一种活法**；
**(c) 函数平坦拿到正面证据**。判 (a) 生死仍需 S1 载体分离（要 `L_bd` 梯度，集群）。

**T3 也跑完了（2026-09-05，`results/t3/`，结果全文在 `PLAN_T0T4.md §11`）**：

| 等 ACC 代价 0.05 下 | ASR | 残留 |
|---|---|---|
| 基线（`zero` = 200 轮良性漂移的终点） | 0.3637 | — |
| **real（沿良性方向再外推）** | **0.0540** | **15%** |
| shuffled | 0.2462 | 68% |
| sign_flipped / gaussian | 0.340 / 0.342 | 94% |

→ **残留后门由方向决定，不由位移幅度决定**（36/36 客户端同向，配对 t≈19.5）。
⚠️ 两条不能读出来的：`real` 是**外推**不是复现，所以**不**等于"良性训练能洗掉后门"
（`zero` 格自己就是 200 轮良性训练之后的 0.3637）；Δ 需要预知未来漂移，
**不是可部署防御**，属于机制干预。下一步见 `PLAN_T0T4.md §11.5`。

---

原下一步（T3）的说明保留如下。**代码已就绪**
（`diag/exp_t3.py`，设计见 `PLAN_T0T4.md §10`，用法见 `README §3.5f`）：

```bash
# 1) CPU：标定，不排队。先看自检三条对不对，再花机时  ✅ 已完成，自检正常
python -m diag.exp_t3 --mode build \
    --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \
    --drift-from 200 --drift-to 400 --base-relative 0.156 --out-dir results/t3

# 2) dry-run：查缺文件 + 报格子数（不导入 torch，本机也能跑）
python -m diag.exp_t3 --mode eval \
    --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \
    --drift-from 200 --drift-to 400 --base-relative 0.156 \
    --data-root ./data --model-size 18 --device 0 --out-dir results/t3

# 3) 小规模试水：确认 zero 格复现 B2 的基线 ASR，再上全量
python -m diag.exp_t3 --mode eval ... --multipliers 1 --seeds 0 --max-clients 4 \
    --out-dir results/t3_pilot --execute

# 4) 全量 53 配方 × 全部良性客户端；被抢占后原样重跑加 --resume
python -m diag.exp_t3 --mode eval \
    --ckpt-dir checkpoints/attack_a0.5_s0_e1b_persist_s0 \
    --drift-from 200 --drift-to 400 --base-relative 0.156 \
    --data-root ./data --model-size 18 --device 0 \
    --out-dir results/t3 --execute
```

`--model-size 18` / `--data-root` / `--device` **必须按环境给全**，缺省值不一定对。
eval 需要 run **根目录**的 `meta.json` / `generator.pt` / `client_<cid>.pt`
（`hooks.save_run` 写的）加漂移两端的 `round_XXXX/global.pt`；dry-run 会一次报齐缺哪些。

问的是：同样 0.156 的幅度、换成**随机方向**，ASR 掉不掉？掉不动 = 宽盆 = (c)；
掉得动 = 良性训练的漂移方向特殊。六个方向族把「幅度」「坐标身份」「方向相干性」
逐个剥开。**判词在 ACC 掉超过 0.05 时拒绝给 ASR 结论** —— 把模型打坏也能让 ASR 掉。

---

## 1. 已确立的事实（真实集群数据，非冒烟）

| 事实 | 出处 |
|---|---|
| δ 主导、ξ 次要：只用 δ 掉约 10% ASR，只用 ξ 大幅下降 | 论文 ablation + 本地复现 |
| 检测**有效**：Multi-Krum Youden's J=+0.654，FLAME +0.225（bad=10）/ +0.327（bad=5） | 实验 J |
| 恶意密度越低越**好**检出：l2 AUC 0.9622 / 0.9225 / 0.8529（bad=1/5/10） | 实验 J |
| Invariant 的 AND-mask **完全不响应攻击者**，停在随机符号基线；trim 干全部的活 | 实验 I 掩码诊断 |
| 无防御时 ASR 在所有密度都 >80%（与原文一致）；FLAME/Invariant 在 bad=5 压到约 30%、bad=1 约 10% | 密度扫描 |
| 最终 ASR（尾-5 均值）：FedAvg 0.999 / Multi-Krum 0.979 / Invariant 0.915 / FLAME 0.834（均 bad=10） | 实验 J |
| ACC 代价：Multi-Krum 每压 1 个 ASR 点付 **3.28** 个 ACC 点（比不设防更差）；FLAME 0.73 | 实验 J |
| E 组（触发器跨环境迁移）**全部失败** | 实验 E |
| A–D（因果不变性特征测量）**无结果** | 实验 A/B/C/D |

## 2. 核心论点：检测有效，但不够（"曝光量"要按幅度算，不是次数）

用户的直觉"恶意终端需要一定曝光量才能植入后门"方向对，但**变量是幅度不是次数**：

| 设定 | 漏入聚合的恶意客户端-轮次 | ASR |
|---|---|---|
| fedavg, bad=1 | 47（全额未裁剪） | >0.80 |
| FLAME, bad=5 | 107 | ≈0.30 |
| FLAME, bad=10 | 234 | 0.834 |

漏得更多反而 ASR 更低 —— 因为 FLAME 把 95.1% 的恶意更新**裁到中位范数**，
漏过去的幅度被削掉了。所以自变量是

    E = Σ_t Σ_{k∈mal} I_k(t)·s_k(t)·‖g_k(t)‖ / D(t)

`diag/analysis_exposure.py`（"步骤 0"）已实现，从现有 npz 离线算，**无需重训**。
在合成 fixture 上验证过坍缩判据（含零假设，不硬拟合）。**真实数据上的坍缩与否
还没跑** —— 这是最便宜、最先该做的一步。

> ⚠️ 我曾说过一句错话："裁剪只是重新缩放、不是保护来源"。上面的算术说反了 ——
> **裁剪承担了大部分保护**。报告里要改掉。

## 3. 下一步研究方向（已与用户讨论，用户在权衡）

三条路，可用同一个杠杆 E 统一（各动一个因子）：

| 路 | 动什么 | ACC 代价 | 状态 |
|---|---|---|---|
| 1. 轻量对抗训练 | 抬高阈值 K | 客户端算力 + ACC | 补充，不是主线 |
| 2. 异常检测新指标 | 降漏过率与幅度 | 看方法 | **别按"找新指标"做**，见下 |
| 3. FedRep / 分层 FL | 缩小共享面 | 几乎不付 | 先跑实验 F 定生死 |

- **路 2 的正确形态**：不是"找一个抗 PGD-BKD 投影的指标"（军备竞赛，审稿人一眼
  看穿）。指标不缺（l2 AUC 0.85–0.96），缺的是**决策阶段**（FLAME 裁剪阶段
  95.1% vs 43.1% 分离度强，聚类阶段弱）。赢的写法是**把保证写在 E 上**：受投影
  约束的攻击者必须压低每轮幅度 → 需要更多轮 → 更多轮更多次被抓。有定理形状。
- **路 3 靠实验 F 定生死**（`README.md` §3.4 的 P0–P4，代码全就绪）：
  - **H_stable**（冻结 δ 在后续 θ 仍有效）→ 共享参数里有稳定的东西 → 路 3 活；
  - **H_coadapt**（δ 只对训练它的 θ_g 有效）→ 路 3 作为"测不变特征"是死的，
    而 A–D 的空结果是**正确答案**不是失败。
  - E 组全败强烈指向 H_coadapt。bad=5 的快照已 **92.5% 齐备**，可直接跑。
- 用户的"实际 Gap 在 FedRep→Bad-PFL→我的方法"直觉是对的：FedRep 类改进
  **结构性**降 E 不付 ACC 代价，鲁棒聚合**统计性**降 E 且要付。

## 4. 可运行的工具（都在 diag/，默认 dry-run 的用 --execute 真跑）

| 命令 | 作用 |
|---|---|
| `python -m diag.run_fl --mode attack ...` | 训练驱动，逐轮 npz + 周期评估 |
| `python -m diag.analysis_exposure` | 步骤 0：ASR vs 累积吸收幅度 E |
| `python -m diag.exp_f_precheck` → `exp_f` → `analysis_f` | 实验 F：时间衰减矩阵（定路 3 生死） |
| `python -m diag.exp_ij` / `analysis_ij` | 实验 I/J 汇总与五张图 |
| `python -m diag.analysis_density` | 图 J-6：ACC/ASR × 恶意客户端数 |
| `python -m diag.run_exp1` / `analysis_exp1` | **正式实验 1/1B**：剂量边界 + 攻击持续时间 |
| `python -m diag.represent` | 离线表征分析（trigger_pull 等） |
| `python -m diag.exp_t0` | **T0（Stage 0）**：全局模型的逐坐标位移剖面，纯 CPU 不排队 |
| `python -m diag.exp_t3` | **T3**：同幅度扰动 vs 真实漂移（`--mode build` 纯 CPU，`--mode eval` 要 GPU） |

## 4b. Stage B 收敛标定（导师意见 #5，**先于主力重跑**）

用数据定预算，不凭感觉把 local budget 从 15 steps 改成别的。

```bash
# 集群（Arrhenius）：整条链必须在 torch_fl.sif 容器里 —— run_exp1 --execute 用
# subprocess 起 `python -m diag.run_fl`，子进程继承容器命名空间。
sbatch diag/run_calib.sbatch          # 6 个 run（local_steps {15,45,75} x seed {0,1}）
sbatch diag/run_calib.sbatch 0        # 只跑 seed 0，先探路

# 登录节点看清单（dry-run，不需要 GPU，但仍要在容器里才 import 得到 diag.config）
source diag/cluster_env.sh && $PY -m diag.run_exp1 --stage calib --seeds 0 1

python3 -m diag.read_calibration      # 读数：纯 stdlib，容器内外都行
```

> ⚠️ **Bad-PFL 的容器是 `torch_fl.sif`，入口是 `python`**；tf-dpfl 那边是
> `tensorflow.sif` + `python3`。两个仓库不要混用。收口在 `diag/cluster_env.sh`。
> 守卫：`diag/tests/test_cluster_scripts.py`。

- 唯一自变量 = `local_steps in {15, 45, 75}`（≈ 0.38 / 1.15 / 1.9 epoch），
  剂量固定在十字扫描的交叉点 (Nm=4, ρ=0.1)。配置在 `config.yaml` 的
  `exp1.calibration` 段：`total_round: 80`（主力 200）、`eval_every: 2`（主力 5）。
- **`calib` 不含在 `--stage all` 里** —— 预算与主力不可比，混进并表就是把
  两批数字画进一张图。
- `read_calibration.py` 出「到平台的 GPU-秒」表。平台是**可计算的判据**
  （final = 末 3 点均值，平台轮 = 最早的「此后再没离开 final ± tol」的轮次）；
  判不出来报 `n/a`，不猜 —— 「80 轮还没平」是结论本身。
- 数据来自本轮新加的 `train_wall_s` / `eval_wall_s` 两列（`diag/track.py`）。
  **此前 implantation CSV 一个时间列都没有**，这个问题在 Bad-PFL 侧根本答不了。
  旧 CSV 没有这两列 → 报 `n/a` 而不是 0。
- 与 tf-dpfl 侧 (`experiments/calibration/`, `local_epochs in {1,3,5}`) 用**同一套
  判据**，但 local budget 的单位不同（step vs epoch），各标各的，别直接对读。

## 4a. FedRep 门禁（**在 Stage B/D 之前**）

```bash
sbatch diag/verify_fedrep.sbatch      # 约 30-60 分钟，比一个主力 run 还便宜
```

FedRep 是本轮在 `diag/pfl_fedrep.py` 新写的旁路实现，**从没在集群上跑过**，
而 Stage B/D 现在默认就是 fedrep（用户决策：Exp 1 与 Exp 3 对齐）。
不先验证 = 拿 100 个 GPU-run 赌一个没跑过的实现。

三步，任一步失败立刻非零退出：
1. `test_pfl_fedrep` 全部 15 条（本机只能跑 10 条，5 条需要 torch —— 在这里才真跑）
2. fedbn / fedrep 各一个 10 轮短 run
3. `evidence_fedrep` 对拍：fedrep 臂 `linear.*` 不动、BN 变；fedbn 臂反过来

## 4c. Stage D 主力重跑（Exp 1，job array）

**先跑完 Stage B 标定**（§4b）：单 run 成本由那边的 `GPU-s→plat` 给出，
没有它就只能猜总机时。

```bash
bash diag/submit_exp1.sh 1 --dry-run      # 看清单与规模，不提交
SEEDS="0" bash diag/submit_exp1.sh 1      # 单 seed 探路（13 个 job）
bash diag/submit_exp1.sh 1                # 十字扫描全量（65 个 job）
bash diag/submit_exp1.sh corner           # 塌陷角（30）
bash diag/submit_exp1.sh arm              # 对照臂 fedbn（3）
PFL=fedrep bash diag/submit_exp1.sh arm   # 对照臂 fedrep（3）
```

| 阶段 | 格 | seed | run | 本地 batch | 主力当量 |
|---|---|---|---|---|---|
| 门禁 `verify_fedrep` | — | — | 2 | 2,400 | 0.1× |
| B 标定 `calib` | 4 | 2 | 8 | 468,480 | 19.5× |
| 十字扫描 `1` | 13 | 5 | 65 | 1,560,000 | 65× |
| 塌陷角 `corner` | 6 | 5 | 30 | 720,000 | 30× |
| 对照臂 `arm` ×2 | 1+1 | 3 | 6 | 144,000 | 6× |
| **合计** | | | **111** | **2,894,880** | **120.6×** |

> 「主力当量」= 折算成多少个主力 run（1 个主力 run = 24,000 本地 batch）。
> **标定占 19.5×**，因为网格要覆盖导师要的 5 epoch = 195 steps = 基线的 13 倍。
> 这 19.5 个当量是全盘最有杠杆的部分 —— 它定的是后面 101 个 run 的单价。

- **一个 array task = 一个 run**（`exp1_array.sbatch`）。顺序跑装不下 101 个
  小时量级的 run，也没有断点续跑。默认并发上限 8，`--max-parallel N` 可调。
- 清单由提交器**一次性**生成并落盘到 `results/lists/`，array 按行号取 ——
  现生成的话 `--skip-existing` 一变行号就错位。
- 提交器用 **`$PY_NOGPU`**（不带 `--nv` 的同一容器）：生成清单在登录节点跑，
  那里没有 NVIDIA 驱动。
- `corner` 与 `arm` **不含在 `--stage all` 里**，按需单独提交。

## 5. 正式实验 1/1B（最近一次交付，代码就绪，等集群跑）

- 设定：**40 客户端 / ResNet-10 / 200 轮 / 2 seed**（`config.yaml` 的 `exp1` 段）。
  **⚠️ 骨干已从 ResNet-18 换成 ResNet-10**（2026-09-07）：与 tf-dpfl 的
  `build_resnet10` 同构（两边都是 BasicBlock [1,1,1,1]），两库的 Exp1/Exp3 才能
  同框比较；报告 §2 一直写着 "ResNet-10"，现在这句才成立。
  **换骨干前跑出来的 exp1 结果与之后的不可比**，别混进一张图。
  **⚠️ 换设定不是延续**：实验 A–J 是 100 客户端 / 1000 轮，与 exp1 同样不可比。
- 逐轮记录已扩展：主任务（clean_loss/mta/target-class acc，个性化+全局）、
  后门（逐 edge ASR 存 `exp_ij_edge_*.csv`）、参数（layer norm/cos、分组 cos）、
  表征（离线）。**`mta` ≠ `acc`**：mta 是全部样本，acc 是非目标类样本（ASR 分母口径）。
- 扫描是**十字形**（固定一个变量扫另一个），不是全因子。
- 六种调度（1B）：continuous/burst/intermittent/after_mta/early/late。
- **一个实测到的坑**：ASR-vs-MTA 会假阳性 —— MTA(t) 饱和会机械地让"越过时的
  MTA"显得集中。`threshold_verdict` 因此带零假设。这只是相关性证据，
  分因果要靠 1B 的 `onset_analysis`。

## 5b. ⚠️ 已定案：exp1 的低 ASR 是**测量口径**问题，不是攻击/轮数问题

2026-08 集群实测，逐层坐实：

| 证据 | 数值 | 结论 |
|---|---|---|
| diag 恶意正对照 `asr_targeted` | 0.98→0.999（尾段） | 攻击**完全植入**，排除欠训/轮数 |
| diag 良性个性化 `asr_personalized_targeted` | 0.19–0.33，且不升反降 | 加轮数救不了；不是训练长度问题 |
| **原始 `main.py`**（100/10/ResNet-10/300/fedbn） | **Avg ASR 91.95%**，逐客户端 57–100% | **原仓库复现论文**，无环境级 gap |

→ 后门**确实**存活在良性个性化模型上（~90%）。diag 早先读到的 ~0.25 是
**perturb 分解路径**（重建 δ/ξ + 共享 probe）在弱后门模型上低估所致，**不是
真实现象**。我一度把它解读成"后门穿不过个性化的发现"——**那是错的，已作废**。

**论文口径（`main.py:127–138`）**：遍历**全体**客户端，各自 `local_model`，
在**各自 test loader** 上用**原始 `full_poison_func`**（poison_ratio=1.0）、
`model.eval()`、**不过滤目标类**，求平均。

**已修**（埋点 12 / PATCHES）：`track.py` 新增 `_paper_asr` + `paper_eval_func`，
落 `asr_paper_benign/malicious/all` 与逐客户端 `asr_paper`；`analysis_exp1`
默认用 `asr_paper_all`（缺列回退旧列并告警）。**perturb 路径只留给实验 E**。
exp1 需**重跑**才有论文口径列（旧 CSV 无 `asr_paper_*`）。

## 5c. 2026-09-09 定案：Exp 1 走 **FedBN + 论文配置**；FedRep 通路**暂停**

### 决定

`exp1` 回到上游 `main.py` 的默认（`diag/config.yaml` 的 `exp1` 注释里有完整理由）：

| | 值 | 出处 |
|---|---|---|
| `client_num` | **100** | `main.py:26` |
| `select_per_round` | **10**（参与率 10%） | `main.py:28` |
| `total_round` | **300** | `main.py:23` |
| `local_steps` | **15 = 1 个 local epoch**（500 张 / 32，drop_last） | `main.py:32` |
| `pfl` | **fedbn** | `main.py:34`，且 `main.py:110` 只实现了这一个 |

**判据**：探针 A 在这套配置下跑出 **ASR 0.8201**，论文报 0.8222，本仓库用原始
`main.py` 复现出 0.9195（§5b）。**这是唯一被端到端验证过的通路。**

> **报告 §2 的「1 local epoch」是对的。** 此前算出「0.38 epoch」，是因为我们
> 自己把 `client_num` 改成了 40（每端 1250 张 → 1 epoch = 39 步）。
> 导师意见 #5 的「3–5 epoch」在 100 端下就是 **45 / 75 步**。

**代价（必须写进报告的方法学限制，不要藏）**：Exp 1 = FedBN、Exp 3 = hier_fedrep，
两库落在不同的 PFL 方法上 —— FedBN 个性化**归一化**，FedRep 个性化**分类头**。
两边的 ASR 绝对值**不可同框**，只能各自内部比较。

**剂量轴含义变了**：`Nm ∈ {1,2,4,8,16,32}` 在 100 端下是 **1%–32%**（此前 40 端是
2.5%–80%），论文自己的点（10 = 10%）落在网格里。

### FedRep 通路：查到了什么、还剩什么没解释

`diag/pfl_fedrep.py` 与 `--pfl fedrep` **保留未删**。三轮探针（每轮约 3 GPU-h）
查出并修好了四个缺陷，但 **FedRep 臂的准确率始终追不上 FedBN，原因未定**。

已修（都留在代码里，且各有守卫）：

| # | 缺陷 | 守卫 |
|---|---|---|
| 1 | 评估用 `[漂移后的 φ′, 阶段1 的头]`，头/骨干失配。FedRep 的个性化模型是 `[共享表示, 私有头]` | `merge_shared_and_private` + `test_merge_*` |
| 2 | `_fedrep_set_phase` 用 `m.eval()` 冻 BN，被 `client.py:local_update` **每步**的 `.train()` 撤销 → 头阶段 BN running stats 一直在更新并上传 | `test_norm_freeze_uses_track_running_stats_not_eval_mode` |
| 3 | `_FakeBaseClient.local_update` 漏了 `.train()` → 缺陷 2 在本地全绿（**假绿**） | `test_the_fake_base_client_replicates_the_real_local_update` |
| 4 | `acc_local_*` 用了 `utils.evaluate_accuracy` 的**百分数**没换算，比邻列大 100 倍 | `test_per_client_accuracy_is_a_fraction_not_a_percentage` |

顺带确立的一件事（**对 FedBN 一样重要**）：`mta_personalized` 测在
`self.probe`（共享的**类别均衡**探针），而 `asr_paper_*` 测在客户端**自己的**
test loader —— 两个数字不在同一个 population（tf-dpfl 陷阱 #11 的同一类错误）。
新增的 `acc_local_personalized` 与 ASR 同 population，也与 tf-dpfl 的 `pm_acc` 同口径。

**仍未解释**：修完这四条之后，FedRep 臂在**逐客户端分片**上的准确率仍低于 FedBN。
缺陷 2 的修复把 ASR 从 0.066 抬到 0.111（unfiltered，同 seed 同配置），
方向明确但量级不足以解释准确率的差。

**下一个该查的**（还没查）：**私有头在轮次之间到底有没有被保住**。
`use_fedrep` 的 `fedrep_distribute` 把 `linear.*` 从下发字典里 pop 掉，配
`server.py:32` 的 `load_state_dict(..., strict=False)`，理论上客户端保留自己的头 ——
但**这条路径从来没有被端到端验证过**：`evidence_fedrep` 验的是**全局模型**的
`linear.*` 不动，那与「客户端的头没被覆盖」是两回事。
验法：在 `receive_model` 前后各存一次客户端的 `linear.weight`，断言逐位相同。
这条**不需要 GPU 上的完整 run**，一个几轮的 smoke 就够。

## 6. 待办 / 未决

- [ ] **最优先**：真实数据上跑 `analysis_exposure`（步骤 0），看 E 是否坍缩成一条曲线。
- [ ] **实验 F**（步骤 1）：定路 3 生死。快照已备。
- [ ] **exp1 需重跑**：换论文口径 ASR（埋点 12）后，旧 CSV 没有 `asr_paper_*` 列，
      `analysis_exp1` 会回退旧口径并告警。重跑 26 个 run 才能得到论文尺的 ASR。
- [ ] 实验 I §6 归因图（后门维度 sign consistency `c_k` 分布 vs 随机维度）——
      从现有 checkpoint 可算，一直没做。用户称其为"论文核心图"。
- [ ] "Invariant 弱于 FLAME"在 bad=5 上两者都约 30% 无分离，要下这个结论需 ≥3 seed。
- [x] ~~E1 vs 论文 82.22% 的复现 gap~~ **已定案**：原始 `main.py` 复现出
      Avg ASR 91.95%（见 §5b），gap 是 diag 的 perturb 测量口径造成的，不是复现
      失败。exp1 改用论文口径后应对齐。
- [ ] 长期开口：`sign_consistency` 的逐客户端定义是我对 Wang et al. 的推广，
      需对照原文确认。

## 7. 提交历史锚点（最近）

```
dc6d132 正式实验 1/1B —— 扫描驱动、六张图、离线表征分析
96477bd 正式实验 1/1B 的埋点基础 —— 剂量、调度、逐轮全指标
bc87ccb 步骤 0 —— ASR vs 累积吸收的恶意幅度 E
f0b5eb9 图 J-6 + 密度写进数据
6b22457 静默恶意客户端（oracle_exclude）
```
