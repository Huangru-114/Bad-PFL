# 口径说明（Metric conventions）—— 回应导师意见 #1 #2

> **两库各存一份，内容相同**：`Bad-PFL/diag/METRICS.md` 与
> `tf-dpfl/experiments/METRICS.md`。改一处要同步改另一处。
> 每一条都注明了代码出处，**不要凭记忆改**。
> 最后核对：2026-09-07（骨干统一到 ResNet-10、target 统一到 0、ASR 探针改为留出分片之后）。

导师的两条：
> (1) specify target/source classes, trigger types, visualize the attacks
> (2) specify how test data is processed, how many samples got backdoors

下面是逐条的确切答案。**英文版在最后一节，可直接贴进报告。**

---

## 1. 威胁模型：target / source class

| 项 | 值 | 出处 |
|---|---|---|
| 攻击类型 | **all-to-one** | 下方"source class"一行 |
| target class | **0 = airplane**（CIFAR-10） | `Bad-PFL/main.py:36` (`--ba_target_label` 默认 0)；`tf-dpfl` 各 config 的 `backdoor.target_label: 0` |
| source class | **无限制** —— 任何样本都可被投毒 | `fba.py:48` 的 `poison_mask` 只按 `rand() <= ρ` 抽，不看标签 |
| 恶意端标签改写 | 被选中的样本标签**全改成 target** | `fba.py:51` `torch.full([...], target_label)` |

> ⚠️ **target class 曾经两库不一致**：tf-dpfl 一度是 `9 = truck`，Bad-PFL 一直是 0。
> 2026-09 统一到 0。**统一之前的两库数字不可同框讨论**，Exp 3 的旧结果已归档到
> `experiments/attack/hfl-propagation/results/archive-pre-fix/`。

---

## 2. 触发器：公式与预算

Bad-PFL（ICLR 2025）的触发器是**两个分量之和**，两者都受 `ε = 4/255` 约束：

```
T(x) = clip_{[0,1]}( x + ξ(x) )  +  δ(x)

  ξ(x)  1 步 PGD 扰动，ε = α = 4/255           fba.py:6  pgd_attack
        adv = clip(x + U(-ε, ε))                       # 随机起点
        adv = clip( x + clip(α·sign(∇_x CE(f(adv), y_true)), -ε, ε) )
        ——**无目标**（梯度上升，推离 y_true），不是推向 target

  δ(x)  自编码器输出，Tanh 后按 4/255 缩放      fba.py:38  generator.py:34
        δ = Autoencoder(x) / 255. * 4.   与 Tanh ∈ [-1,1] 合成 δ ∈ [-4/255, 4/255]
```

- **两个分量是相加的**，所以 `T(x)` 相对 `x` 的总 L∞ 预算最坏情况是 **8/255**，
  不是 4/255。（`x + ξ` 先 clip 到 [0,1]，`δ` 加上去之后代码里**没有再 clip**
  —— `fba.py:53`。这一点报告里要写明。）
- δ 的生成器**只在恶意客户端自己的模型上训练**（`fba.py:32-42`，30 步 Adam lr=1e-2），
  评估时却要迁移到良性客户端的个性化模型上 —— 见陷阱 #5，这是已知的方法学限制，
  不是 bug。
- ξ 依赖 `client.local_model`：**触发器是 model-dependent 的**，同一张图在不同
  客户端/不同轮次上得到的扰动不同。所以"触发器可视化"必须注明是哪个模型、哪一轮。

### 触发器可视化（导师第 1 条的 "visualize the attacks"）

```bash
# 先查 checkpoint 齐不齐（不需要 torch，秒级）
python -m diag.viz_trigger --ckpt-dir checkpoints/<run> --client-id 0 --check-only
# 出图（CPU 就够，不用排 GPU 队）
python -m diag.viz_trigger --ckpt-dir checkpoints/<run> --client-id 0 \
    --data-root ./data --out results/figs/trigger_visualization.png
```

五联图：clean `x` / ξ / δ / `T(x)` / 总残差。三个扰动 panel **按 4/255 满量程
放大 32 倍**并把倍数写在标题里 —— 不写倍数的扰动图看不出它是 4/255 还是 40/255。
超范围**截断**而不是重新归一化，所以跨 panel 可比（总残差能到 8/255 这件事
画得出来）。每个 panel 还打出实测的 L∞ 及其占 4/255 的比例，读者可自行核对预算。

### ρ（poison rate / poisoning probability）的定义

**逐样本的伯努利概率**，不是"每批固定投多少张"：

```python
poison_mask = torch.rand(label.size(0)) <= poison_ratio     # fba.py:48
```

- 每个 batch 里被投毒的张数是 `Binomial(batch_size, ρ)` 的一次抽样，不是 `ρ·B` 的定值。
- 未被选中的样本**原样保留**（图与标签都不动）——`fba.py:53-54` 的 mask 混合。
- **ρ = 1.0 是一个奇点，不是"更强的攻击"**：那时恶意客户端的
  `fetch_data()` 返回的已经是投毒过的数据、标签全是 target，而生成器训练
  （`fba.py:36-37`）把它当"干净数据+干净标签"用 → `pgd_attack` 在做**推离 target**
  的无目标 PGD，而生成器损失（`fba.py:39`）在**推向 target**，两个分量对消。
  报告里 ρ=1.0 那格的塌陷有这个机制解释，导师第 4 条问的就是它。
  **这一条是代码路径推断，尚无数值证据** —— Stage 3 的 ρ ∈ {0, 0.3, 0.7, 0.9}
  扫描就是去判定它是"渐进下滑"还是"孤立奇点"。

### 评估时用的触发器

`eval_func = partial(our_poison_func, poison_ratio=1.0, client=mal[0])`（`fba.py:63`）
—— 评估时 **ρ 强制为 1.0**，即探针里**每一张图**都被打上触发器。
ξ 用的是 `mal[0]` 的模型。

---

## 3. 测试数据怎么构造的（两库**不一样**）

### Bad-PFL（扁平两层，Exp 1 / 1B）

```python
# main.py:65-77
client_train_sample_nums = [len(train_dataset) // client_num] * client_num
client_test_sample_nums  = [len(test_dataset)  // client_num] * client_num
#   官方 train(50000) 与官方 test(10000) **各自独立**做 Dirichlet 划分
#   （两次共用同一份 class_priors，所以每个客户端的类别分布 train/test 一致）
```

| 项 | 值（`client_num = 40`） |
|---|---|
| 每客户端训练样本 | 50000 / 40 = **1250** |
| 每客户端测试样本 | 10000 / 40 = **250** |
| DataLoader | `batch_size=32, drop_last=True`（`main.py:82-83`） |
| **实际参与评估的张数** | `floor(250/32) × 32 = 224`（每客户端） |
| 报 ASR 的客户端数 | `min(10, len(benign_ids))` 个良性端 + 全部恶意端（`diag/run_fl.py:457`） |

> **官方 train 与 test 从不合并** → Bad-PFL 侧**没有**训练/评估重叠问题。
> `drop_last=True` 意味着每客户端有 26 张（250−224）从不参与评估；
> 这是上游行为，铁律 #1 下不改，但要在报告里写明。

### tf-dpfl（三层 HFL，Exp 3）

按 **PFLlib 口径**：先把官方 train+test **合并**，再逐客户端划分，
最后在**每个客户端分片内部**切 train/test。

```python
# fedavg/main.py:~687
x_all = np.concatenate([x_train, x_test])          # 60000
clients, ... = build_clients(x_all, y_all, ...)
# data/partition.py: noniid_partition 是一个**划分**（每个 index 恰好归一个客户端）
#                    split_client_train_test 再在分片内部按 per_client_test_ratio 切
```

**合并本身不造成泄漏**：分区是划分 + 切分在客户端内部 ⇒
「所有留出分片的并集」与「所有训练数据的并集」**全局不相交**。

| 项 | 值（`n_clients = 100`, `per_client_test_ratio = 0.25`） |
|---|---|
| 每客户端样本 | 60000 / 100 ≈ **600**（非 IID，实际有涨落） |
| 每客户端留出 | ≈ **150** |
| 留出里的非目标类（= ASR 分母） | ≈ 150 × 0.9 ≈ **135** |
| `asr_max_samples` | 2000（取**前** N 个合格样本；测试集 `shuffle=False`，所以确定性） |

> ⚠️ **曾经错的地方在评估侧，已修**：`BackdoorCloudServer` 一度把**原始 `x_test`
> 数组**又当独立探针去算六个 ASR —— 合并之后那些图已经分给客户端、其中
> 1−0.25 = 75% 进了训练集，等于在训练过的图上测 ASR。现在三层各测各的留出分片：
>
> | 层 | 探针 |
> |---|---|
> | global | `merge_test_datasets(所有 edge)` |
> | edge | `edge.get_test_dataset()` |
> | client | `client.test_dataset`（与它自己的 `pm_acc` **同一个集合**） |
>
> **影响幅度未知，不要替它下结论**：落在恶意端训练分片的那部分被真的投毒训过
> （记忆 → 抬高 ASR），落在良性端的按正确标签训过（更难翻 → 压低 ASR）。
> 净效果要同一 config 跑新旧两套探针各一次才能定。
> 量化工具：`bash run_evidence.sh`（不需要 GPU）。

---

## 4. ASR 的分母：filtered vs unfiltered

**这是两库此前最大的口径分歧，也是导师第 2 条问的核心。**

| 口径 | 分母 | 用在哪 |
|---|---|---|
| **filtered**（社区通行，**报告正文一律用这个**） | 真实标签 **≠ target** 的样本 | tf-dpfl 一直是这个（`attack/backdoor_eval.py:43` 的 `mask = (y != target)`）；Bad-PFL 新增列 `asr_paper_filtered_*` |
| unfiltered | **全部**样本（含真实标签本来就是 target 的） | Bad-PFL 此前唯一的口径，逐行复刻上游 `main.py:131`，列名 `asr_paper_*` |

CIFAR-10 类别均匀时目标类约占 1/10，所以

```
filtered ≈ (unfiltered − 0.1) / 0.9
```

**两者有约 10 个百分点的系统差**。报告 §2 写的定义是 filtered，而 Exp 1 的图
此前画的是 unfiltered —— 图注 E1-6 自己写着 "All still unfiltered"，与 §2 直接打架。
现在 `diag/track.py` 在**同一次前向**里同时算出两个口径（零额外机时），
图注按实际用的列自动说明是哪一个。

### 三档 scope（Bad-PFL）

| 档 | 含义 |
|---|---|
| `benign` | 只在**良性**客户端的个性化模型上测 = 后门传给受害者的强度。**报告只用这一档。** |
| `malicious` | 恶意客户端自己的模型（≈ 1.0，是攻击者的自查，不是效果） |
| `all` | 前两者按客户端数的**无权平均** —— 随 `Nm` 机械上升，**不要写进报告** |

### 无定义 ≠ 0

任何分组为空时一律报 `None` / `n/a` / `nan`，**绝不填 0**：
- tf-dpfl：`compute_asr` 无合格样本返回 `None`（非 IID 下客户端留出分片可能
  一个非目标类样本都没有）；`_mean`/`_std` 空组返回 `None`；日志打 `n/a`；
  `collect_metrics` 解析成 JSON `null`；`plot_exp3.py` 标 "n/a" 而不画 0 柱。
- Bad-PFL：铁律 #5。

---

## 5. 两库之间**不能**直接比什么

| 可比 | 不可比 |
|---|---|
| 骨干（都是 ResNet-10 = BasicBlock [1,1,1,1]，2026-09 统一） | **ASR 绝对值** —— 探针不同（Bad-PFL 是官方 test 的独立划分，tf-dpfl 是合并后的留出分片） |
| target class（都是 0） | **local budget** —— Bad-PFL 是 `local_steps`（15 步 ≈ 0.38 epoch），tf-dpfl 是 `local_epochs`（真的整个数据集一遍） |
| 攻击实现（同一套 T(x) 与 ρ 语义） | **PFL 框架** —— Bad-PFL 默认 FedBN（BN 私有、分类头共享），tf-dpfl 是 FedRep（分类头私有、BN 聚合）。Bad-PFL 侧新增了 `--pfl fedrep` 对齐臂 |
| ASR 口径（都出 filtered） | **客户端数与参与率** —— 40 客户端选 8 vs 100 客户端选 10 |

---

## 6. 英文版（可直接贴进报告 §2）

**Threat model.** All-to-one backdoor with no source-class restriction: any sample
may be poisoned regardless of its true label. Target class **0 (airplane)** for
CIFAR-10 in both codebases.

**Trigger.** The trigger is the sum of two L∞-bounded components, both with
ε = 4/255:

  T(x) = clip_[0,1]( x + ξ(x) ) + δ(x)

where ξ(x) is a **single-step, untargeted** PGD perturbation (random start,
α = ε = 4/255, ascending the cross-entropy of the *true* label) computed against
the attacker's own local model, and δ(x) is the output of a convolutional
autoencoder whose Tanh output is rescaled by 4/255. Because the two components
are added and no clipping is applied after δ, the worst-case L∞ distance from
the clean image is 8/255. The trigger is therefore **model-dependent**: the same
image yields different perturbations for different client models and rounds.

**Poisoning probability ρ.** ρ is a **per-sample Bernoulli probability**, not a
fixed per-batch count: each sample in a batch is poisoned independently with
probability ρ, so the number poisoned per batch is Binomial(B, ρ). Unselected
samples keep both their image and their label. At evaluation time ρ is forced to
1.0, i.e. every probe image carries the trigger.

**Test data.** The two codebases partition data differently and their ASR values
are therefore **not directly comparable in absolute terms**. In the flat
two-layer setting (Experiments 1/1B) the official CIFAR-10 train and test splits
are partitioned independently across 40 clients with a shared Dirichlet class
prior (α = 0.5), giving 1250 training and 250 held-out images per client; the
evaluation loader uses batch size 32 with `drop_last=True`, so **224 images per
client** are actually scored. ASR is reported over the benign clients
(min(10, #benign) of them). In the hierarchical setting (Experiment 3) we follow
the PFLlib convention: the official train and test splits are merged and then
partitioned across 100 clients, and each client's own shard is split 75/25 into
train and held-out. Because the partition is a true partition and the train/test
split happens *inside* each shard, the union of all held-out shards is globally
disjoint from all training data. Each level is evaluated on its own held-out
data — the global model on the union of all edge held-out sets, each edge model
on its own, and each client on the same shard used for its personalized accuracy.
This yields roughly 150 held-out images per client, of which ~135 are non-target
class and thus enter the ASR denominator.

**ASR convention.** Unless stated otherwise, all ASR values in this report are
**filtered**: the denominator excludes samples whose true label is already the
target class. For CIFAR-10 with a roughly uniform class distribution,
filtered ≈ (unfiltered − 0.1) / 0.9, a systematic gap of about 10 percentage
points. Values are reported for benign clients only; the attacker's own models
trivially reach ≈ 1.0 and averaging them in would make the metric rise
mechanically with the number of malicious clients. Any group that is empty under
a given configuration is reported as undefined (blank / `n/a`), never as zero.
