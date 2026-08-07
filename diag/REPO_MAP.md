# REPO_MAP.md — Bad-PFL 仓库侦察报告 (Phase 0)

> 本文档只描述**现状**，不含任何实验结论。所有行号基于 commit `ad845a5`（`main` 分支）。
> 阅读方式：带 ⚠️ 的条目是会影响诊断实验设计的发现，带 🛑 的条目是**必须先讨论才能继续**的阻塞项。

---

## 1. 文件清单与职责

| 文件 | 行数 | 职责 |
|---|---|---|
| `main.py` | 138 | 唯一入口。参数解析、数据划分、客户端/服务器构造、PFL 配置、攻击配置、启动训练、最终评估 |
| `fl_process.py` | 46 | FL 主循环 `basic_fl_process` |
| `client.py` | 141 | `BasicClient` / `PMClient` / `PoisonClient` / `PMPoisonClient` |
| `server.py` | 46 | `BasicServer` + `agg_avg` |
| `fba.py` | 68 | **攻击核心**：`pgd_attack`（ξ）、`use_our_attack`（生成器训练 + 投毒函数） |
| `generator.py` | 42 | `Autoencoder` —— 生成 δ 的网络 |
| `pfl.py` | 25 | `use_fedbn` —— 唯一实现的 PFL 方法 |
| `trigger.py` | 44 | `grid_trigger_adder`（BadNets 式方块触发器）— **在 `main.py` 中未被调用** |
| `utils.py` | 117 | 种子、混合精度装饰器、客户端采样、评估、Dirichlet 划分、划分统计 |
| `event_emitter.py` | 49 | 全局事件总线 `fl_event_emitter` |
| `resnet.py` / `mobilenet.py` / `densenet.py` | — | 模型定义（`main.py` 只用 resnet） |

---

## 2. 执行流程

### 2.1 `main.py` 逐步骤

| # | 步骤 | 位置 |
|---|---|---|
| 1 | 解析参数 | `main.py:17-41` |
| 2 | 设备选择 + `set_random_seed(args.seed)` | `main.py:49-53` |
| 3 | 构造 CIFAR-10 train/test dataset（`download=False`，仅 `ToTensor()`，**无归一化、无数据增强**） | `main.py:58-60` |
| 4 | 计算每客户端样本数：train `50000/100=500`，test `10000/100=100` | `main.py:65-66` |
| 5 | `class_priors = np.random.dirichlet([dir_alpha]*10, size=client_num)` | `main.py:67` |
| 6 | **数据划分**：train 与 test 各调一次 `client_inner_dirichlet_partition`，**共用同一份 `class_priors`**（所以同一客户端的 train/test 分布一致） | `main.py:68-77` |
| 7 | 构造 100 个 `DataLoader`（`SubsetRandomSampler`, `batch_size=32`, **`drop_last=True`**） | `main.py:78-84` |
| 8 | 构造客户端：前 `100-10=90` 个 `BasicClient`，后 10 个 `PoisonClient`（`poison_func=None`，稍后注入） | `main.py:89-94` |
| 9 | `shuffle(clients)`，然后按新顺序赋 `client.cid = idx` | `main.py:95-99` |
| 10 | 构造 `BasicServer` + 全局 resnet10 | `main.py:103-106` |
| 11 | PFL 配置：`if args.pfl == "fedbn": use_fedbn(server)` | `main.py:110-111` |
| 12 | 攻击配置：`if args.ba == "our": full_poison_func = use_our_attack(...)` | `main.py:116-117` |
| 13 | 训练：`basic_fl_process(...)` | `main.py:122-123` |
| 14 | 评估：对**每个客户端**算 ACC 与 ASR，打印均值/标准差 | `main.py:127-138` |

### 2.2 训练循环调用栈（`fl_process.py:4-42`）

```
basic_fl_process(server, clients, local_steps=15, training_rounds=300, select_rule)
└── for cur_round in 1..300:
    ├── client_indices = select_rule(server, clients)          # utils.py:23-26, torch.randperm 取前 10
    ├── global_state = server.distribute_model()               # server.py:23-27
    │   └── hook "before_distribute_global" → fedbn_distribute # pfl.py:14-21 (删掉 bn / shortcut.1 键)
    ├── for indice in client_indices:                          # 串行，非并行
    │   ├── clients[i].init_round()
    │   ├── clients[i].receive_model(global_state)             # client.py:23-25
    │   │   ├── load_state_dict(..., strict=False)             # 缺失的 BN 键保留本地值 = FedBN
    │   │   └── hook "before_local_training"
    │   │       └── trigger_gen_trainer(client)   ← 仅恶意客户端  # fba.py:31-43
    │   └── for _ in range(15): clients[i].local_update()      # client.py:32-41
    │       ├── optimizer.zero_grad()
    │       ├── data = fetch_data()      ← PoisonClient 在此投毒  # client.py:124-125
    │       ├── pred = forward(data)     ← autocast 包裹
    │       ├── loss = loss_computation(pred, data)
    │       └── backward_and_update(loss, optimizer)           # GradScaler
    └── server.agg_and_update([c.upload_model() for c in selected])  # server.py:29-34
        └── hook "before_update_global" → fedbn_update         # pfl.py:5-12
```

**事件总线现状**：`fl_process.py` 发射 `on_fl_begin` / `on_round_begin` / `on_client_begin` / `on_client_end` / `on_round_end` / `on_fl_end`，但**当前没有任何代码注册监听器**。这是理想的埋点接入点 —— 埋点只需 `fl_event_emitter.on("on_fl_end", handler)`，**零侵入**。

---

## 3. 数据

### 3.1 划分逻辑

- **位置**：`utils.py:51-85` `client_inner_dirichlet_partition`
- **支持的划分方式**：**只有 Dirichlet 一种**。`args.client_dist`（默认 `"non_iid"`）被解析但**从未被读取**，没有 IID 分支。
- **异构度参数**：`--dir_alpha`，**默认 `0.5`**。α 越小越异构。
- **算法**：每客户端一个 Dirichlet 先验 `class_priors[cid]`，循环随机挑客户端 → 按其先验采样类别 → 若该类样本耗尽则随机换一个还有余量的类。所以**每客户端样本数严格等于 500 / 100**，但在 α 很小时后期会因类耗尽而偏离先验。
- `client_sample_nums` 被**原地修改**（`utils.py:68` 递减至 0），调用方不能复用同一个 list（`main.py` 为 train/test 各建了一个，没问题）。

### 3.2 客户端 / 恶意客户端配置

| 项 | 参数 | 默认 | 位置 |
|---|---|---|---|
| 客户端总数 | `--client_num` | 100 | `main.py:26` |
| 恶意客户端数 | `--bad_client_num` | 10（即 10%） | `main.py:27` |
| 每轮采样数 | `--select_client_num_per_round` | 10 | `main.py:28` |
| 本地步数 | `--client_local_step` | 15（是 **step 数**，不是 epoch） | `main.py:32` |
| 投毒比例 | `--ba_poison_rate` | 0.2 | `main.py:37` |
| 目标类 | `--ba_target_label` | **0**（CIFAR-10 = airplane） | `main.py:36` |
| 总轮数 | `--total_round` | 300 | `main.py:23` |

⚠️ **恶意客户端与数据分片的绑定**：`PoisonClient` 恒定拿到分片索引 `90..99`（`main.py:91-94`），`shuffle` 只打乱 `cid` 标签，不改变"哪个分片是恶意的"。所以恶意客户端的标签分布 = 分片 90-99 的分布，与 `cid` 无关。埋点必须同时记录 `cid` 和原始分片索引。

### 3.3 🛑 随机性审计（**最关键的发现**）

`set_random_seed`（`utils.py:8-13`）**只设置了 torch 的种子**：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)          # 仅在 cuda 可用时
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**没有 `np.random.seed()`，没有 `random.seed()`。** 后果：

| 问题 | 影响 | 严重性 |
|---|---|---|
| `np.random.dirichlet`（`main.py:67`）与 `client_inner_dirichlet_partition` 内部全部 `np.random.*`（`utils.py:65,71,74`）未播种 | **数据划分完全不可复现**。同一条命令跑两次得到不同划分 | 🛑 阻塞 |
| `shuffle(clients)`（`main.py:95`，`random.shuffle`）未播种 | `cid` ↔ 分片映射不可复现 | 🛑 阻塞 |
| 客户端采样 `torch.randperm`（`utils.py:26`）共用**全局 torch RNG** | 见下 | 🛑 阻塞 |

⚠️ **攻击开关会让两次 run 的随机流错位**：数据划分发生在 `main.py:67-84`，**早于**攻击配置（`main.py:116`），且 clean/attack 两种模式在划分前消耗的 torch RNG 量相同（`BasicClient` 与 `PoisonClient` 都各建一个 resnet10）。所以**只要补上 numpy/random 种子，划分本身在两种模式下就是一致的**。

但**训练开始后**，攻击 run 会额外消耗 torch RNG：
- `Autoencoder()` 初始化（`fba.py:27`）
- `pgd_attack` 每次调用的 `torch.zeros_like(images).uniform_(...)`（`fba.py:8`）
- 生成器训练每轮 30 次迭代（`fba.py:33`）

而 `random_select` 用的是**同一个全局 torch RNG**（`utils.py:26`）。因此 **clean run 与 attack run 每轮选中的客户端集合会完全不同**。这会破坏对照实验（两组的"每个客户端被训练了多少次"不一致）。

**建议修复（需你确认）**：给 `random_select` 一个独立的 `torch.Generator`，只由 seed 决定，不受训练过程中的 RNG 消耗影响。这会改变原实现的采样序列，但不改变采样分布，**不影响攻击语义**。

---

## 4. 攻击组件

### 4.1 生成器

- **定义**：`generator.py:6-40` `Autoencoder`，4 层 stride-2 卷积编码器（3→16→32→64→128）+ 4 层转置卷积解码器（128→64→32→16→3），末端 **`Tanh()`** → 输出 ∈ [−1, 1]，与输入同形状 `[B,3,32,32]`。
- **实例化**：`fba.py:27`，`Adam(lr=1e-2)`（`fba.py:28`）。
- ⚠️ **全体恶意客户端共享同一个生成器实例和同一个优化器**（闭包捕获，`fba.py:27-28`，在 `fba.py:60-65` 的循环中被所有 `PoisonClient` 共用）。这是"串谋"假设。
- **保存/加载**：**当前完全没有**。需新增（普通 `state_dict`）。
- **训练时机**：注册在 `"before_local_training"`（`fba.py:62`），该 hook 在 `receive_model` 中于 `load_state_dict` **之后**立即触发（`client.py:23-25`）。所以生成器优化时，`client.local_model` **≈ 当轮全局模型**（仅 BN 相关参数因 FedBN 保留本地值）。
  → **这一点直接支撑 H1 的前提：δ 确实是对着（近似的）全局模型优化的。**

### 4.2 δ 与 ξ 🛑 **可分离，确认**

`fba.py:46-58` `our_poison_func` 的合成过程：

```python
poison_data = pgd_attack(client.local_model, poison_data, label).detach().clone()   # 行53 → x + ξ
gen_trigger = trigger_gen(data) / 255. * 4.                                          # 行54 → δ(x)
poison_data = mask * (poison_data + gen_trigger) + (~mask) * data                    # 行55 → x + ξ + δ
```

| 组件 | 表达式 | L∞ 预算 | 性质 |
|---|---|---|---|
| **ξ** | `pgd_attack(model, x, y) − x` | 4/255 | **无目标** PGD（对真实标签 y 做梯度上升），破坏真实类特征 |
| **δ** | `trigger_gen(x) / 255. * 4.` | 4/255（Tanh∈[−1,1] × 4/255） | 生成器输出 |
| 合成 | `x + ξ + δ` | ≤ 8/255 | 最终触发器 |

**结论：δ 与 ξ 可以完全独立取出**。`delta_only` = `x + δ`，`xi_only` = `pgd_attack(...)`。`perturb.py` 可以按规格实现。

但有 **四个必须先讲清楚的性质**，它们会改变实验 A/B/C 的解释：

#### ⚠️ (a) δ 不是一个固定张量，而是**输入条件化**的
`δ = trigger_gen(x)`，每张图片的 δ 不同。所以：
- `hooks.py` 规格里的 `delta.pt` **无法保存"最终的 δ"**——只能保存**生成器权重**。
- 任务书 §1 的表述"同一个 δ 在不同客户端本地模型上"需要修正为：**同一个生成器 + 同一批图片 → 同一个 δ，施加到不同客户端模型上**。H1 依然可测，但措辞要改。
- 我的计划：`generator.pt` 存权重；`delta.pt` 只存"在一个固定参考 batch 上算出的 δ"，仅作可视化/存档，**不参与任何指标计算**。

#### 🛑 (b) ξ 是**模型相关**且在评估时现算的 —— 需要你做设计决策
`pgd_attack(client.local_model, ...)` 用的是**当前客户端自己的模型**。跨客户端比较时有两种选择：

| 方案 | ξ 的来源 | 含义 | 风险 |
|---|---|---|---|
| **A. 攻击者忠实** | 每个客户端用自己的模型算 ξ | 与原攻击一致 | ξ 本身就跨客户端不同 → 测到的离散度**无法归因给 δ** |
| **B. 受控** | ξ 固定用同一个参考模型（如全局模型）算一次 | 隔离 δ 的变化 | 与原攻击不完全一致 |

**实验 A 的核心是"δ 的跨客户端离散度"，我倾向方案 B（并把方案 A 作为附加对照跑一遍）。这需要你拍板。**

另注：ξ 需要**真实标签 y**（无目标 PGD 对 y 上升）。诊断场景下标签已知，可行。

#### ⚠️ (c) ξ 是**随机**的
`pgd_attack`（`fba.py:8`）以 `uniform_(-ε, ε)` 随机初始化，且 `num_iter=1`、`alpha == epsilon == 4/255`——即一次符号步后 clamp 回 ε 球，实际接近 FGSM。同一输入多次调用得到**不同的 ξ**。诊断代码必须固定 RNG 并记录，否则指标不可复现。

#### ⚠️ (d) `pgd_attack` 有**副作用**：污染模型梯度
`fba.py:16` 的 `loss.backward()` 会把梯度累加进**模型参数**。在 `local_update` 中 `zero_grad()` 发生在 `fetch_data()` **之前**（`client.py:34-36`），所以 PGD 的梯度会混入本轮参数更新——这是原实现的既有行为，**我不会改**。但我的诊断代码复用 `pgd_attack` 时必须自己 `zero_grad()`，避免对加载的模型产生副作用。

#### ⚠️ (e) 生成器训练用的"clean_data"其实是**已投毒**的数据
`fba.py:36` `clean_data, clean_label = client.fetch_data()`，而 `PoisonClient.fetch_data`（`client.py:124-125`）已被重载为"先取数再投毒"。所以变量名 `clean_data` 名不副实，生成器是在**带 δ+ξ 的输入**上训练的（自递归）。这是原实现的行为，**不改**，但埋点文档里要写明。

### 4.3 ASR 的计算方式

`main.py:131`：`asr = evaluate_accuracy(client.local_model, client.test_dataloader, full_poison_func)`

| 维度 | 现状 |
|---|---|
| 在哪个模型上 | **每个客户端自己的 `local_model`**（FedBN 下即个性化模型；本仓库没有独立的 personalized model） |
| 在哪些样本上 | 该客户端的**本地测试集**（约 100 样本，`drop_last=True` 后实际 96） |
| poison_ratio | `1.0`（`fba.py:64` 的 `eval_func`） |
| ⚠️ **是否排除目标类原样本** | **否**。`our_poison_func` 把**全部**标签改成 `target_label`（`fba.py:52`），本来就是 airplane 的样本会被算作攻击成功 → **ASR 被目标类占比系统性抬高** |
| ⚠️ ξ 用谁的模型算 | `eval_func` 在 `fba.py:64` 绑定的是**循环里最后一个恶意客户端**的 `client`。所以对所有客户端评估 ASR 时，ξ 都由那**一个特定客户端**的模型生成 —— 但被测模型是各客户端自己的。这是个不一致之处 |

→ 实验 C 里我会自己定义 ASR（排除目标类原样本），并在文档中与原实现的口径并列说明。

---

## 5. 模型

- **架构**：`get_resnet(size=10)` → `ResNet(BasicBlock, [1,1,1,1])`（`resnet.py:137-151`）。`main.py:89-103` **硬编码** size=10，`--model` / `--model_size` 参数**未被使用**。
- **倒数第二层特征**：`resnet.py:97-107` 的 `forward` 里，`avg_pool2d(out,4)` + `view` 之后、`self.linear` 之前，已被存进 `self.feature`（`resnet.py:105`），并有 `extract_feature()`（`resnet.py:109-110`）。
  - 维度 **d = 512**（`512 × BasicBlock.expansion(=1)`）。
  - 我的实现会用 **`model.linear` 的 forward pre-hook 取输入**，而不是依赖 `self.feature`：这样与架构解耦，也避免依赖"上一次 forward 的残留状态"。
- **分类头**：`model.linear.weight` → `[10, 512]`，`model.linear.bias` → `[10]`（`resnet.py:87`）。`margin_matrix` 直接取这两个。
- ⚠️ `model.device` 是 `main.py:98` / `main.py:105` **外部注入**的属性，不是 `nn.Module` 的原生属性，但 `evaluate_accuracy`（`utils.py:30`）依赖它。诊断代码加载 checkpoint 后必须手动补上。

### per-client checkpoint 机制
**完全没有。** 全仓库无 `torch.save`（已 grep 确认）。`upload_model` 返回的是 `state_dict()` 的**引用**（`client.py:28`），且 `agg_avg`（`server.py:4-10`）会**原地修改 `state_dicts[0]`** —— 埋点保存时必须 `copy.deepcopy` 或 `.clone()`，否则存下来的是被聚合污染的张量。

**最小侵入加法**：注册 `fl_event_emitter.on("on_fl_end", save_all)`，在训练结束时遍历 `clients` 存 `local_model.state_dict()` + 存 `server.global_model` + 存生成器。**对 `fl_process.py` / `client.py` / `server.py` 零改动。**

---

## 6. PFL 方法支持情况

| 方法 | 状态 |
|---|---|
| **FedBN** | ✅ 唯一实现（`pfl.py:3-24`）。聚合与下发时删除所有含 `"bn"` 或 `"shortcut.1"` 的键 → BN 参数与统计量永久本地化 |
| Ditto | ❌ **未实现** |
| FedProx | ❌ **未实现** |
| 其他 | ❌ |

⚠️ `client.py:86-125` 的 `PMClient` / `PMPoisonClient`（双模型个性化，可支撑 Ditto 式方法）**已定义但在 `main.py` 中从未被实例化**。任务书提到的"论文里的 Ditto、FedProx，R=0.1"**在这份开源代码里不存在**。若诊断需要覆盖多种 PFL 方法，需要自行补实现 —— 这超出当前任务范围，先记录。

## 6.1 其他"解析了但没用"的参数

`--model`、`--model_size`、`--dataset`、`--client_dist`、`--ba_trigger_position` 完全未被引用（已 grep 确认）。`--agg_rule` 只是赋给 `server.agg_rule`（`main.py:106`），但 `agg_and_update`（`server.py:29-34`）**恒定调用 `agg_avg`**，从不读该字段。`trigger.py` 的 `grid_trigger_adder` 在 `main.py:11` 被 import 但从未调用。

---

## 7. 环境

### 7.1 🛑 当前容器状态（阻塞）

| 项 | 状态 |
|---|---|
| GPU | **无**（`nvidia-smi` not found，CPU-only 容器） |
| CPU / 内存 | 4 核 / 15 GB |
| 磁盘 | 30 GB 可用 |
| PyTorch | **未安装** |
| torchvision / pandas / scipy / matplotlib / sklearn / yaml | **未安装** |
| CIFAR-10 数据 | **不存在**（`./data` 目录没有，而 `main.py:59-60` 是 `download=False`） |
| 仓库 requirements 文件 | **没有**。README 只说"请先安装必要的包"，无版本约束 |

**依赖安装尝试**：`download.pytorch.org` 被出口代理按组织策略拒绝（403）。PyPI (`pypi.org` / `files.pythonhosted.org`) 在代理白名单内，**可以走 PyPI 安装**，但该命令被你中止了，所以**尚未安装**。

→ **Phase 2 的冒烟测试需要 CPU 版 torch。在你批准安装之前无法执行。** 详见下方"待确认事项"。

### 7.2 预判的坑（**未经验证**，因为还没装上 torch）

1. **AMP 在 CPU 上**：`BasicClient.__init__`（`client.py:15`）无条件构造 `torch.cuda.amp.GradScaler()`；`enable_mix_precision`（`utils.py:16-20`）用 `torch.cuda.amp.autocast()` 包裹 forward。CPU-only 环境下这两者通常只是**告警并降级为 no-op**，但新版 torch 已把 `torch.cuda.amp.*` 标记为 deprecated。冒烟测试可能需要在 `diag/` 里做兼容处理（**不改原文件**）。
2. **`torch.cuda.empty_cache()`** 散布在 `client.py` / `fba.py` / `utils.py`：CUDA 未初始化时是 no-op，应无害。
3. **`drop_last=True` 用在测试集上**（`main.py:83`）：每客户端 100 个测试样本 → 实际只有 3 个 batch = **96 个可用样本**。
   ⚠️ 这对诊断是个**实质性问题**：`config.yaml` 里 `min_target_samples=30`、`knn_k=20`，而本地测试集只有 96 个样本、目标类样本可能个位数。**特征提取很可能必须改用本地训练集（500 样本）或一个更大的公共评估集**。这是需要你决策的第二个设计问题。
4. **无数据归一化**（`main.py:58` 只有 `ToTensor()`）：像素 ∈ [0,1]，所以 `pgd_attack` 的 `clamp(0,1)` 与 ε=4/255 的口径是自洽的。`perturb.py` 沿用 [0,1] clamp。
5. **训练开销**：300 轮 × 10 客户端 × 15 步 ≈ 45,000 次 batch-32 前后向；恶意客户端每次被选中还要额外做 30 次生成器迭代（每次含一次 PGD）。**在 4 核 CPU 上跑完整配置不现实**（粗估单次 run 数小时到十几小时，24 组扫描完全不可行）。正式实验需要 GPU。

### 7.3 复现默认配置的 ACC / ASR

**未执行。** 原因：无 torch、无数据、无 GPU，且本次任务明确要求"不要启动任何正式训练"。因此本文件**不含任何 ACC/ASR 数字**，也无法与论文报告值对比。

---

## 8. Phase 0 必答清单 —— 完成状态

| 条目 | 状态 |
|---|---|
| `main.py` 完整流程 + 行号 | ✅ §2.1 |
| 训练循环入口与调用栈 | ✅ §2.2 |
| 数据划分位置 / 方式 / Dirichlet / 异构度参数 | ✅ §3.1（只有 Dirichlet，参数 `--dir_alpha`，默认 0.5） |
| 客户端数 / 每轮采样 / 恶意数 配置位置 | ✅ §3.2 |
| 随机种子设置位置 / 划分是否受攻击开关影响 | ✅ §3.3（**发现重大问题**） |
| 生成器定义 / 输入输出 / 保存加载 | ✅ §4.1（保存加载：**当前没有**，需新增） |
| **δ / ξ 是哪个变量、能否分离** | ✅ §4.2（**可分离**，但有 5 条重要性质） |
| 目标类设定与默认值 | ✅ §3.2（`--ba_target_label`，默认 0） |
| 投毒比例参数 | ✅ §3.2（`--ba_poison_rate`，默认 0.2） |
| ASR 计算口径 | ✅ §4.3（**不排除目标类原样本**） |
| 模型架构 / 倒数第二层特征 / 分类头访问 | ✅ §5 |
| per-client checkpoint 机制 | ✅ §5（**没有**，最小侵入方案已给出） |
| 支持哪些 PFL 方法 | ✅ §6（**只有 FedBN**；无 Ditto/FedProx） |
| 依赖安装 / 版本 / 坑 | ⚠️ **部分**。§7.1 已确认环境现状与代理限制；**版本与实际运行的坑尚未验证**，因为 torch 未安装 |
| 训练耗时 / 显存 | ❌ **未确认**。无 GPU 可测；CPU 估算见 §7.2.5，**仅为粗估，未实测** |
| 默认配置的 ACC / ASR 复现 | ❌ **未执行**，见 §7.3 |

---

## 9. 🛑 需要你确认后才能进入 Phase 1 的事项

1. **是否批准从 PyPI 安装 CPU 版依赖**（`torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `pyyaml`）？
   不装则 Phase 2 的冒烟测试无法运行，我只能交付未经执行验证的代码。
   （注：PyPI 上 Linux 的 `torch` 默认 wheel 捆绑 CUDA 运行时，约 2–3 GB；磁盘够用，但如果你希望更省，可以只装 `torch` 而跳过 `scikit-learn` 等，我会用 numpy/scipy 自行实现 AUC。）

2. **ξ 的生成模型口径**（§4.2b）：方案 A（每客户端用自己的模型）还是方案 B（固定参考模型）？
   我的建议：**B 为主、A 为附加对照**。这决定 `perturb.py` 的 `xi_fn` 签名。

3. **特征提取用哪个数据集**（§7.2.3）：本地测试集只有 96 个可用样本，撑不起 `knn_k=20` + `min_target_samples=30`。
   我的建议：**特征提取改用本地训练集（500 样本/客户端）**，并在文档中标注这不是 held-out 数据；或者额外划一个全局共享的评估集。需要你选。

4. **是否接受为对照实验修复随机性**（§3.3）：补 `np.random.seed` / `random.seed`，并给 `random_select` 独立 RNG。
   这会改变原实现的具体采样序列（不改变分布与攻击语义），但**不修复就做不了 clean/attack 对照**。会逐条记进 `PATCHES.md`。

5. **`delta.pt` 的语义**（§4.2a）：确认接受"只存生成器权重，δ 按需现算"，`delta.pt` 降级为不参与计算的存档产物。

---

## 10. 尚未确认 / 存疑的条目（诚实清单）

- 训练一次完整实验的**实际**耗时与显存占用 —— 无 GPU、未实测，§7.2.5 只是数量级估算。
- 默认配置能否复现论文的 ACC/ASR —— 未运行。
- AMP (`torch.cuda.amp.*`) 在当前 torch 版本 + CPU 下**究竟是告警还是报错** —— 未验证。
- CIFAR-10 能否从容器内下载（`torchvision` 的下载源是否在代理白名单内）—— 未验证。
- `client_inner_dirichlet_partition` 在极小 α（如 0.05）下的实际类别分布形态 —— 未实测；算法中"类别耗尽则随机改派"的逻辑可能让实际异构度**低于**名义 α，这会直接影响实验 A 的 x 轴含义，**建议 Phase 1 加一个划分统计的 sanity check**（`utils.py:88-116` 已有现成的 `partition_report` 可用）。
