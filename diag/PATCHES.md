# PATCHES.md — 对原仓库的改动记录

## 摘要

**对原仓库文件的改动为零。**

```
$ git diff --stat main -- . ':(exclude)diag'
(空)
```

`main.py` / `client.py` / `server.py` / `fba.py` / `fl_process.py` / `pfl.py` /
`utils.py` / `generator.py` / `trigger.py` / `resnet.py` / `mobilenet.py` /
`densenet.py` / `event_emitter.py` 全部保持原样，随时可 diff。

诊断所需的全部行为差异都在 `diag/` 里以**旁路实现**的方式完成，而不是修改原文件。
下面逐条列出这些差异 —— 因为**语义上**它们等价于对原实现打补丁，即使字面上没有改动原文件。

---

## 一、旁路实现的差异（等价于补丁）

### P1. 补齐随机性播种

| 项 | 内容 |
|---|---|
| **对应原实现** | `utils.py:8-13` `set_random_seed` |
| **诊断侧实现** | `diag/config.py::set_all_seeds` |
| **原实现做了什么** | 只播种 `torch.manual_seed` / `torch.cuda.manual_seed_all`，并设置 cudnn 确定性 |
| **问题** | **没有播种 numpy，也没有播种 python `random`**。而数据划分完全依赖 `np.random`（`main.py:67` 的 `np.random.dirichlet`，`utils.py:65/71/74` 的 `randint`/`uniform`），客户端顺序依赖 `random.shuffle`（`main.py:95`）。后果是**同一条命令跑两次得到不同的数据划分** |
| **诊断侧做法** | 额外调用 `random.seed(seed)` 与 `np.random.seed(seed)` |
| **是否影响训练动力学** | **是，但这是必需的修复**。它改变了具体划分出来的分片内容（因为随机流不同了），但不改变划分的**分布**（仍是同参数的 Dirichlet），也不触碰任何攻击逻辑。**不做这个修复就无法构造 clean/attack 对照组** |
| **用户确认** | 已确认（"接受全部修复"） |

### P2. 隔离客户端采样的随机流

| 项 | 内容 |
|---|---|
| **对应原实现** | `utils.py:23-26` `random_select`（用全局 torch RNG 的 `torch.randperm`） |
| **诊断侧实现** | `diag/config.py::make_select_rule`（独立的 `torch.Generator`） |
| **接入方式** | `basic_fl_process` 的 `select_rule` 本来就是参数（`fl_process.py:4,11`），直接传入即可，**无需改动 `utils.py`** |
| **问题** | 攻击 run 在训练过程中会额外消耗全局 torch RNG：`Autoencoder()` 初始化（`fba.py:27`）、每次 `pgd_attack` 的 `uniform_` 随机起点（`fba.py:8`）、每轮 30 次生成器迭代（`fba.py:33`）。而客户端采样共用同一条全局流，导致 **clean run 与 attack run 每轮选中的客户端集合完全不同** —— 两组"每个客户端被训练了多少次"对不上 |
| **是否影响训练动力学** | **是**。采样序列与原实现不同（分布相同，均匀无放回）。攻击语义、投毒函数、聚合规则均不受影响 |
| **可验证性** | `meta.json` 记录 `n_participations`；`verify_partition_consistency` 会在两次 run 的参与轮次不同时给出提示 |
| **用户确认** | 已确认（"接受全部修复"） |

### P3. 评估路径改用 `drop_last=False`

| 项 | 内容 |
|---|---|
| **对应原实现** | `main.py:82-84`（测试 loader 用了 `drop_last=True`） |
| **诊断侧实现** | `diag/probe.py::make_eval_loader`（`drop_last=False`, `shuffle=False`） |
| **问题** | `drop_last=True` 是训练时的约定（保证 batch 形状一致、BN 统计稳定），泄漏到了评估路径。每个客户端静默丢掉最多 `batch_size - 1` 个样本；本地测试集只有 100 个样本时实际只剩 96。而且丢弃比例随客户端样本数变化，在高异构下会给指标注入**系统性偏差** |
| **是否影响训练动力学** | **否**。只影响评估，训练 loader 仍保留 `drop_last=True`（见 `diag/run_fl.py` 中的注释） |
| **回归测试** | `diag/tests/test_eval_loader.py`（专门盯这个 bug） |

### P4. 生成 ξ 时把模型置于 `eval()` 模式

| 项 | 内容 |
|---|---|
| **对应原实现** | 投毒路径中模型处于 `train()`：`client.py:35` 的 `self.local_model.train()` 先于 `client.py:36` 的 `self.fetch_data()`，而 `fetch_data` 会触发 `our_poison_func` → `pgd_attack` |
| **诊断侧实现** | `diag/perturb.py::make_xi_fn`（`use_eval_mode=True`，默认开启） |
| **理由** | 诊断必须在 `eval()` 下前向，否则 BatchNorm 用 batch 统计量并**更新 running stats**，污染所有下游特征、原型、边距。这是任务 spec 明确列出的陷阱 |
| **代价** | 诊断时算出的 ξ 与训练时刻的 ξ **不完全相同**（BN 行为不同）。这是刻意的取舍：eval 是可复现的、无副作用的 |
| **是否影响训练动力学** | **否**。只在离线诊断路径调用，训练时完全不经过这段代码 |
| **可关闭** | `make_xi_fn(..., use_eval_mode=False)` 可切回 train 模式，供敏感性检查 |

### P5. 固定 ξ 的随机起点

| 项 | 内容 |
|---|---|
| **对应原实现** | `fba.py:8` `torch.zeros_like(images).uniform_(-epsilon, epsilon)`，使用全局 RNG，未固定 |
| **诊断侧实现** | `diag/perturb.py::_temporary_torch_seed`：调用 `pgd_attack` 前保存全局 RNG 状态、设定 `seed + call_index`，调用后恢复 |
| **理由** | 同一输入多次调用会得到不同的 ξ，指标不可复现 |
| **关键点** | **`pgd_attack` 本身一行未改**，是直接 `import fba` 后调用原函数。只控制它读取的随机流 |
| **是否影响训练动力学** | **否**。RNG 状态在调用前后被完整保存/恢复，不扰乱调用方所在的随机流（有 `test_probe.py::test_build_probe_set_does_not_disturb_global_numpy_rng` 的同类保证） |

### P6. 调用 `pgd_attack` 后清零模型梯度

| 项 | 内容 |
|---|---|
| **对应原实现** | `fba.py:16` 的 `loss.backward()` 会把梯度**累加进模型参数**。在 `local_update` 中 `optimizer.zero_grad()`（`client.py:34`）发生在 `fetch_data()`（`client.py:36`）**之前**，所以 PGD 的梯度会混入本轮参数更新 |
| **诊断侧实现** | `diag/perturb.py::make_xi_fn` 在 `finally` 中 `model.zero_grad(set_to_none=True)` |
| **理由** | 诊断只做读操作，不应在加载的模型上留下副作用 |
| **是否影响训练动力学** | **否**。原实现的这一行为**被完整保留**（我没有改 `fba.py` 也没有改 `client.py`），训练时该怎样还是怎样。清零只发生在离线诊断路径 |
| **回归测试** | `test_perturb.py::test_xi_fn_does_not_leave_gradients_on_model` |

---

## 二、只读埋点（不改变任何行为）

> **命名说明**：本节用「埋点 1–4」编号。
> 不要与 `H1/H2/H3` 混淆 —— 那三个是**科学假设**的编号
> （H1 = δ 的跨环境不变性，H2 = ξ 的散度痕迹，H3 = 可观测性），
> 定义在任务书里，出现在 `exp_a.py` / `exp_c.py` / `exp_d.py` 的 docstring 中。
> 早先本节误用了 H1–H4，造成过一次实际的误读。

### 埋点 1. checkpoint 与元数据保存

- **接入点**：`event_emitter.fl_event_emitter` 的 `on_fl_end` 事件。
- **为什么零侵入**：`fl_process.basic_fl_process` 本来就发射了 6 个事件
  （`fl_process.py:6,14,22,32,38,41`），但原仓库**没有注册任何监听器**。
  `diag/hooks.py::attach_save_hook` 只是往这个既有总线上挂了个 handler。
- **保存内容**：全局模型、每个客户端的本地模型、生成器、`meta.json`。
- **注意事项（已修正措辞）**：`BasicClient.upload_model`（`client.py:28`）把
  `state_dict()` 的返回值存到 `self.upload_state_dict`，而 `server.agg_avg`
  （`server.py:4-10`）的 `average_dict = state_dicts[0]` 是**引用**。
  但第 8 行 `average_dict[key] = average_dict[key] + state_dicts[idx][key]`
  是**重新绑定字典条目**（创建新张量），**不是张量级的原地写** ——
  所以客户端 `local_model` 的参数并不会被污染，被污染的是
  `client.upload_state_dict` 这个属性所持的 dict。

  结论不变：任何读 `upload_state_dict` 的代码必须自己深拷贝。
  本工具包一律从 `module.state_dict()` 现取，`save_state_dict` 再
  `detach().cpu().clone()`，两种情况都安全
  （回归测试 `test_hooks.py::test_save_state_dict_is_detached_cpu_copy`）。

### 埋点 2. 参与轮次计数

- **接入点**：`on_client_begin` 事件（`fl_process.py:22`）。
- **行为**：只对客户端对象上一个自定义属性 `diag_n_participations` 做自增。
- **训练结束后注销**（`diag/run_fl.py` 的 `finally` 块），避免跨 run 污染。

### 埋点 3. 生成器提取

- **实现**：`diag/hooks.py::extract_generator`。
- **做法**：`fba.use_our_attack`（`fba.py:25-66`）只返回 `eval_func`，把
  `trigger_gen` 留在闭包里。这里通过 `__closure__` / `co_freevars` **只读**取出。
- **失败时的行为**：若原仓库将来重命名了这个自由变量，会抛出带实际变量名的
  `RuntimeError`，便于定位 —— 刻意不做静默回退
  （回归测试 `test_hooks.py::test_extract_generator_reports_actual_freevars_on_failure`）。

### 埋点 4. 客户端自定义属性

在 `diag/run_fl.py` 里给客户端对象附加了以下属性，全部以 `diag_` 或明确语义命名，
**不与原仓库的任何属性重名**：

| 属性 | 含义 |
|---|---|
| `partition_idx` | 该客户端拿到的原始数据分片编号（`cid` 是 shuffle 之后才赋的，两者不同） |
| `diag_is_malicious` | 本次 run 中是否**实际**投毒（clean run 恒为 False） |
| `diag_is_malicious_slot` | 是否占据"恶意槽位"（分片 90..99），两种模式下一致，用于跨模式配对 |
| `diag_poison_ratio` | 实际投毒比例 |
| `diag_n_participations` | 被选中参与训练的轮次数 |

### 埋点 5. 按轮次的快照（实验 F）

- **接入点**：`on_client_end`（`fl_process.py:32`）与 `on_round_end`
  （`fl_process.py:38`），同样是既有事件，**原仓库文件仍为零 diff**。
- **实现**：`diag/snapshots.py::SnapshotRecorder`，由
  `diag/run_fl.py --snapshot-every N` 启用。
- **保存内容**：每 N 轮把全局模型、抽样客户端的本地模型、生成器存到
  `{ckpt_dir}/round_XXXX/`，另写 `snapshot_manifest.json`。
- **为什么需要**：实验 F 要构造 (生成器轮次 s, 模型轮次 t) 的矩阵。
  生成器侧的按轮次快照早已有（`attach_generator_checkpoint_hook`），
  **模型侧从来没有实现过** —— `save_run` 只在 `run_fl.py` 末尾调用一次，
  官方仓库里 `torch.save` 一次都没出现。
- **网格对齐的参与式快照**：FedBN 下客户端只在被选中的轮次才更新
  `local_model`（100 个客户端每轮选 10 个）。若在网格轮次无差别地保存，
  拿到的多半是陈旧权重且陈旧程度不被记录。因此改为"网格点之后的**首次参与**
  时保存"，并把 `staleness = 实际轮次 − 网格轮次` 逐份写进 manifest，
  作为分析中的协变量。缺失的网格点写进 `missing` 字段，
  **不补齐、不插值**。
- **是否影响训练动力学**：**否**。埋点只做 I/O —— 不消耗任何 RNG、
  不触碰 optimizer、不做前向。这一点不是口头保证：
  `run_fl.py --verify-against <旧 run 目录>` 会在训练结束后逐位比对
  `global.pt` / `generator.pt` / `client_*.pt` 的参数哈希
  （`hooks.compare_run_checkpoints`）。
- **回归测试**：`test_exp_f.py` 的
  `test_snapshot_recorder_aligns_to_grid_and_records_staleness` /
  `test_snapshot_recorder_catches_up_multiple_grid_points_at_once` /
  `test_snapshot_recorder_reports_missing_without_filling_them_in`。

### 埋点 6. 服务器端聚合规则的替换（实验 I / J）

- **接入点**：`diag/defenses.py::use_defense` 用 `types.MethodType` 绑定一个新的
  `agg_and_update` 到 **server 实例**上。`BasicServer` 类本身不受影响，
  `server.py` 仍为零 diff。
- **为什么不能挂钩子**：`before_update_global` 在 `agg_avg` **之后**才触发
  （`server.py:32-33`），拿不到聚合前的各客户端更新，无法替换聚合规则本身。
- **FedBN 行为一位不变**：新的 `agg_and_update` 仍然把 FedBN 私有 key 与非浮点
  key 以简单平均放进 `server.update`，再调 `call_registered_func(
  "before_update_global")`、再 `load_state_dict(..., strict=False)` ——
  与 `server.py:32-34` 的顺序完全一致，`fedbn_update` 照常 pop。
- **`--defense fedavg` 完全不接管**，走原仓库的 `agg_avg`。这样"无防御"这一组
  与既有的实验 A–F 结果**严格同源**，不会因为换了一份数值等价的实现而产生
  细微差别（等价性另有 `test_fedavg_matches_repo_agg_avg` 保证）。
- **不改动客户端侧任何代码**，攻击保持原样（实验 I §2.3 / 实验 J 陷阱 13）。
- **回归测试**：`test_defenses.py`（19 个），重点是
  `test_defenses_do_not_mutate_incoming_state_dicts`（`agg_avg` 别名 bug）与
  `test_every_defense_leaves_fedbn_keys_in_the_update_dict`。

### 埋点 7. 训练期的逐轮记录与周期评估（实验 I §5 / 实验 J §2、§5）

- **接入点**：`on_round_begin`（取本轮选中的客户端）、`on_round_end`（周期评估），
  以及 `use_defense` 的聚合回调（拿到与聚合器同一份 `w_prev`）。
- **只写标量**：10 客户端 × 1000 轮 × ResNet-10 的更新向量约 200GB，
  因此每轮只落盘 `round_{t:04d}.npz` 里的标量与长度为 N 的数组。
- **全局模型必须借 BN**：FedBN 下 `server.global_model` 的 BN 从未更新，
  直接评估得到的是失配模型。`acc_global` / `asr_global_*` 一律在
  "全局共享参数 + 一个固定良性客户端的 BN"上算，**另存** `acc_global_raw`
  把退化程度显式暴露，而不是藏起来。
- **是否影响训练动力学**：**不影响 —— 但第一版实现是影响的，靠实测才发现。**

  第一版按上面这套道理写完后，同一 seed 下 `--eval-every 1` 与 `--eval-every 0`
  的 checkpoint 比对结果是：`client_{恶意}.pt` / `generator.pt` / `global.pt`
  **三个哈希不同**，而三个良性客户端**完全一致**。

  成因：`_maybe_evaluate` 为了构造"借 BN 的全局模型"要 `get_resnet()` 新建模型，
  **权重初始化消耗全局 torch RNG**。良性客户端看不出变化，是因为它们的
  DataLoader 迭代器在 `BasicClient.__init__` 时就建好了；而恶意客户端的投毒掩码
  （`fba.py:49` 的 `torch.rand`）、PGD 随机起点（`fba.py:8`）与生成器训练
  都在训练时现取 RNG，于是被整体推移。

  **这种局部差异极易被误读成"防御生效了"** —— 恰恰是恶意客户端和生成器变了。

  修复：`diag/track.py::preserve_rng_state` 把**整个评估过程**包起来，
  保存/恢复 torch（含 CUDA）/ numpy / random 三条流。不是只包新建模型那几行 ——
  评估路径上任何一处碰随机流都会被吸收掉。修复后同一比对为
  **6/6 一致**。回归测试 `test_preserve_rng_state_restores_all_three_streams`。

  同一类问题在 FLAME 的加噪上也存在：`torch.empty_like().normal_()` 若走全局
  RNG 会每轮推进随机流。`Flame` 因此**始终**用独立的 `torch.Generator`
  （`seed + round * 7919`），有 `test_flame_noise_never_touches_the_global_rng`
  盯着。

### 埋点 8. 设备一致性（GPU 上才会暴露的一类 bug）

第一版在集群 GPU 上第一轮聚合就崩：
`RuntimeError: Expected all tensors to be on the same device, but found at
least two devices, cuda:0 and cpu!`

成因是同一族的 **11 处**：所有 float64 累加器都用
`torch.zeros(..., dtype=torch.float64)` 建在**默认设备（CPU）**上，
而逐 key 的中间张量跟随模型在 **GPU** 上。开发容器只有 CPU，
所以整套单元测试与冒烟一个都没抓到。

统一后的约定：

| 位置 | 处理 |
|---|---|
| `instrumentation.rank_window_fraction` | 计数器建在 `flat.device` —— 它要和 `bincount` 的输出直接相加 |
| `instrumentation.round_signals` | 累加器留在 **CPU**，每个 key 只把长度为 N 的归约结果 `.cpu()` 搬回来 |
| `defenses.gram_matrix` | 同上，每个 key 只搬回 N×N |
| `defenses.Median` / `InvariantAggregator` | 影响力累加器留 CPU，逐 key `.cpu()` |
| `defenses.Flame` | `weights` 来自 CPU 上的 gram，乘之前 `.to(stacked.device)`；噪声在 CPU 上按独立 `Generator` 生成后再搬过去（**CPU 的 Generator 不能直接用于 CUDA 张量的 `normal_()`**） |

精度上改用 `sum(dtype=torch.float64)` / `mean(dtype=torch.float64)`
在 float32 数据上做 float64 累加，避免把 `[N, D]` 整个转成 float64
（ResNet-10 上约 400MB）。

**回归测试**：`test_all_defenses_run_on_cuda_when_available`。它在无 GPU 的机器上
**空转并打印"跳过"** —— 所以提交长作业前必须在计算节点上单独跑一次，
确认它是 PASS 而不是跳过。`meta` 设备无法替代：`meta + cpu` 不报错，
且 `bincount` 在 meta 上未实现。

### 埋点 9. "静默"恶意客户端（`oracle_exclude`）

- **接入点**：与其余防御同一个（`use_defense` 换掉 `server.agg_and_update`），
  没有额外的 hook。
- **它读了一个现实中拿不到的标签**：掩码来自
  `[getattr(clients[i], "diag_is_malicious", False) for i in tracker.selected_indices]`,
  按**本轮上传顺序**对齐（长度不一致直接报错，不做静默截断）。因此它是
  **上界对照，不是防御**。
- **`inner` 只作用在留下的客户端上**：`OracleExclude` 把幸存者的子列表交给内层规则，
  再把 `influence` / `trim_survival_rate` **按原索引散射回长度 N 的数组**，
  被排除的位置填 0。若不散射回去，并表时 `is_malicious` 的下标就会错位。
- **全排除会报错而不是返回上一轮的权重**：`bad_client_num` 大到某轮抽中的全是恶意
  客户端时抛 `RuntimeError`。静默地"不更新"会让曲线看起来只是收敛慢一点。
- **良性客户端的 ASR 是负对照，不是结果。** 聚合是恶意→良性的唯一通道，被切断后
  良性 ASR 必须留在基线；涨了就说明有计划外的泄漏（= bug）。这个失败模式在图上
  和"防御生效"长得一样，所以它值得单独有一组。
- **恶意客户端自己的模型另记一套列**：`acc_malicious_own` / `asr_malicious_own` /
  `n_eval_malicious`（`track.py::eval_malicious_ids`）。它必然很高——个性化模型
  就是在投毒数据上练出来的——**绝不能混进良性均值**，否则整组数字失去意义。
  只有 `--eval-include-malicious` 时才记录。

---

## 三、`diag/run_fl.py` 与 `main.py` 的关系

`main.py` 是 `if __name__ == "__main__"` 脚本，无法作为模块复用。
`diag/run_fl.py` **复刻**了它的构造流程（argparse 默认值逐项对齐），
以便原文件保持零 diff。

**已知风险：两边会漂移。** 若将来有人改了 `main.py` 的构造逻辑，
`run_fl.py` 不会自动跟进。逐项对照关系：

| main.py | diag/run_fl.py | 是否一致 |
|---|---|---|
| `:58` `ToTensor()` only，无归一化/增强 | `build_datasets` | ✅ |
| `:65-66` 每客户端 500 / 100 样本 | `train_per_client` / `test_per_client` | ✅ |
| `:67` `class_priors` train/test 共用 | 同 | ✅ |
| `:68-77` Dirichlet 划分 | 调用同一个 `client_inner_dirichlet_partition` | ✅ |
| `:78-84` loader，`drop_last=True` | 训练 loader 相同 | ✅ |
| `:89-94` 前 90 个 Basic / 后 10 个 Poison | 同（clean 模式下全为 Basic） | ✅ 见下 |
| `:95-99` `shuffle` 后赋 `cid` | 同 | ✅（现在可复现） |
| `:103-106` 全局 resnet10 + `agg_rule` | 同 | ✅ |
| `:110-111` FedBN | 同 | ✅ |
| `:116-117` `use_our_attack` | 同（仅 attack 模式） | ✅ |
| `:122-123` `basic_fl_process` | 同，但 `select_rule` 换成隔离 RNG 的版本 | ⚠️ 见 P2 |
| `:127-138` 最终评估 ACC/ASR | **未复刻** | ⚠️ 见下 |

两点补充说明：

1. **clean 模式下恶意槽位仍创建 `BasicClient`**，而不是创建 `PoisonClient` 再关掉投毒。
   这样保证两种模式在训练开始前消耗的 torch RNG **完全相同**
   （两个分支都各自创建一个 resnet10）。
2. **`main.py:127-138` 的 ACC/ASR 评估没有复刻**。原因是它的 ASR 口径有两个问题
   （见 `REPO_MAP.md` §4.3）：不排除目标类原样本（`fba.py:52` 把全部标签改成
   `target_label`），且 ξ 由循环里**最后一个**恶意客户端的模型生成（`fba.py:64`）。
   诊断侧在 `diag/metrics.py::excess_response` 里用 `probe.query`
   （全部为非目标类样本）重新定义了 ASR。**两个口径的数字不可直接比较。**

---

## 四、新增依赖

无。用到的 `torch` / `numpy` / `pandas` / `scipy` / `matplotlib` / `pyyaml`
在当前环境中均已存在。

- `pytest` **未安装**，因此提供了 `diag/tests/run_tests.py` 作为后备运行器。
  测试文件本身是标准 pytest 风格，装了 pytest 的环境可直接 `pytest diag/tests`。
- `torchvision` **未安装**，因此冒烟测试全程使用
  `diag/run_fl.py::SyntheticImageDataset` 合成数据。**正式实验需要 torchvision
  来加载 CIFAR-10。**

### 埋点 10. 攻击调度与正式实验 1 的逐轮指标

- **接入点**：`schedule.gate_attack` 包装 `client.poison_func`
  （`client.py:124` 调用它）与 `registered_funcs["before_local_training"]` 里的
  `trigger_gen_trainer`（`fba.py:63` 注册）。都是 diag 侧对客户端**对象**的
  改写，原仓库文件不动，训练结束后 `restore()` 还原。
- **`continuous` 不装任何包装器**：默认设定与本机制出现之前逐位一致。
  有测试 `test_continuous_installs_no_wrapper_at_all` 盯着。
- **调度之间的 RNG 流无法对齐**。一旦某一轮的训练数据不同，之后所有模型、
  客户端采样、PGD 起点都不同。这是原理上的，不是实现缺陷。因此本模块
  **不**为"关闭"的轮次空转消耗 RNG —— 那样既费一次 PGD 前后向，又只能对齐到
  第一次分歧为止。**调度比较必须靠多 seed。**
- **一轮只判定一次**（`active_for_round`）。`after_mta` 会锁存，所以"什么时候
  问"会改变答案：轮首问（MTA 还是上一轮的）与轮尾问（MTA 已更新）可能不同。
  不缓存的话，本轮实际有没有投毒、与落盘的 `attack_active_this_round`
  就可能差一轮 —— 而那一轮正是 1B 最关心的。

顺带修掉两个**写死 `size=10`** 的地方，它们在 `--model-size 18` 下会静默出错：

| 位置 | 症状 |
|---|---|
| `track.py::_evaluate_now` | 用 ResNet-10 的空壳去 `load_state_dict` ResNet-18 的权重 |
| `exp_f.py::load_snapshot_model` | 同上，实验 F 与离线表征分析都会踩到 |

两处都改成从 `meta.json` / config 读 `model_size`。

### 埋点 11. Gram 原语的去重

`gram_matrix` / `pairwise_cosine` / `pairwise_distance` / `pseudo_grad_stack`
原本在 `defenses.py` 里各有一份，而 `instrumentation.py` 也需要它们来算
分组余弦。两处各写一份必然漂移，**而漂移是静默的**：FLAME 的簇与 Multi-Krum
的分数会基于不同的距离，两张表看起来都对，并到一起才是错的。

现在只在 `instrumentation.py` 定义一份，`defenses.py` 用别名 import
保留旧的私有名，本文件其余部分不改。
