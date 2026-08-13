# diag/audit — P0 排查工具

一次性排查工具，用于判定实验 C / D 已有结果里**哪些能信、哪些不能信**。
产出 `P0_AUDIT.md` + `evidence/*.csv`（每张表都可独立复核）。

## 在集群上怎么跑

```bash
cd <Bad-PFL 仓库根目录>

python -m diag.audit.run_audit \
    --ckpt-root ./checkpoints \
    --raw-dir   ./results/raw \
    --out       diag/audit/P0_AUDIT.md \
    --leak-experiment
```

`--leak-experiment` 会额外跑一次**受控 A/B 重训**（默认 20 客户端 / 30 轮），
用来区分「实现缺陷」与「攻击本来就不隐蔽」。不加这个 flag，任务 1 的**成因**
判定会停在「未能确定」。

其它常用开关：

| 开关 | 用途 |
|---|---|
| `--meta-only` | 只跑不需要模型权重的检查，几秒出结果，适合先看一眼 |
| `--alphas 0.05 0.1` | 只审计指定的 α |
| `--seeds 0` | 只审计指定的 seed |
| `--leak-synthetic` | 受控对照用合成数据（不需要 torchvision） |
| `--device cuda:0` | 指定设备 |

单独跑受控对照：

```bash
python -m diag.audit.leak_experiment --client-num 20 --total-round 30
```

## 输出

```
diag/audit/
    P0_AUDIT.md          # 报告：结论速查表 / 详细证据 / 修正后结果 / 代码改动 / 已知限制
    evidence/*.csv       # 全部中间量（19 张表）
```

## 判定是怎么产生的

所有阈值集中在 `common.py` 的 `THRESHOLDS`，报告会同时打印**规则**和**实测值**。
规则不满足时输出「未能确定」，不猜、不含糊。想改判定标准就改那一处，
不要在报告里手工改措辞。

## 内存

审计要遍历全部客户端模型，因此一律用 `cache_models=False`
（`RunBundle` 的无界缓存在 100 客户端下约 2GB/bundle）。模型用完即弃。

## 注意

- **不修改原仓库文件**。去泄漏对照臂靠 `leak_free_poison_client()` 这个
  上下文管理器 monkey-patch，退出即还原（有测试保证，含异常路径）。
- 加载 checkpoint 时会调 `assert_partition_matches()` 校验**重建出的数据集
  与训练时是同一份**。尺寸相同但内容不同是静默错误，这个守卫会直接中止审计。
- 本目录的代码在**合成数据夹具**上端到端验证过（12 客户端 / 2 个 α），
  但**没有在真实 CIFAR-10 结果上跑过** —— 真实数据上若出现未预料的边界情况，
  脚本按「未能确定」处理，不会猜。
