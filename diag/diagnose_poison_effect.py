#!/usr/bin/env python3
"""诊断投毒率 ρ 对 ASR 影响的根本原因。

输出三个关键检查：
1. 数据完整性：每个 (Nm, ρ) 点的样本量是否一致
2. 主任务性能：MTA 在高 ρ 时是否退化（虚假ASR下降的信号）
3. 稳定性：ASR 方差在高 ρ 时是否异常（后门训练不稳定的信号）
"""

import glob
import pandas as pd
import numpy as np
from pathlib import Path

# 加载所有 Exp1 implantation CSV（排除 calib / edge）
paths = sorted([
    p for p in glob.glob("results/raw/exp_ij_implantation_*_e1*.csv")
    if 'calib' not in p and 'edge' not in p
])

if not paths:
    print("❌ 没有找到 Exp1 主力数据文件")
    exit(1)

print(f"加载 {len(paths)} 个 Exp1 CSV 文件...")
frames = []
for path in paths:
    try:
        df = pd.read_csv(path)
        frames.append(df)
    except Exception as e:
        print(f"  ⚠️  {Path(path).name}: {e}")

if not frames:
    print("❌ 无法加载任何数据")
    exit(1)

df = pd.concat(frames, ignore_index=True)
print(f"合并后：{len(df)} 行\n")

# ─────────────────────────────────────────────────────────────────────
# 检查 1: 数据完整性
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("检查 1: 数据完整性（每个 (Nm, ρ) 的样本量）")
print("=" * 70)

counts = df.groupby(['bad_client_num', 'poison_rate']).size().reset_index(name='count')
nm4_counts = counts[counts['bad_client_num'] == 4].sort_values('poison_rate')

print("\nNm=4 的数据量分布：")
print(nm4_counts.to_string(index=False))
print(f"\n→ 若样本量跨 ρ 差异大，说明某些点跑得不充分")

# ─────────────────────────────────────────────────────────────────────
# 检查 2: 主任务性能 (MTA)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("检查 2: 主任务性能 (MTA) 的 ρ 依赖")
print("=" * 70)
print("若高 ρ 时 MTA 大幅下降，可能是「投毒破坏了主任务」导致虚假ASR下降")
print()

for nm in [1, 2, 4, 8, 16, 32]:
    print(f"Nm={nm}:")
    sub = df[df['bad_client_num'] == nm]

    for rho in sorted(sub['poison_rate'].unique()):
        rho_sub = sub[sub['poison_rate'] == rho].sort_values(['seed', 'round'])

        tail_mtas = []
        for seed in rho_sub['seed'].unique():
            seed_data = rho_sub[rho_sub['seed'] == seed].sort_values('round')
            if len(seed_data) >= 3:
                tail_mta = seed_data.iloc[-3:]['mta_personalized'].mean()
                tail_mtas.append(tail_mta)

        if tail_mtas:
            mean_mta = np.mean(tail_mtas)
            std_mta = np.std(tail_mtas)
            print(f"  ρ={rho:4.2f}: MTA_tail={mean_mta:.4f} ± {std_mta:.4f} (n_seed={len(tail_mtas)})")
    print()

# ─────────────────────────────────────────────────────────────────────
# 检查 3: ASR 尾部稳定性
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("检查 3: ASR 尾部稳定性（方差在高 ρ 时的行为）")
print("=" * 70)
print("若高 ρ 时方差/std 明显增大，说明后门训练不稳定")
print()

for nm in [1, 2, 4]:
    print(f"Nm={nm}:")
    sub = df[df['bad_client_num'] == nm]

    for rho in sorted(sub['poison_rate'].unique()):
        rho_sub = sub[sub['poison_rate'] == rho].sort_values(['seed', 'round'])

        tail_asrs = []
        for seed in rho_sub['seed'].unique():
            seed_data = rho_sub[rho_sub['seed'] == seed].sort_values('round')
            if len(seed_data) >= 3:
                tail_asr = seed_data.iloc[-3:]['asr_paper_filtered_benign'].mean()
                if pd.notna(tail_asr):
                    tail_asrs.append(tail_asr)

        if tail_asrs:
            mean_asr = np.mean(tail_asrs)
            std_asr = np.std(tail_asrs)
            var_asr = np.var(tail_asrs)
            print(f"  ρ={rho:4.2f}: ASR={mean_asr:.4f} ± {std_asr:.4f} (var={var_asr:.6f}, n_seed={len(tail_asrs)})")
    print()

# ─────────────────────────────────────────────────────────────────────
# 诊断结论
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("诊断结论框架")
print("=" * 70)
print("""
📊 三种可能性：

A. ✅ 真实物理现象（后门稳定性随 ρ 下降）
   证据：
   - 样本量均衡 → 不是数据缺陷
   - MTA 不下降 → 不是主任务退化
   - ASR std/var 随 ρ 增大 → 后门训练不稳定

B. ⚠️ 虚假ASR下降（主任务被投毒破坏）
   证据：
   - 高 ρ 时 MTA 明显下降（如 >0.05 相对下降）
   - 但 asr_paper_benign 仍下降 → 评估模型变差，非后门失效

C. 🔴 数据问题
   证据：
   - 高 ρ 点样本量远少于低 ρ 点
   - ASR std 异常高（>0.3）→ 单个运行方差过大

你的观察显示 ρ=0.1 最强 (0.88) → ρ=1.0 很弱 (0.08)。
根据上面的输出，判断是 A/B/C 哪一种。
""")
