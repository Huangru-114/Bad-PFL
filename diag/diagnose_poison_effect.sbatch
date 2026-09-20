#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -p cpu
#SBATCH -J diagnose_poison
#SBATCH -o logs/diagnose_poison.log

# 诊断投毒率 ρ 对 ASR 影响的根本原因
# 用法：sbatch diag/diagnose_poison_effect.sbatch

set -e

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
echo "[diagnose] 工作目录: $(pwd)"

# 加载集群环境
source diag/cluster_env.sh
echo "[diagnose] Python 环境已就绪: $PY_MODE"

mkdir -p logs

echo "[diagnose] 运行诊断分析..."
$PY diag/diagnose_poison_effect.py

echo ""
echo "=================================="
echo "[diagnose] ✅ 完成！"
echo "=================================="
echo ""
echo "诊断报告已保存到："
echo "  logs/diagnose_poison.log"
echo ""
echo "下一步："
echo "  1. 查看日志: cat logs/diagnose_poison.log"
echo "  2. 根据诊断结果判断 ρ 效应是真实还是虚假"
