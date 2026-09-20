#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH -J exp1_analysis
#SBATCH -o logs/analysis_exp1.log

# 在集群上生成 Exp1 的所有图表
# 用法：sbatch diag/analysis_exp1.sbatch

set -e

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
echo "[analysis] 工作目录: $(pwd)"

# 加载集群环境（定义 $PY 变量）
source diag/cluster_env.sh
echo "[analysis] Python 环境已就绪: $PY_MODE"

OUT_DIR="results/figs"
mkdir -p "$OUT_DIR"
mkdir -p logs

echo "[analysis] 步骤 1: 数据检查"
N_FILES=$(find results/raw -name "*_e1*.csv" | wc -l)
echo "  找到 $N_FILES 个 Exp1 CSV 文件"
if [ "$N_FILES" -lt 1 ]; then
    echo "❌ 找不到 Exp1 数据！"
    exit 1
fi

echo "[analysis] 步骤 2: 生成汇总统计与图表"
$PY -m diag.analysis_exp1 \
    --implantation-glob "results/raw/exp_ij_implantation_*_e1*.csv" \
    --out-dir "$OUT_DIR" \
    --summary-prefix "results/exp1" \
    --asr-column "asr_paper_filtered_benign" \
    --tail 3

echo ""
echo "=================================="
echo "[analysis] ✅ 完成！"
echo "=================================="
echo ""
echo "生成的汇总 CSV："
ls -lh results/exp1*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "生成的图表："
ls -lh "$OUT_DIR"/exp1*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  ⚠️  未找到 PNG"
echo ""
echo "下一步："
echo "  1. 查看图表: results/figs/"
echo "  2. 检查汇总: cat results/exp1_crossings.csv"
echo "  3. 查看日志: cat logs/analysis_exp1.log"
