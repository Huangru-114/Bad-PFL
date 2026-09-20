#!/bin/bash
#SBATCH --job-name=exp1_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis_exp1.log

# 在集群上生成 Exp1 的所有图表
# 用法：sbatch diag/analysis_exp1.sbatch

set -e

cd "$(dirname "$0")/.."
echo "[analysis] 工作目录: $(pwd)"

# 加载容器环境
source diag/cluster_env.sh
echo "[analysis] 容器环境已加载: $PYTHON_EXEC"

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
$PYTHON_EXEC -m diag.analysis_exp1 \
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
