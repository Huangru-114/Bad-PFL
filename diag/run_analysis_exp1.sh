#!/bin/bash
# 生成 Exp1 主力的所有图表
# 用法：sbatch run_analysis_exp1.sh  或  bash run_analysis_exp1.sh（本地有环境时）

set -e

echo "[analysis] 准备环境"
source diag/cluster_env.sh 2>/dev/null || echo "⚠️  集群环境变量未加载（本机环境）"

OUT_DIR="${OUT_DIR:-results/figs}"
mkdir -p "$OUT_DIR"

echo "[analysis] 加载 Exp1 植入数据并生成图表"
$PY -m diag.analysis_exp1 \
    --implantation-glob "results/raw/exp_ij_implantation_*_e1*.csv" \
    --out-dir "$OUT_DIR" \
    --summary-prefix "results/exp1" \
    --asr-column "asr_paper_filtered_benign" \
    --tail 3

echo "[analysis] 完成！图表已保存到 $OUT_DIR"
ls -lh "$OUT_DIR"/exp1*.png 2>/dev/null | tail -10 || echo "  ⚠️  未生成 PNG"
