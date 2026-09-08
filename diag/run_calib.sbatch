#!/bin/bash
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=badpfl-calib
# ══════════════════════════════════════════════════════════════════════════
# run_calib.sbatch  —  Stage B 收敛标定（导师意见 #5）的 6 个 run。
#
#   sbatch diag/run_calib.sbatch              # local_steps {15,45,75} x seed {0,1}
#   sbatch diag/run_calib.sbatch 0            # 只跑 seed 0（3 个 run，先探路）
#
# ⚠️ **必须整个跑在容器里**：`run_exp1.py --execute` 会用
#    `subprocess.run(["python", "-m", "diag.run_fl", ...])` 逐个起子进程，
#    子进程继承容器命名空间 —— 所以只要外层在容器内，子进程就也在。
#    反过来，外层在容器外的话，6 个 run 全部 import 不到 torch。
#
# `--execute` 是**顺序**跑 6 个 run 并在任一失败时中止（不是 job array）。
# 单 seed 探路用第一个参数。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

SEEDS="${1:-0 1}"
# 标定必须跑**主力要用的那条臂**，否则标出来的成本不适用。
# 默认 fedrep（Exp 1 与 Exp 3 对齐）；PFL=fedbn 可换。
PFL="${PFL:-fedrep}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=diag/cluster_env.sh
source "$ROOT/diag/cluster_env.sh"

cd "$ROOT"
echo "[calib] ROOT=$ROOT  SEEDS=$SEEDS"
echo "[calib] 先 dry-run 打印清单（不跑），确认参数对得上："
$PY -m diag.run_exp1 --stage calib --pfl "$PFL" --seeds $SEEDS

echo
echo "[calib] 开始执行 ——"
$PY -m diag.run_exp1 --stage calib --pfl "$PFL" --seeds $SEEDS --execute
RC=$?

echo "[calib] exit code = $RC"
echo "[calib] 读数（纯 stdlib，登录节点上裸 python3 也行）："
echo "        python3 -m diag.read_calibration"
exit $RC
