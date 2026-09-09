#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 08:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=badpfl-exp1
# ══════════════════════════════════════════════════════════════════════════
# exp1_array.sbatch  —  Stage D 主力重跑：**一个 array task = 一个 run**。
#   由 submit_exp1.sh 提交，一般不手动调用。
#
#   sbatch --array=1-N diag/exp1_array.sbatch <命令清单文件>
#
# **为什么是 job array 而不是 `--execute` 顺序跑**：主力是 65~100 个 run，
# 每个小时量级；顺序跑一个 job 装不下（也没有断点续跑）。array 里每格独立
# 排队、独立失败、独立重投。
#
# 清单由 submit_exp1.sh 用 `run_exp1.py --list-only` **一次性**生成并落盘，
# array 各 task 只按行号取 —— 现生成的话，`--skip-existing` 之类一变行号就错位。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

LIST="${1:?用法: sbatch --array=1-N diag/exp1_array.sbatch <命令清单文件>}"
IDX="${SLURM_ARRAY_TASK_ID:?本脚本必须以 job array 提交（缺 SLURM_ARRAY_TASK_ID）}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=diag/cluster_env.sh
source "$ROOT/diag/cluster_env.sh"

cd "$ROOT"
[ -f "$LIST" ] || { echo "[exp1] 清单不存在: $LIST"; exit 2; }

CMD="$(sed -n "${IDX}p" "$LIST")"
[ -n "$CMD" ] || { echo "[exp1] 清单第 $IDX 行是空的（--array 范围超出清单长度？）"; exit 2; }

LOGDIR="${BADPFL_LOGDIR:-$ROOT/logs}"
mkdir -p "$LOGDIR"
# 从命令里抠出 run-tag 当日志名 —— 每格独立，互不覆盖
TAG="$(printf '%s\n' "$CMD" | sed -n 's/.*--run-tag \([^ ]*\).*/\1/p')"
LOG="$LOGDIR/exp1_${TAG:-idx$IDX}.log"

echo "[exp1] array task $IDX / 清单 $LIST"
echo "[exp1] tag = ${TAG:-<无>}"
echo "[exp1] 大日志 -> $LOG （留集群，不进 git）"
echo "[exp1] 命令: $CMD"
echo

# 清单里的命令以 `python -m diag.run_fl ...` 开头；换成容器里的解释器。
# **必须整条在容器里**，否则 import 不到 torch。
RUN="${CMD/#python /$PY }"
set +e
eval "$RUN" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[exp1] exit code = $RC"
echo "[exp1] 产物：results/raw/ 下该 tag 的 implantation CSV"
exit $RC
