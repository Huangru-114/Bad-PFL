#!/bin/bash
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=badpfl-vfedrep
# ══════════════════════════════════════════════════════════════════════════
# verify_fedrep.sbatch  —  **烧 100 个 run 之前，先证明 FedRep 真的在做 FedRep。**
#
#   sbatch diag/verify_fedrep.sbatch
#
# FedRep 是本轮在 diag/ 里新写的旁路实现（pfl_fedrep.py），**从没在集群上跑过**。
# 它的 15 条单元测试里有 5 条需要 torch、本机只能 skip；而 Stage B/D 现在默认
# 就是 fedrep —— 不先验证就等于拿 100 个 GPU-run 赌一个没跑过的实现。
#
# 三步（约 30–60 分钟，比一个主力 run 还便宜）：
#   1. 跑 test_pfl_fedrep 的**全部** 15 条（本机 skip 掉的 5 条在这里才真跑）
#   2. 两条臂各一个**短** run（10 轮），产出 ckpt
#   3. evidence_fedrep 对拍：fedrep 臂 linear.* 不动 / BN 变；fedbn 臂反过来
#
# 任一步失败就直接退出并给出非零码 —— 这是 Stage B/D 的门禁。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=diag/cluster_env.sh
source "$ROOT/diag/cluster_env.sh"
cd "$ROOT"

LOGDIR="${BADPFL_LOGDIR:-$ROOT/logs}"; mkdir -p "$LOGDIR"
CKPT="${BADPFL_CKPT_ROOT:-$ROOT/checkpoints}"
ROUNDS="${VERIFY_ROUNDS:-10}"          # 只要能看出「哪些参数动了」，不需要收敛

echo "══ [1/3] FedRep 单元测试（含本机 skip 掉的 5 条）══"
$PY -m diag.tests.run_tests test_pfl_fedrep 2>&1 | tee "$LOGDIR/verify_fedrep_units.log"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    echo "[verify] ✗ 单元测试没过（exit $RC）—— 先修实现，不要往下跑。"
    exit "$RC"
fi

echo
echo "══ [2/3] 两条臂各一个 $ROUNDS 轮的短 run ══"
for ARM in fedbn fedrep; do
    TAG="vfedrep_${ARM}"
    echo "── $ARM ──"
    $PY -m diag.run_fl \
        --mode attack --alpha 0.5 --seed 0 --defense fedavg \
        --client-num 40 --select-per-round 8 --local-steps 15 \
        --model-size 10 --total-round "$ROUNDS" --eval-every "$ROUNDS" \
        --eval-include-malicious --pfl "$ARM" \
        --bad-client-num 4 --poison-rate 0.1 \
        --instrument-dir instrumentation --results-dir results/raw \
        --ckpt-root "$CKPT" --run-tag "$TAG" \
        2>&1 | tee "$LOGDIR/verify_fedrep_${ARM}.log"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        echo "[verify] ✗ $ARM 臂的短 run 挂了（exit $RC）"
        exit "$RC"
    fi
done

echo
echo "══ [3/3] 对拍：两条臂各自冻结/更新了正确的参数 ══"
# run_fl 的 ckpt 目录名：attack_a<alpha>_s<seed>_<run-tag>
$PY -m diag.evidence_fedrep \
    --fedbn  "$CKPT/attack_a0.5_s0_vfedrep_fedbn" \
    --fedrep "$CKPT/attack_a0.5_s0_vfedrep_fedrep" \
    2>&1 | tee "$LOGDIR/verify_fedrep_evidence.log"
RC=${PIPESTATUS[0]}

echo
if [ "$RC" -eq 0 ]; then
    echo "[verify] ✓ FedRep 通过门禁 —— 可以跑 Stage B 标定了。"
else
    echo "[verify] ✗ 对拍失败（exit $RC）：FedRep 没有按预期冻结/聚合参数。"
    echo "[verify]   **不要**用 fedrep 跑 Stage B/D，先看上面的逐层报告。"
fi
exit $RC
