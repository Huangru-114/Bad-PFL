#!/bin/bash
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=badpfl-probe-fedrep
# ══════════════════════════════════════════════════════════════════════════
# probe_fedrep.sbatch  —  「新的 FedRep 臂对不对」的四格探针（约 3 GPU-h）
#
#   sbatch diag/probe_fedrep.sbatch            # 四格顺序跑
#   sbatch diag/probe_fedrep.sbatch A          # 只跑锚点格
#
# ── 为什么需要它 ──────────────────────────────────────────────────────────
#
# 两件已知的事：
#
# 1. **上游没有 FedRep。** `main.py:110` 只有 `if args.pfl == "fedbn"`。
#    所以「我的 FedRep 与 Bad-PFL 原文一致吗」没有代码可对拍。能对的是
#    **管线**：先用 fedbn + 论文配置复现 `HANDOFF.md:325` 的锚点
#    （100/10/ResNet-10/300 轮/ρ=0.2 → Avg ASR 91.95%，论文报 82.22%）。
#    锚点不成立的话，FedRep 的任何数都没法解释 —— 先修管线。
#
# 2. **头:表示比大概率错了。** FedRep 原文（Collins et al., ICML 2021）的
#    官方代码是 `local_ep`(头)=10 / `local_rep_ep`(表示)=1，Flower 复现
#    Table 1 用 5:1。而 `diag/pfl_fedrep.py::default_head_steps` 是
#    **1:1**（`local_steps // 2`）。FedRep 的核心主张就是「每更新一次表示，
#    在低维的头上做很多次廉价更新」—— 1:1 把这个结构抹掉了。
#    征兆：同 steps=15 下 fedrep 的 MTA 0.2917 **低于** fedbn 的 0.4129，
#    而 FedRep 本该是更强的个性化器。
#
# ── 口径 ──────────────────────────────────────────────────────────────────
#
# 四格全部用**论文配置**：client_num=100 / select 10 / bad 10 / ρ=0.2 /
# 300 轮 / resnet10 / dir_alpha=0.5 / target=0 / local budget 以 **epoch** 计。
# ⚠️ 100 端下每端 500 张、batch 32、drop_last → **1 epoch = 15 steps**。
#    （40 端下是 39 steps —— 这正是我们此前把「15 steps」误读成 0.38 epoch 的来源。）
#
# FedRep 的两个预算是**相加**的（论文语义），不是把 local_steps 对半分：
#   B2: 表示 1 epoch(15) + 头 5x(75)  = 90 步   ← Flower 复现 Table 1 的比
#   B3: 表示 1 epoch(15) + 头 10x(150)= 165 步  ← FedRep 原文 CIFAR-10 的比
#
# ── 读什么 ────────────────────────────────────────────────────────────────
#
#   A  的 unfiltered ASR 应落在 0.82–0.92 → 管线成立
#   B1 → B2 → B3 的 MTA 若单调上升，就坐实 1:1 是问题；不动则另有原因
#   ASR 用 `asr_paper_*`（unfiltered，论文口径）与 A 对读；
#   filtered 换算见 diag/METRICS.md（unfiltered ≈ filtered*0.9 + 0.1）
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ONLY="${1:-}"
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck source=diag/cluster_env.sh
source "$ROOT/diag/cluster_env.sh"
cd "$ROOT"

LOGDIR="${BADPFL_LOGDIR:-$ROOT/logs}"; mkdir -p "$LOGDIR"
CKPT="${BADPFL_CKPT_ROOT:-$ROOT/checkpoints}"
ROUNDS="${PROBE_ROUNDS:-300}"          # 论文默认

# 格子：名字|臂|总步数|头步数（fedbn 臂头步数写 -，不传该参数）
# ⚠️ **A 也要跑**（2026-09-09 更正）：`acc_local_*` 是**与臂无关**的列，
#    fedbn 臂照样有值 —— 而它正是现在唯一能把两条臂放在一起比的那一列。
#    只有 MTA_sh / ASR_sh 是 FedRep 专属（fedbn 的全局 BN 停在初始化值）。
CELLS=(
  "A|fedbn|15|-"        # 锚点：论文配置，复现 HANDOFF.md:325 的 0.9195
  "B1|fedrep|15|7"      # 现状 1:1（default_head_steps 的结果）
  "B2|fedrep|90|75"     # 5:1，表示预算 = 1 epoch（Flower Table-1 复现）
  "B3|fedrep|165|150"   # 10:1，表示预算 = 1 epoch（FedRep 原文 CIFAR-10）
)

RC_ALL=0
for CELL in "${CELLS[@]}"; do
    IFS='|' read -r NAME ARM STEPS HEAD <<< "$CELL"
    if [ -n "$ONLY" ] && [ "$ONLY" != "$NAME" ]; then continue; fi

    TAG="probe_${NAME}_${ARM}_s${STEPS}"
    echo "══ $NAME | arm=$ARM | 总步数=$STEPS | 头步数=$HEAD ══"

    EXTRA=()
    if [ "$HEAD" != "-" ]; then EXTRA+=(--fedrep-head-steps "$HEAD"); fi

    $PY -m diag.run_fl \
        --mode attack --alpha 0.5 --seed 0 --defense fedavg \
        --client-num 100 --select-per-round 10 --bad-client-num 10 \
        --poison-rate 0.2 --local-steps "$STEPS" \
        --model-size 10 --total-round "$ROUNDS" --eval-every 10 \
        --eval-include-malicious --pfl "$ARM" "${EXTRA[@]}" \
        --instrument-dir instrumentation --results-dir results/raw \
        --ckpt-root "$CKPT" --run-tag "$TAG" \
        2>&1 | tee "$LOGDIR/probe_${NAME}.log"
    RC=${PIPESTATUS[0]}
    echo "[probe] $NAME exit=$RC"
    [ "$RC" -ne 0 ] && RC_ALL=$RC
done

echo
echo "[probe] 读数："
echo "  python3 -m diag.read_calibration --results-dir results/raw --asr-col asr_paper_benign"
echo "  （--asr-col 用 unfiltered 的论文口径，才能和 HANDOFF.md:325 的 0.9195 对读）"
exit $RC_ALL
