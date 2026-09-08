#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit_exp1.sh  —  Stage D 主力重跑的提交器（登录节点跑，自己 sbatch 一个 array）
#
#   bash diag/submit_exp1.sh <stage> [--dry-run] [--max-parallel N]
#
#   stage ∈ 1 | corner | arm | 1b | persist
#     1        十字扫描：ρ 8 档 @ Nm=4  +  Nm 6 档 @ ρ=0.1（13 格）
#     corner   塌陷角：ρ{0.7,0.9,1.0} × Nm{16,32}（6 格，Nm=4 已在十字里）
#     arm      PFL 对照臂：只在十字交叉点，fedbn/fedrep 各跑
#     1b       攻击时间结构（调度）
#     persist  B2 持续性长跑
#
#   SEEDS="0 1 2" bash diag/submit_exp1.sh 1        # 换 seed 集
#   PFL=fedrep    bash diag/submit_exp1.sh arm      # 换 PFL 臂
#
# 例（推荐顺序）：
#   bash diag/submit_exp1.sh 1      --dry-run       # 先看清单与规模
#   SEEDS="0" bash diag/submit_exp1.sh 1            # 单 seed 探路（13 个 job）
#   bash diag/submit_exp1.sh 1                      # 全量 5 seed
#
# ⚠️ 本脚本**不 source cluster_env.sh 起 GPU 容器**：它只在登录节点生成清单 +
#    sbatch。生成清单要 import diag.config（需要 torch），所以用 $PY_NOGPU
#    —— **不带 --nv** 的同一个容器。登录节点没有 NVIDIA 驱动，带 --nv 会让
#    容器直接起不来（tf-dpfl 那边为此连拆三轮正确设计，见其 CLAUDE.md 陷阱 #17）。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

STAGE="${1:?用法: bash diag/submit_exp1.sh <1|corner|arm|1b|persist> [--dry-run]}"
shift || true

DRY=0
MAXP="${MAX_PARALLEL:-8}"        # array 并发上限，别一次占满队列
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --max-parallel) shift; MAXP="${1:?--max-parallel 缺参数}" ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=diag/cluster_env.sh
source "$ROOT/diag/cluster_env.sh"

SEEDS="${SEEDS:-}"
PFL="${PFL:-fedbn}"
LISTDIR="$ROOT/results/lists"
mkdir -p "$LISTDIR"
LIST="$LISTDIR/exp1_${STAGE}_${PFL}.txt"

# seed 缺省值**显式**从 config 读出来再传下去，不靠 run_exp1 的隐式回退
# （那个回退是 exp1.seeds = 2 个，而 Stage D 要 main.seeds = 5 个）。
# 显式传的另一个好处：清单里每条命令都带 --seed，事后一眼看得出跑了哪些。
if [ -z "$SEEDS" ]; then
    SEEDS="$($PY_NOGPU - "$STAGE" <<'PYEOF'
import sys
from diag.config import load_config
exp1 = load_config("diag/config.yaml").exp1
stage = sys.argv[1]
if stage == "arm" and "main" in exp1 and "control_arm" in exp1.main:
    seeds = exp1.main.control_arm.seeds
elif stage in ("1", "corner") and "main" in exp1:
    seeds = exp1.main.seeds
else:
    seeds = exp1.seeds
print(" ".join(str(int(s)) for s in seeds))
PYEOF
)"
    echo "[submit] seed 缺省 <- config：$SEEDS"
fi
SEED_ARGS=(--seeds $SEEDS)

echo "[submit] seeds = $SEEDS"
echo "[submit] 生成清单 -> $LIST"

echo "[submit] stage=$STAGE  pfl=$PFL"
$PY_NOGPU -m diag.run_exp1 --stage "$STAGE" --pfl "$PFL" \
    "${SEED_ARGS[@]}" --list-only > "$LIST"

N=$(wc -l < "$LIST")
if [ "$N" -eq 0 ]; then
    echo "[submit] 清单为空，没有可提交的格子。" >&2; exit 1
fi

echo
echo "──────────────────────────────────────────────"
echo "  清单长度 = $N 个 run（= $N 个 array task）"
echo "  并发上限 = $MAXP"
echo "  单 run 成本：**由 Stage B 标定给出**（read_calibration 的 GPU-s→plat）"
echo "  总机时 ≈ $N × 单 run 成本"
echo "──────────────────────────────────────────────"
echo

if [ "$DRY" -eq 1 ]; then
    echo "[submit] --dry-run：前 3 条命令 ——"
    head -3 "$LIST" | sed 's/^/    /'
    echo "    ..."
    echo "[submit] 未提交。去掉 --dry-run 即真正 sbatch。"
    exit 0
fi

sbatch --array="1-${N}%${MAXP}" "$ROOT/diag/exp1_array.sbatch" "$LIST"
echo "[submit] 已提交。查看：squeue -u \$USER"
echo "[submit] 跑完后分析：\$PY_NOGPU -m diag.analysis_exp1 \\"
echo "           --implantation-glob 'results/raw/exp_ij_implantation_*_e1*.csv' \\"
echo "           --out-dir results/figs --summary-prefix results/exp1"
