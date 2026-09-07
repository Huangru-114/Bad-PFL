# diag/cluster_env.sh  —  解析「用什么 python 跑」的**唯一**地方（被 source，不要直接执行）
#
# ══════════════════════════════════════════════════════════════════════════
# 集群上所有 python 都必须在 apptainer 容器里跑，否则 torch 都 import 不到。
#
# ⚠️ **Bad-PFL 用的容器与 tf-dpfl 不是同一个**：
#       Bad-PFL  torch_fl.sif    入口 `python`
#       tf-dpfl  tensorflow.sif  入口 `python3`
#    两个仓库的容器不要混用。
#
# Arrhenius 上实测可用的标准写法（用户给的示例）：
#     module load GPU/buildenv-nvhpc/25.9-cu13.0
#     apptainer exec --nv <abs .sif> python -m diag.exp_t3 ...
#   —— **没有 --bind**。这台机器的 apptainer 已在系统级把 /nobackup 挂进容器；
#      $PWD 也保留，所以 `--data-root ./data` 这类相对路径照常能用。
#
# 用法：
#     ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
#     source "$ROOT/diag/cluster_env.sh"
#     cd "$ROOT" && $PY -m diag.run_exp1 ...
#
# 覆盖：
#     BADPFL_PY="python3"       强制裸 python（本地开发）
#     BADPFL_SIF=/path/to.sif   换容器
#     BADPFL_BIND=/some/path    显式加 --bind（默认不加，见上）
# ══════════════════════════════════════════════════════════════════════════

BADPFL_SIF="${BADPFL_SIF:-/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/torch_fl.sif}"
BADPFL_BIND="${BADPFL_BIND:-}"

if [ -n "${BADPFL_PY:-}" ]; then
    PY="$BADPFL_PY"
    PY_MODE="override"
elif command -v apptainer >/dev/null 2>&1 && [ -f "$BADPFL_SIF" ]; then
    if type module >/dev/null 2>&1; then
        module load GPU/buildenv-nvhpc/25.9-cu13.0 2>/dev/null || \
            echo "[env] 警告：module load GPU/buildenv-nvhpc/25.9-cu13.0 失败，GPU 可能不可用" >&2
    fi
    # 入口是 `python`（torch_fl.sif 里如此），不是 python3 —— 与 tf-dpfl 那边相反。
    if [ -n "$BADPFL_BIND" ]; then
        PY="apptainer exec --nv --bind $BADPFL_BIND $BADPFL_SIF python"
    else
        PY="apptainer exec --nv $BADPFL_SIF python"
    fi
    PY_MODE="apptainer"
else
    # 本机：无 apptainer / 无容器 → 裸 python3。本机没有 torch，
    # 需要 torch 的测试会自己 skip（diag/tests/run_tests.py 支持 SkipTest）。
    PY="python3"
    PY_MODE="local"
fi

# 静默地跑在错误的环境里是最难查的一类问题 —— 每次跑之前扫一眼这行。
echo "[env] python = $PY   (mode=$PY_MODE)"
