"""集群脚本：容器、SLURM 头、入口名（Bad-PFL 侧）。

**为什么需要**：这个仓库此前**一个集群脚本都没有**，全靠手敲命令。
于是「Stage B 怎么跑」这件事只存在于聊天记录里，而其中一版是错的 ——
漏了容器，6 个 run 会全部 import 不到 torch。

三条最容易错、且错了要等 GPU 排到才发现的：
  1. 容器搞混：Bad-PFL 是 **torch_fl.sif**，tf-dpfl 是 tensorflow.sif。
  2. 入口名搞混：这边是 **`python`**，tf-dpfl 那边是 `python3`。
  3. 默认加了 `--bind`：Arrhenius 上标准写法不带它，加了反而失败。

纯 stdlib，本地秒级。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV = ROOT / "diag" / "cluster_env.sh"
SBATCH = sorted((ROOT / "diag").glob("*.sbatch"))

SIF = "/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/torch_fl.sif"
TF_SIF_NAME = "tensorflow.sif"


def _code(path: Path) -> str:
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


def test_there_are_cluster_scripts_to_check():
    """反向自检：路径写错时不能退化成「零个文件全部通过」。"""
    assert ENV.exists(), "diag/cluster_env.sh 不存在"
    assert SBATCH, "diag/ 下一个 .sbatch 都没有"


def test_uses_the_torch_container_not_the_tensorflow_one():
    """两个仓库的容器不同。用错的那个 → torch import 不到，
    而报错长得像「环境没装好」。"""
    env = ENV.read_text(encoding="utf-8")
    assert SIF in env, f"容器不是 {SIF}"
    assert TF_SIF_NAME not in _code(ENV), \
        "diag/cluster_env.sh 里出现了 tf-dpfl 的容器"


def test_entrypoint_is_python_not_python3():
    """torch_fl.sif 的入口是 `python`（用户给的标准示例如此）。"""
    env = _code(ENV)
    assert re.search(r'PY="apptainer exec --nv \$BADPFL_SIF python"', env), \
        "不带 --bind 的那条路径入口名不对（应为 python）"
    assert not re.search(r"\$BADPFL_SIF python3", env), \
        "用成了 python3 —— 那是 tf-dpfl 那边的入口"


def test_bind_is_off_by_default():
    """Arrhenius 上标准写法不带 --bind；加了反而失败，
    而失败信息长得像「容器里看不到仓库目录」（tf-dpfl 侧踩过）。"""
    env = _code(ENV)
    assert 'BADPFL_BIND="${BADPFL_BIND:-}"' in env, "BADPFL_BIND 默认不该有值"
    assert 'if [ -n "$BADPFL_BIND" ]; then' in env, "显式设值时应仍然生效"


def test_sbatch_header_matches_the_arrhenius_standard():
    for path in SBATCH:
        head = "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                         if ln.startswith("#SBATCH"))
        for pat, what in [(r"^#SBATCH\s+-A\s+naiss2026-4-650-gpu\s*$", "-A ..._gpu"),
                          (r"^#SBATCH\s+-p\s+gpu\s*$", "-p gpu"),
                          (r"^#SBATCH\s+--gpus\s+1\s*$", "--gpus 1"),
                          (r"^#SBATCH\s+-c\s+4\s*$", "-c 4")]:
            assert re.search(pat, head, re.M), f"{path.name} 缺 {what}：\n{head}"
        assert "--gpus-per-node" not in head, f"{path.name} 带着 Alvis 的写法"


def test_sbatch_sources_cluster_env_and_uses_PY():
    """整条链必须在容器里：`run_exp1 --execute` 用 subprocess 起
    `python -m diag.run_fl`，子进程继承容器命名空间 —— 外层在容器外的话
    6 个 run 全部 import 不到 torch。"""
    for path in SBATCH:
        code = _code(path)
        assert "diag/cluster_env.sh" in code, f"{path.name} 没 source cluster_env.sh"
        assert "$PY " in code, f"{path.name} 没用 $PY"
        bad = [ln.strip() for ln in code.splitlines()
               if re.search(r"(?<![\w$/])python3?\s+-m\s", ln) and "$PY" not in ln
               and not ln.lstrip().startswith("echo")]
        assert not bad, f"{path.name} 里有裸 python 调用：{bad}"


def test_container_path_is_not_hardcoded_in_sbatch():
    """容器路径收口在 cluster_env.sh 一处。"""
    for path in SBATCH:
        assert SIF not in _code(path), \
            f"{path.name} 硬写了容器路径；source cluster_env.sh 用 $PY"


# ══════════════════════════════════════════════════════════════════════════
# Stage D：job array 的接线
# ══════════════════════════════════════════════════════════════════════════
ARRAY = ROOT / "diag" / "exp1_array.sbatch"
SUBMIT = ROOT / "diag" / "submit_exp1.sh"


def test_stage_d_scripts_exist():
    assert ARRAY.exists() and SUBMIT.exists()


def test_array_task_reads_its_own_line():
    """一个 array task = 一条命令。靠 SLURM_ARRAY_TASK_ID 取行，
    所以必须(a)要求该变量存在、(b)按行号取、(c)取不到就报错而不是静默跑空。"""
    code = _code(ARRAY)
    assert "SLURM_ARRAY_TASK_ID" in code, "没用 array task id"
    assert 'sed -n "${IDX}p"' in code, "不是按行号取命令"
    assert '[ -n "$CMD" ]' in code, "空行没有拦截 —— 会静默跑一个空命令"


def test_array_runs_inside_the_container():
    """清单里的命令以裸 `python -m diag.run_fl` 开头，必须换成 $PY。"""
    code = _code(ARRAY)
    assert 'RUN="${CMD/#python /$PY }"' in code, "没有把 python 换成 $PY"
    assert "diag/cluster_env.sh" in code, "没 source cluster_env.sh"


def test_submitter_uses_the_no_gpu_interpreter():
    """生成清单在**登录节点**跑，那里没有 NVIDIA 驱动 —— 用带 --nv 的 $PY
    会让容器直接起不来。tf-dpfl 为此连拆三轮正确设计（其 CLAUDE.md 陷阱 #17）。"""
    code = _code(SUBMIT)
    assert "$PY_NOGPU" in code, "提交器没用 $PY_NOGPU"
    assert not re.search(r"\$PY\s+-m\s+diag", code), "提交器用了带 --nv 的 $PY"


def test_no_gpu_interpreter_really_drops_nv():
    env = _code(ENV)
    m = re.search(r'PY_NOGPU="apptainer exec ([^"]*)"', env)
    assert m, "cluster_env.sh 里没有 PY_NOGPU"
    assert "--nv" not in m.group(1), f"PY_NOGPU 还带着 --nv：{m.group(1)}"


def test_array_caps_concurrency():
    """一次占满队列会挡住别人（也挡住自己的标定）。"""
    code = _code(SUBMIT)
    assert "%" in code and "--array=" in code, "没有 array 并发上限"


# ══════════════════════════════════════════════════════════════════════════
# FedRep 门禁：烧 100 个 run 之前先证明它真的在做 FedRep
# ══════════════════════════════════════════════════════════════════════════
VERIFY = ROOT / "diag" / "verify_fedrep.sbatch"


def test_fedrep_verification_gate_exists():
    """FedRep 是本轮新写的旁路实现，**从没在集群上跑过**，而 Stage B/D 现在
    默认就是 fedrep。没有门禁 = 拿 100 个 GPU-run 赌一个没跑过的实现。"""
    assert VERIFY.exists(), "缺 diag/verify_fedrep.sbatch"


def test_gate_covers_all_three_checks():
    code = _code(VERIFY)
    assert "diag.tests.run_tests test_pfl_fedrep" in code, \
        "没跑单元测试（本机 skip 掉的 5 条只有在集群上才真跑）"
    assert code.count("--pfl") >= 1 and "fedbn" in code and "fedrep" in code, \
        "没有两条臂各跑一个短 run"
    assert "diag.evidence_fedrep" in code, "没有对拍两条臂冻结了哪些参数"


def test_gate_stops_on_the_first_failure():
    """任一步失败必须立刻退出并给非零码 —— 门禁「跑完但没过」等于没有门禁。"""
    code = _code(VERIFY)
    assert code.count("exit \"$RC\"") >= 2, "中间步骤失败没有立刻退出"
    assert "RC=${PIPESTATUS[0]}" in code, \
        "用 tee 之后没取管道首个命令的退出码（会永远拿到 tee 的 0）"


def test_gate_is_cheap():
    """门禁比一个主力 run 还便宜才会有人跑它。"""
    code = _code(VERIFY)
    assert 'ROUNDS="${VERIFY_ROUNDS:-10}"' in code, "轮数不是短跑"
    head = "\n".join(ln for ln in VERIFY.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("#SBATCH"))
    assert "-t 02:00:00" in head, f"时限不像短跑：\n{head}"
