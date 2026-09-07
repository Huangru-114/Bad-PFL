"""两库骨干对齐：Bad-PFL 的 Exp1 与 tf-dpfl 的 Exp3 必须跑同一个网络。

**为什么要一条测试而不是一句注释**：报告 §2 的 "Common configuration" 里写着
ResNet-10，而 `exp1.model_size` 一直是 **18** —— 两库的 ASR 绝对值因此不可同框，
而这件事从任何一份产物里都看不出来（CSV 不记骨干）。现在把它钉成断言：
config 一改回 18 就红。

纯 stdlib（YAML 用行扫描、resnet.py 用源码正则），不 import torch/numpy，
所以本地也跑得动 —— 这条正是「本地测不了就没人跑」的反例。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "diag" / "config.yaml"
RESNET = ROOT / "resnet.py"

# tf-dpfl 侧的 build_resnet10 是 BasicBlock [1,1,1,1]（models/cnn.py）。
# 那个仓库在这里够不着，所以把期望值写死在这里，并在下面断言 Bad-PFL 的
# size=10 分支确实是这个形状 —— 两边同时改才可能悄悄错开。
TFDPFL_RESNET10_BLOCKS = [1, 1, 1, 1]


def _yaml_scalar(section: str, key: str) -> str:
    """从 config.yaml 里取 `section:` 段下的 `key:` 标量。

    不用 yaml 库（本地可能没有），按缩进做行扫描：顶层段是 0 缩进，
    段内条目是 2 缩进。行尾注释切掉。
    """
    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    in_section = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_section = line.split(":")[0].strip() == section
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith(f"{key}:"):
                return stripped.split(":", 1)[1].split("#")[0].strip()
    raise AssertionError(f"config.yaml 里找不到 {section}.{key}")


def test_exp1_backbone_is_resnet10():
    """Exp1 与 tf-dpfl 的 Exp3 同骨干 —— 否则两库的 ASR 绝对值不能同框比较。

    要改回 18 的话，请同时改掉报告 §2 的 "ResNet-10"，并明确写出
    「两库骨干不同，绝对值不可比」。
    """
    assert _yaml_scalar("exp1", "model_size") == "10", (
        "exp1.model_size 必须是 10（与 tf-dpfl 的 build_resnet10 同构）")


def test_size_10_really_means_basicblock_1111():
    """`--model-size 10` 落到的分支必须与 tf-dpfl 的 build_resnet10 同形状。

    只对齐一个数字（10）是不够的 —— 两个仓库各自定义「ResNet-10」是什么。
    这条扫 resnet.py 的源码，确认 size==10 分支就是 BasicBlock [1,1,1,1]。
    """
    src = RESNET.read_text(encoding="utf-8")
    m = re.search(r"size\s*==\s*10:\s*\n\s*return ResNet\((\w+),\s*\[([\d,\s]+)\]", src)
    assert m, "resnet.py 里找不到 size==10 的分支（get_resnet 改过了？）"
    block, dims = m.group(1), [int(x) for x in m.group(2).split(",")]
    assert block == "BasicBlock", f"size=10 用的是 {block}，tf-dpfl 侧是 BasicBlock"
    assert dims == TFDPFL_RESNET10_BLOCKS, (
        f"size=10 是 {dims}，tf-dpfl 的 build_resnet10 是 {TFDPFL_RESNET10_BLOCKS}")


def test_fl_default_and_exp1_agree():
    """`fl.model_size`（默认）与 `exp1.model_size` 不一致会造成难查的错配：
    `run_exp1` 显式传 `--model-size`，而手敲 `run_fl` 会落到 `fl` 的默认值上，
    两条路径跑出不同的网络、CSV 长得一模一样。"""
    assert _yaml_scalar("fl", "model_size") == _yaml_scalar("exp1", "model_size")
