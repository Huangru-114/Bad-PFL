"""FedRep / FedBN 两条臂的正确性证据 —— 逐位判据，互为对照。

# 要证明什么

「某组参数有没有被聚合」的**唯一决定性读数**是「它相对训练开始时的初始值动没动」。
比较客户端之间的差异是不够的：客户端收到广播后还会本地训练，所以任何一组参数
最后都各不相同 —— 那种比较区分不出「私有」与「共享后又各自训了」。

`run_fl` 现在在训练开始前存 `init_global.pt`，于是判据变成逐位的，而且两条臂
**互为对照**（同一份代码、同一个判据，只换 `--pfl`）：

| 全局模型的参数组 | fedbn 臂 | fedrep 臂 |
|---|---|---|
| `linear.*`（分类头） | **变了**（共享，参与聚合） | **逐位不变**（私有，被 pop 掉） |
| `bn*` / `shortcut.1*`（BN） | **逐位不变**（私有，被 pop 掉） | **变了**（参与聚合） |

对角线全对 = 两条臂都实现正确。任何一格反了，就是那条臂接错了。
只有一条臂的结果说明不了问题 —— 比如 fedrep 的 linear 不变，也可能是
「整个聚合根本没跑」；但同一份代码下 fedbn 的 linear 变了，就排除了这种可能。

# 怎么产生输入

    # 两条臂，除了 --pfl 之外逐字相同
    python -m diag.run_fl --mode attack --smoke --pfl fedbn  --local-steps 4 \\
           --run-tag ev_fedbn  --ckpt-root ./checkpoints
    python -m diag.run_fl --mode attack --smoke --pfl fedrep --local-steps 4 \\
           --run-tag ev_fedrep --ckpt-root ./checkpoints

    python -m diag.evidence_fedrep \\
        --fedbn  checkpoints/attack_a0.5_s0_ev_fedbn \\
        --fedrep checkpoints/attack_a0.5_s0_ev_fedrep

⚠️ `--local-steps 4` 不能省：smoke 默认 `local_steps=1`，此时 FedRep 的
`default_head_steps` 返回 0，**头阶段一步都不跑**，两阶段训练没有被执行。

退出码 0 = 2×2 全部符合预期；1 = 有格子不符（正文列出哪一格）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from diag.fedbn import is_fedbn_private_key
from diag.pfl_fedrep import is_fedrep_private_key


def _load(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise SystemExit(f"❌ 找不到 {path}\n"
                         "   （init_global.pt 需要 run_fl 在本轮改动之后跑过；"
                         "旧的 checkpoint 目录里没有这个文件）")
    return torch.load(path, map_location="cpu")


def _changed(init: Dict[str, torch.Tensor], final: Dict[str, torch.Tensor],
             keys: List[str]) -> Tuple[int, int, float]:
    """返回 (变了的 key 数, 总 key 数, 最大逐元素绝对差)。"""
    n_changed, max_abs = 0, 0.0
    for k in keys:
        a, b = init[k], final[k]
        if not torch.is_tensor(a) or not a.dtype.is_floating_point:
            if not torch.equal(a, b):
                n_changed += 1
            continue
        d = (b.float() - a.float()).abs().max().item()
        max_abs = max(max_abs, d)
        if d > 0:
            n_changed += 1
    return n_changed, len(keys), max_abs


def audit_arm(ckpt_dir: Path, arm: str) -> Dict[str, object]:
    init = _load(ckpt_dir / "init_global.pt")
    final = _load(ckpt_dir / "global.pt")
    assert set(init) == set(final), "init 与 final 的 key 集合不同，模型结构变了？"

    head_keys = sorted(k for k in init if is_fedrep_private_key(k))
    bn_keys = sorted(k for k in init if is_fedbn_private_key(k))
    body_keys = sorted(k for k in init
                       if not is_fedrep_private_key(k) and not is_fedbn_private_key(k))
    assert head_keys, "找不到分类头的 key（linear.*）—— 模型结构变了，请更新本脚本"
    assert bn_keys, "找不到 BN 的 key —— 模型结构变了，请更新本脚本"

    out = {"arm": arm, "dir": str(ckpt_dir)}
    for name, keys in (("head", head_keys), ("bn", bn_keys), ("body", body_keys)):
        n, tot, mx = _changed(init, final, keys)
        out[name] = {"changed": n, "total": tot, "max_abs_diff": mx}
    return out


def _fmt(g: Dict[str, object]) -> str:
    return f"{g['changed']:>3}/{g['total']:<3} 变，max|Δ|={g['max_abs_diff']:.3e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fedbn", required=True, type=Path, help="fedbn 臂的 ckpt 目录")
    ap.add_argument("--fedrep", required=True, type=Path, help="fedrep 臂的 ckpt 目录")
    args = ap.parse_args(argv)

    bn = audit_arm(args.fedbn, "fedbn")
    rp = audit_arm(args.fedrep, "fedrep")

    print("\n全局模型：训练结束 vs 训练开始（逐位比较）\n")
    print(f"{'参数组':<22} {'fedbn 臂':<32} {'fedrep 臂':<32}")
    print("-" * 88)
    print(f"{'linear.* (分类头)':<20} {_fmt(bn['head']):<32} {_fmt(rp['head']):<32}")
    print(f"{'bn* / shortcut.1*':<22} {_fmt(bn['bn']):<32} {_fmt(rp['bn']):<32}")
    print(f"{'其余 backbone':<22} {_fmt(bn['body']):<32} {_fmt(rp['body']):<32}")

    checks = [
        ("fedbn  的分类头应当**变了**（共享，参与聚合）",
         bn["head"]["changed"] > 0),
        ("fedbn  的 BN 应当**逐位不变**（私有，被 pop）",
         bn["bn"]["changed"] == 0),
        ("fedrep 的分类头应当**逐位不变**（私有，被 pop）",
         rp["head"]["changed"] == 0),
        ("fedrep 的 BN 应当**变了**（参与聚合）",
         rp["bn"]["changed"] > 0),
        ("两条臂的其余 backbone 都应当变了（否则聚合根本没跑）",
         bn["body"]["changed"] > 0 and rp["body"]["changed"] > 0),
    ]

    print("\n判据：")
    bad = 0
    for text, ok in checks:
        print(f"  {'✅' if ok else '❌'}  {text}")
        bad += 0 if ok else 1

    if bad:
        print(f"\n❌ {bad} 条不符。对角线不成立说明至少一条臂接错了 —— "
              "先看 use_fedrep / use_fedbn 的 key 判定与挂载点。")
        return 1
    print("\n✅ 2×2 对角线全部成立：两条臂的私有/共享划分都与定义一致，"
          "且互为对照（同一份代码只换 --pfl）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
