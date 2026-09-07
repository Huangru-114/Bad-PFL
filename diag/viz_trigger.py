#!/usr/bin/env python3
"""触发器可视化（回应导师意见 #1 "visualize the attacks"）。

出一张 clean / ξ / δ / T(x) / 残差 的五联图，并把**实测的扰动幅度**打在图上，
让读者能自己核对 §2 里写的预算。

## 为什么必须放大显示

ξ 与 δ 的 L∞ 上界都是 4/255 ≈ 0.0157 —— 直接画出来是一块均匀的灰。
本脚本把每个扰动分量按 `|v| <= scale` 线性映射到 [0,1] 再画，
并把 **scale 的实际值写进 panel 标题**。不写放大倍数的扰动图等于没有信息。

## 为什么每张图必须注明"哪个模型、哪一轮"

ξ(x) = 1 步 PGD(client.local_model, x, y_true) —— **触发器是 model-dependent 的**，
同一张图在不同客户端、不同轮次上得到的扰动不同。所以图注固定写出
checkpoint 路径、客户端 id、轮次。见 diag/METRICS.md §2。

## 用法（集群，需要 torch + torchvision + 数据）

    python -m diag.viz_trigger \\
        --ckpt-dir checkpoints/attack_a0.5_s0_e1_bad4_rho0p1_s0 \\
        --client-id 0 --data-root ./data \\
        --out results/figs/trigger_visualization.png

`--ckpt-dir` 里要有 `generator.pt` 与 `client_<id>.pt`（`diag.hooks.save_run` 写的）。
**不加 `--execute` 也会先自检文件齐不齐并报缺哪个**，不用等 GPU 排到才发现。

⚠️ 铁律 #4：图里一律英文。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["amplify_for_display", "perturbation_stats", "panel_titles",
           "required_files", "main"]

# CIFAR-10 类名（图注用；main.py 的 transform 只有 ToTensor，图本来就在 [0,1]，
# 不需要反归一化）
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")

EPS = 4.0 / 255.0          # fba.py:6 的 pgd_attack 默认 epsilon / 生成器缩放


# ---------------------------------------------------------------------------
# 纯计算（不 import torch，本机可测）
# ---------------------------------------------------------------------------
def _display_value(v: float, scale: float) -> float:
    """单个像素的映射。**numpy 路径与纯 Python 路径共用这个公式的定义**，
    免得两条路各算各的（那是最典型的假绿）。"""
    return min(1.0, max(0.0, v / (2.0 * scale) + 0.5))


def amplify_for_display(values, scale: float):
    """把一个 ±scale 范围内的扰动线性映射到 [0, 1] 以便显示。

    ``0 -> 0.5``（中性灰），``+scale -> 1``，``-scale -> 0``。超出范围的**截断**
    而不是重新归一化 —— 重新归一化会让「这张图的扰动更大」变得看不出来，
    而跨 panel 可比正是这张图的意义。

    numpy 数组走向量化路径（imshow 要数组），嵌套 list 走递归路径。
    两条路的**公式**由 ``_display_value`` 唯一定义，
    `test_viz_trigger` 里有一条在 numpy 可用时逐元素对拍两条路。
    """
    if scale <= 0:
        raise ValueError(f"scale 必须为正，收到 {scale}")
    if hasattr(values, "ravel") and hasattr(values, "shape"):   # ndarray / tensor
        import numpy as np
        return np.clip(np.asarray(values, dtype=np.float64) / (2.0 * scale) + 0.5,
                       0.0, 1.0)
    if isinstance(values, (list, tuple)):
        return [amplify_for_display(v, scale) for v in values]
    return _display_value(float(values), scale)


def _flatten(values) -> List[float]:
    """numpy 数组 / 嵌套 list / 标量 -> 一维 float 列表。

    有 ravel 就用（快），否则纯 Python 递归 —— 于是 ``perturbation_stats``
    只有**一份**算术，本机没有 numpy 也测得动。
    """
    if hasattr(values, "ravel"):
        return [float(v) for v in values.ravel().tolist()]
    if isinstance(values, (list, tuple)):
        out: List[float] = []
        for v in values:
            out.extend(_flatten(v))
        return out
    return [float(values)]


def perturbation_stats(delta) -> Dict[str, float]:
    """一个扰动张量的 L∞ / L2 / 均值绝对值，以及占 4/255 预算的比例。

    图上要打的就是这几个数：读者据此核对 §2 声称的预算，而不是只能相信图注。
    空输入返回 nan（**不是 0** —— 0 会被读成「扰动为零」）。
    """
    a = _flatten(delta)
    if not a:
        nan = float("nan")
        return {"linf": nan, "l2": nan, "mean_abs": nan, "frac_of_eps": nan}
    linf = max(abs(v) for v in a)
    return {
        "linf": linf,
        "l2": sum(v * v for v in a) ** 0.5,
        "mean_abs": sum(abs(v) for v in a) / len(a),
        "frac_of_eps": linf / EPS,
    }


def panel_titles(stats_xi: Dict[str, float], stats_delta: Dict[str, float],
                 stats_total: Dict[str, float], scale: float) -> List[str]:
    """五个 panel 的英文标题（含实测幅度与放大倍数）。

    放大倍数**必须**出现在标题里：不写倍数的扰动图看起来像随机噪声图，
    读者无法判断它是 4/255 还是 40/255。
    """
    amp = f"amplified x{1.0 / (2.0 * scale):.0f}"
    return [
        "clean $x$",
        (r"PGD component $\xi$  (" + amp + ")\n"
         f"$L_\\infty$={stats_xi['linf']:.4f} "
         f"({stats_xi['frac_of_eps']:.2f} of 4/255)"),
        (r"generator component $\delta$  (" + amp + ")\n"
         f"$L_\\infty$={stats_delta['linf']:.4f} "
         f"({stats_delta['frac_of_eps']:.2f} of 4/255)"),
        r"triggered $T(x)=\mathrm{clip}(x+\xi)+\delta$",
        (r"total residual $T(x)-x$  (" + amp + ")\n"
         f"$L_\\infty$={stats_total['linf']:.4f} "
         f"({stats_total['frac_of_eps']:.2f} of 4/255)"),
    ]


def required_files(ckpt_dir, client_id: int) -> List[Path]:
    """这张图需要的 checkpoint 文件。缺哪个要**提前**报出来。"""
    d = Path(ckpt_dir)
    return [d / "generator.pt", d / f"client_{int(client_id)}.pt", d / "meta.json"]


def missing_files(ckpt_dir, client_id: int) -> List[Path]:
    return [p for p in required_files(ckpt_dir, client_id) if not p.exists()]


# ---------------------------------------------------------------------------
# 需要 torch 的部分
# ---------------------------------------------------------------------------
def _load(ckpt_dir: Path, client_id: int, device: Any):
    import torch
    from generator import Autoencoder
    from resnet import get_resnet

    meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    model_size = int(meta.get("model_size", 10))
    num_classes = int(meta.get("num_classes", 10))

    gen = Autoencoder().to(device)
    gen.load_state_dict(torch.load(ckpt_dir / "generator.pt", map_location=device))
    gen.eval()

    model = get_resnet(size=model_size, num_classes=num_classes).to(device)
    model.load_state_dict(
        torch.load(ckpt_dir / f"client_{client_id}.pt", map_location=device))
    model.eval()
    return gen, model, meta


def build_figure(ckpt_dir, client_id: int, data_root: str, out_path,
                 n_images: int = 4, device: str = "cpu",
                 scale: float = EPS, seed: int = 0):
    """画图并保存。返回 (out_path, 每张图的统计量)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision

    from fba import pgd_attack
    from .analysis import assert_no_cjk_in_figure
    from .figstyle import apply_publication_style

    ckpt_dir = Path(ckpt_dir)
    gone = missing_files(ckpt_dir, client_id)
    if gone:
        raise FileNotFoundError(
            "缺这些 checkpoint 文件：\n  " + "\n  ".join(str(p) for p in gone)
            + "\n（它们由 diag.hooks.save_run 在 run 结束时写出）")

    apply_publication_style()
    dev = torch.device(device)
    gen, model, meta = _load(ckpt_dir, client_id, dev)

    # 与 main.py:58 完全一致：只有 ToTensor，**没有 Normalize** ——
    # 所以图本来就在 [0,1]，ε=4/255 是直接的像素尺度，不需要反归一化。
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(data_root, train=False,
                                           download=False, transform=transform)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=int(n_images), replace=False)
    x = torch.stack([dataset[int(i)][0] for i in idx]).to(dev)
    y = torch.tensor([dataset[int(i)][1] for i in idx]).to(dev)

    # ξ：1 步无目标 PGD，与 fba.py 的投毒路径同一个函数（不是复刻）
    x_adv = pgd_attack(model, x, y)
    xi = (x_adv - x).detach()
    # δ：生成器输出 / 255 * 4，与 fba.py:38 / :52 同一个式子
    with torch.no_grad():
        delta = (gen(x) / 255.0 * 4.0).detach()
    triggered = (x_adv + delta).detach()
    residual = (triggered - x).detach()

    xs = x.cpu().numpy().transpose(0, 2, 3, 1)
    xis = xi.cpu().numpy().transpose(0, 2, 3, 1)
    ds = delta.cpu().numpy().transpose(0, 2, 3, 1)
    ts = triggered.cpu().numpy().transpose(0, 2, 3, 1)
    rs = residual.cpu().numpy().transpose(0, 2, 3, 1)

    n = len(idx)
    fig, axes = plt.subplots(n, 5, figsize=(13.0, 2.7 * n), squeeze=False)
    all_stats = []
    for r in range(n):
        s_xi = perturbation_stats(xis[r])
        s_d = perturbation_stats(ds[r])
        s_t = perturbation_stats(rs[r])
        all_stats.append({"index": int(idx[r]),
                          "label": CIFAR10_CLASSES[int(y[r])],
                          "xi": s_xi, "delta": s_d, "residual": s_t})
        panels = [xs[r],
                  amplify_for_display(xis[r], scale),
                  amplify_for_display(ds[r], scale),
                  np.clip(ts[r], 0.0, 1.0),
                  amplify_for_display(rs[r], scale)]
        titles = panel_titles(s_xi, s_d, s_t, scale)
        for c in range(5):
            ax = axes[r][c]
            ax.imshow(panels[c], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
        axes[r][0].set_ylabel(f"true: {CIFAR10_CLASSES[int(y[r])]}", fontsize=8)

    rounds = meta.get("total_round", "?")
    fig.suptitle(
        "Bad-PFL trigger decomposition   "
        f"(client {client_id}, round {rounds}, "
        f"target class {meta.get('target_class', 0)} = "
        f"{CIFAR10_CLASSES[int(meta.get('target_class', 0))]})",
        fontsize=11)
    fig.text(0.5, -0.01,
             "The trigger is model-dependent: xi is a 1-step untargeted PGD "
             "step against THIS client's model, so the same image yields a "
             "different perturbation on another client or round.  "
             "xi and delta are each bounded by 4/255; they are added and no "
             "clipping is applied after delta, so the total residual can reach "
             "8/255.  Perturbation panels are amplified (factor in the title); "
             "values outside the range are clipped, not renormalised, so panels "
             "stay comparable.",
             ha="center", va="top", fontsize=7.5, wrap=True)
    fig.tight_layout()

    assert_no_cjk_in_figure(fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path, all_stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results/figs/trigger_visualization.png")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda:0。这张图很小，CPU 就够，不用排 GPU 队")
    ap.add_argument("--scale", type=float, default=EPS,
                    help="扰动显示的放大基准（默认 4/255 = 满量程）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-only", action="store_true",
                    help="只查 checkpoint 文件齐不齐，不画图（不需要 torch）")
    args = ap.parse_args(argv)

    gone = missing_files(args.ckpt_dir, args.client_id)
    if gone:
        print("[viz_trigger] 缺这些文件：")
        for p in gone:
            print(f"  {p}")
        print("（由 diag.hooks.save_run 在 run 结束时写出；先确认 run 跑完了）")
        return 1
    print(f"[viz_trigger] checkpoint 齐全：{args.ckpt_dir}")
    if args.check_only:
        return 0

    out, stats = build_figure(args.ckpt_dir, args.client_id, args.data_root,
                              args.out, n_images=args.n_images,
                              device=args.device, scale=args.scale,
                              seed=args.seed)
    print(f"[viz_trigger] {out}")
    for s in stats:
        print(f"  idx={s['index']:>5} ({s['label']:<10}) "
              f"xi_linf={s['xi']['linf']:.4f}  "
              f"delta_linf={s['delta']['linf']:.4f}  "
              f"total_linf={s['residual']['linf']:.4f} "
              f"({s['residual']['frac_of_eps']:.2f} x 4/255)")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
