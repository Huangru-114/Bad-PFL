"""离线重算 exp1 的**社区口径 ASR**（target-排除），只用最终 checkpoint，**不重训**。

# 为什么需要它

`analysis_exp1` 里的 `asr_paper_*` 是 **unfiltered**（目标类样本计入分母），且
`asr_paper_all` 把攻击者自身（ASR≈1.0）也平均进去。社区惯例的 ASR 通常：
(1) 只看受害者（良性客户端），(2) **排除真实标签已是目标类的样本**。第 (2) 点
无法从已落盘的逐客户端标量推出（需要逐样本预测，CSV 没存），但可以从最终
checkpoint **离线前向重算**——`save_run` 存了全部 40 个客户端的 `client_<cid>.pt`
与 `generator.pt`（hooks.py），`meta.json` 存了每客户端的 `test_indices`。

# 口径与忠实性

触发器**逐字复刻原始 `fba.our_poison_func`**（不是 diag 的 perturb 重建）：

    perturbed = fba.pgd_attack(client_model, x, y) + generator(x) / 255. * 4.

在**该客户端自己的 test 分区**（`meta.json` 的 `test_indices` → `Subset`）上评估，
`model.eval()`，argmax==target 记命中。同时给出：

- ``asr_std_filtered``：**排除目标类**（分母只数真实标签≠target 的样本）—— 社区口径。
- ``asr_unfiltered``：不排除（分母全部）—— 与 CSV 的 `asr_paper_*` 同口径，**用作
  自检**：离线 unfiltered 应与该 run 最终轮 CSV 的 `asr_paper_benign/malicious`
  吻合（证明触发器重建 + loader 忠实），吻合后 filtered 才可信。

# 边界（诚实交代）

- 只有**最终轮**：逐轮曲线（E1-1/E1-2）需要逐轮逐客户端模型，没存，不在此列。
- ξ 每次评估现算（原始攻击定义），是"部署触发器"口径，不是纯权重驻留。
- 需要 torch + torchvision + CIFAR-10 数据 + checkpoint，在集群上跑。

用法::

    python -m diag.recompute_asr_final --ckpt-root ./checkpoints --glob '*e1_*'
    python -m diag.recompute_asr_final --ckpt-root ./checkpoints --glob '*e1_*' --execute
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import load_config

__all__ = ["aggregate_tiers", "client_asr", "recompute_run", "main"]


# ---------------------------------------------------------------------------
# 纯聚合（可脱离 torch/数据单测）
# ---------------------------------------------------------------------------
def aggregate_tiers(per_client: Sequence[Dict[str, Any]],
                    key: str) -> Dict[str, float]:
    """把逐客户端的某个 ASR 键聚合成 benign / all / malicious 三档均值。

    只对有限值求均值；某档为空则记 nan（不填 0，遵"无定义留空"铁律）。
    ``per_client`` 每项需含 ``is_malicious`` 与 ``key``。
    """
    def _mean(rows: List[Dict[str, Any]]) -> float:
        vals = [float(r[key]) for r in rows
                if r.get(key) is not None and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    benign = [r for r in per_client if not r["is_malicious"]]
    mal = [r for r in per_client if r["is_malicious"]]
    return {"benign": _mean(benign), "malicious": _mean(mal),
            "all": _mean(list(per_client))}


# ---------------------------------------------------------------------------
# 单客户端评估（torch）
# ---------------------------------------------------------------------------
def client_asr(model: Any, generator: Any, loader: Any, target_class: int,
               device: Any) -> Dict[str, float]:
    """在 loader 上用**原始触发器**评估 model 的 ASR（filtered + unfiltered）。

    触发器 = ``fba.pgd_attack(model, x, y) + generator(x)/255.*4.``，逐字对应
    ``fba.our_poison_func``（poison_ratio=1.0，无末端 clamp）。filtered 排除真实
    标签==target 的样本。评估后清梯度（pgd 会往 model 累加）。
    """
    import fba  # 原仓库模块，只读复用 pgd_attack，不重写

    target = int(target_class)
    was_training = model.training
    model.eval()
    gen_training = generator.training
    generator.eval()
    n_all = n_hit_all = n_elig = n_hit_elig = 0
    try:
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            # pgd 需要梯度；生成器只前向
            adv = fba.pgd_attack(model, images, labels.long())
            with torch.no_grad():
                trig = generator(images) / 255.0 * 4.0
                preds = model((adv + trig).detach()).argmax(dim=1)
            eligible = labels != target
            n_all += int(labels.numel())
            n_hit_all += int((preds == target).sum())
            n_elig += int(eligible.sum())
            n_hit_elig += int((preds[eligible] == target).sum())
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        if gen_training:
            generator.train()
    return {
        "asr_std_filtered": (n_hit_elig / n_elig if n_elig else float("nan")),
        "asr_unfiltered": (n_hit_all / n_all if n_all else float("nan")),
        "n_eligible": int(n_elig), "n_total": int(n_all),
    }


# ---------------------------------------------------------------------------
# 单个 run 的重算
# ---------------------------------------------------------------------------
def recompute_run(ckpt_dir: Path, test_dataset: Any, *, model_size: int,
                  device: Any, batch_size: int = 128) -> List[Dict[str, Any]]:
    """重算一个 checkpoint 目录里全部客户端的 filtered/unfiltered ASR。

    需要 ``meta.json``（target_class / num_classes / clients[].test_indices /
    dose）、``generator.pt``、``client_<cid>.pt``。缺任一则跳过并抛信息。
    """
    from torch.utils.data import DataLoader, Subset
    from resnet import get_resnet
    from generator import Autoencoder
    from .hooks import load_client_model

    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    target_class = int(meta["target_class"])
    num_classes = int(meta["num_classes"])
    bad_num = int(meta.get("bad_client_num", -1))
    poison_rate = float(meta.get("poison_rate", float("nan")))
    seed = int(meta.get("seed", -1))

    generator = Autoencoder().to(device)
    gen_state = torch.load(ckpt_dir / "generator.pt", map_location=device)
    generator.load_state_dict(gen_state)
    generator.device = device

    def factory():
        return get_resnet(size=int(model_size), num_classes=num_classes)

    rows: List[Dict[str, Any]] = []
    for record in meta["clients"]:
        cid = int(record["client_id"])
        test_indices = record.get("test_indices")
        if not test_indices:
            continue
        model = load_client_model(ckpt_dir / f"client_{cid}.pt", factory, device)
        loader = DataLoader(Subset(test_dataset, list(test_indices)),
                            batch_size=batch_size)
        asr = client_asr(model, generator, loader, target_class, device)
        del model
        rows.append({
            "run_id": ckpt_dir.name, "bad_client_num": bad_num,
            "poison_rate": poison_rate, "seed": seed,
            "client_id": cid, "is_malicious": bool(record["is_malicious"]),
            **asr,
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_test_dataset(data_root: str):
    import torchvision
    tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(data_root, train=False, download=False,
                                        transform=tf)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="离线重算 target-排除的社区口径 ASR（最终轮，不重训）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--glob", default="*e1_*",
                        help="ckpt-root 下匹配 run 目录的 glob（默认 exp1 的 e1_*）")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--model-size", type=int, default=None,
                        help="ResNet 规模；缺省取 config 的 exp1.model_size")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="results/exp1_final_asr_standard.csv")
    parser.add_argument("--execute", action="store_true",
                        help="真跑；缺省只打印将处理的 run 清单")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model_size = (int(args.model_size) if args.model_size is not None
                  else int(cfg.exp1.model_size))
    dirs = sorted(Path(p) for p in globlib.glob(
        str(Path(args.ckpt_root) / args.glob))
        if (Path(p) / "meta.json").exists()
        and (Path(p) / "generator.pt").exists())

    print(f"=== 离线重算社区口径 ASR：匹配到 {len(dirs)} 个 run 目录 ===")
    for d in dirs:
        print(f"  {d}")
    if not args.execute:
        print(f"\n[dry-run] 未执行。加 --execute 真跑（model_size={model_size}）。")
        return 0

    device = torch.device("cpu" if args.device == "cpu"
                          else f"cuda:{args.device}")
    test_dataset = _load_test_dataset(args.data_root)

    import pandas as pd
    all_rows: List[Dict[str, Any]] = []
    for d in dirs:
        rows = recompute_run(d, test_dataset, model_size=model_size,
                             device=device, batch_size=args.batch_size)
        all_rows.extend(rows)
        f = aggregate_tiers(rows, "asr_std_filtered")
        u = aggregate_tiers(rows, "asr_unfiltered")
        print(f"[{d.name}] filtered benign/all/mal = "
              f"{f['benign']:.3f}/{f['all']:.3f}/{f['malicious']:.3f}  |  "
              f"unfiltered(自检 vs CSV asr_paper) benign/mal = "
              f"{u['benign']:.3f}/{u['malicious']:.3f}")

    frame = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\n逐客户端 -> {out}（{len(frame)} 行）")

    # 附带出一张 filtered benign 的剂量-响应热力图（复用 analysis_exp1.plot_e1_5，
    # 与 unfiltered 的 E1-5 并列对照）。绘图失败不影响 CSV。
    try:
        from .analysis_exp1 import plot_e1_5
        resp_rows = []
        for run_id, group in frame.groupby("run_id"):
            first = group.iloc[0]
            resp_rows.append({
                "bad_client_num": int(first["bad_client_num"]),
                "poison_rate": float(first["poison_rate"]),
                "seed": int(first["seed"]),
                "asr": aggregate_tiers(group.to_dict("records"),
                                       "asr_std_filtered")["benign"],
            })
        fig_out = out.parent / "figs" / "exp1_E5b_dose_heatmap_filtered_benign.png"
        plot_e1_5(pd.DataFrame(resp_rows), fig_out)
        print(f"filtered benign 剂量-响应热力图 -> {fig_out}")
    except Exception as exc:               # 绘图非核心，失败仅提示
        print(f"（跳过附带热力图：{exc}）")

    print("先用 unfiltered 列与 CSV 的 asr_paper_benign/malicious 对拍确认忠实，"
          "再采 filtered 三档。")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
