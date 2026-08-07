"""诊断用的 FL 训练驱动：clean / attack 两种模式 + checkpoint 埋点。

# 为什么不直接改 main.py

原仓库的 ``main.py`` 是一个 ``if __name__ == "__main__"`` 脚本，无法作为模块
复用。本文件复刻它的构造流程（并逐项对齐 argparse 默认值），从而让
**原仓库文件保持零 diff、始终可 diff**。代价是两边可能漂移，
已在 ``diag/README.md`` 的"已知限制"中标注。

# 相对 main.py 的三处刻意差异（全部记录在 PATCHES.md）

1. **补齐随机性**：``diag.config.set_all_seeds`` 同时播种
   torch / numpy / python-random。原实现只播种 torch（utils.py:8-13），
   而数据划分完全依赖 ``np.random``，导致划分不可复现。
2. **隔离客户端采样的随机流**：用 ``diag.config.make_select_rule`` 的独立
   ``torch.Generator`` 取代 ``utils.random_select`` 的全局 RNG。
   否则攻击 run 额外消耗的 RNG（生成器初始化、每次 PGD 的随机起点）会让
   两种模式每轮选中的客户端集合完全不同，对照实验失效。
3. **训练结束时保存 checkpoint 与 meta**：挂在既有的 ``on_fl_end`` 事件上。

以上三项都**不触碰攻击语义**：投毒函数、生成器训练、PGD、聚合规则、
FedBN 逻辑全部原样调用原仓库实现。

# clean / attack 的唯一变量

两种模式共用同一份 seed、同一份数据划分、同一份客户端顺序、同一套模型初始化，
唯一的差别是"分片 90..99 的客户端是否投毒"。clean run 中这些客户端仍被标记为
``is_malicious_slot=True``，便于下游做同一批客户端的跨模式配对比较。
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from . import REPO_ROOT  # noqa: F401  (副作用：把仓库根目录加入 sys.path)
from .config import Cfg, load_config, make_select_rule, set_all_seeds
from .hooks import (build_client_meta, extract_generator, run_dir_name, save_run)

from client import BasicClient, PoisonClient
from event_emitter import fl_event_emitter
from fba import use_our_attack
from fl_process import basic_fl_process
from pfl import use_fedbn
from resnet import get_resnet
from server import BasicServer
from utils import client_inner_dirichlet_partition

__all__ = ["SyntheticImageDataset", "build_datasets", "run_fl", "main"]


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class SyntheticImageDataset(Dataset):
    """冒烟测试用的合成图像数据集（**无任何科学意义**）。

    存在的唯一理由：让整条流水线在没有 torchvision、没有 CIFAR-10 下载的环境里
    也能跑通。图像按类条件生成（每类一个随机基底 + 噪声），使得 2 轮训练也能
    学到一点结构，从而让特征/原型/边距等量不是纯噪声。

    标签是**严格类别均衡**的（而不是随机抽样），这样探针集的容量需求可以精确
    计算，不会因为某一类恰好抽少了而随机失败。
    """

    def __init__(self, n_samples: int, num_classes: int = 10, seed: int = 0,
                 image_size: int = 32, channels: int = 3, noise: float = 0.15):
        generator = torch.Generator().manual_seed(int(seed))
        per_class = max(1, int(n_samples) // int(num_classes))
        labels = torch.arange(num_classes).repeat_interleave(per_class)
        labels = labels[torch.randperm(len(labels), generator=generator)]
        basis = torch.rand(num_classes, channels, image_size, image_size,
                           generator=generator)
        noise_tensor = torch.randn(len(labels), channels, image_size, image_size,
                                   generator=generator) * float(noise)
        self.data = (basis[labels] + noise_tensor).clamp(0.0, 1.0)
        self.targets = labels.tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def build_datasets(cfg: Cfg, smoke: bool) -> Tuple[Dataset, Dataset, int]:
    """返回 ``(train_dataset, test_dataset, num_classes)``。

    ``smoke=True`` 用合成数据（不依赖 torchvision）；否则加载 CIFAR-10。
    """
    if smoke:
        smoke_cfg = cfg.smoke
        num_classes = int(smoke_cfg.num_classes)
        n_train = int(smoke_cfg.num_clients) * int(smoke_cfg.samples_per_client)
        n_test = int(smoke_cfg.num_clients) * int(smoke_cfg.test_samples_per_client)

        # 探针集从测试集抽，容量需求是**按类**算的，不是按总数：
        #   目标类需要 n_ref + n_target
        #   每个非目标类需要 n_other_per_class，外加分摊的 n_query
        # 数据集是类别均衡的，所以总数 = 每类需求 × 类别数即可（再留一点余量）。
        probe_cfg = smoke_cfg.probe
        n_other_classes = max(1, num_classes - 1)
        need_target_class = int(probe_cfg.n_ref) + int(probe_cfg.n_target)
        need_other_class = int(probe_cfg.n_other_per_class) + -(
            -int(probe_cfg.n_query) // n_other_classes)   # 向上取整
        per_class = max(need_target_class, need_other_class)
        n_test = max(n_test, (per_class + 4) * num_classes)
        return (SyntheticImageDataset(n_train, num_classes, seed=1234),
                SyntheticImageDataset(n_test, num_classes, seed=5678),
                num_classes)

    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "加载 CIFAR-10 需要 torchvision，但当前环境未安装。"
            "冒烟测试请改用 --smoke（使用合成数据，不依赖 torchvision）。") from exc

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    root = str(cfg.data.root)
    # 与 main.py:59-60 一致：只做 ToTensor，无归一化、无数据增强，像素 ∈ [0,1]。
    train_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                                transform=transform)
    return train_dataset, test_dataset, int(cfg.data.num_classes)


# ---------------------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------------------
def run_fl(cfg: Cfg, mode: str, alpha: float, seed: int, *, smoke: bool = False,
           device: Optional[str] = None,
           ckpt_root: Optional[str] = None) -> Path:
    """跑一次完整的 FL 训练并保存 checkpoint，返回 checkpoint 目录。"""
    if mode not in ("clean", "attack"):
        raise ValueError(f"mode 必须是 'clean' 或 'attack'，收到 '{mode}'")

    if smoke:
        smoke_cfg = cfg.smoke
        client_num = int(smoke_cfg.num_clients)
        bad_client_num = int(smoke_cfg.bad_client_num)
        select_per_round = int(smoke_cfg.select_per_round)
        local_steps = int(smoke_cfg.local_steps)
        total_round = int(smoke_cfg.rounds)
        batch_size = int(smoke_cfg.batch_size)
        device = device or str(smoke_cfg.device)
    else:
        fl_cfg = cfg.fl
        client_num = int(fl_cfg.client_num)
        bad_client_num = int(fl_cfg.bad_client_num)
        select_per_round = int(fl_cfg.select_per_round)
        local_steps = int(fl_cfg.local_steps)
        total_round = int(fl_cfg.total_round)
        batch_size = int(fl_cfg.batch_size)
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    torch_device = torch.device(device)
    target_class = int(cfg.data.target_class)
    poison_rate = float(cfg.fl.poison_rate)
    learning_rate = float(cfg.fl.learning_rate)

    # --- 1. 播种（torch + numpy + random 三条流） -------------------------
    seed_info = set_all_seeds(
        seed, cudnn_deterministic=bool(cfg.determinism.cudnn_deterministic),
        cudnn_benchmark=bool(cfg.determinism.cudnn_benchmark))

    # --- 2. 数据与划分 ----------------------------------------------------
    train_dataset, test_dataset, num_classes = build_datasets(cfg, smoke)
    train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
    test_labels = np.asarray(test_dataset.targets, dtype=np.int64)

    train_per_client = [len(train_dataset) // client_num] * client_num
    test_per_client = [len(test_dataset) // client_num] * client_num
    # class_priors 在 train/test 之间共用，与 main.py:67-77 一致，
    # 使同一客户端的训练/测试分布一致。
    class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=client_num)

    train_indices = client_inner_dirichlet_partition(
        train_labels, client_num, num_classes=num_classes, dir_alpha=alpha,
        client_sample_nums=list(train_per_client), class_priors=class_priors)
    test_indices = client_inner_dirichlet_partition(
        test_labels, client_num, num_classes=num_classes, dir_alpha=alpha,
        client_sample_nums=list(test_per_client), class_priors=class_priors)

    train_loaders = [
        DataLoader(train_dataset, batch_size=batch_size,
                   sampler=SubsetRandomSampler(train_indices[i]), drop_last=True)
        for i in range(client_num)]
    # 注意：训练 loader 保留 drop_last=True（与 main.py:80 一致，属于训练约定）。
    # 所有**评估**路径一律走 diag.probe.make_eval_loader（drop_last=False）。
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size,
                   sampler=SubsetRandomSampler(test_indices[i]), drop_last=True)
        for i in range(client_num)]

    # --- 3. 客户端 --------------------------------------------------------
    optimizer_factory = partial(torch.optim.SGD, lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    def model_factory():
        return get_resnet(size=int(cfg.fl.get("model_size", 10)),
                          num_classes=num_classes)

    clients: List[Any] = []
    n_benign = client_num - bad_client_num
    for i in range(client_num):
        model = model_factory().to(torch_device)
        is_slot_malicious = i >= n_benign
        # clean run 里恶意槽位的客户端也是普通 BasicClient —— 这正是唯一的变量。
        if is_slot_malicious and mode == "attack":
            client = PoisonClient(model, train_loaders[i], test_loaders[i],
                                  loss_func, optimizer_factory, poison_func=None)
            client.diag_is_malicious = True
            client.diag_poison_ratio = poison_rate
        else:
            client = BasicClient(model, train_loaders[i], test_loaders[i],
                                 loss_func, optimizer_factory)
            client.diag_is_malicious = False
            client.diag_poison_ratio = 0.0
        client.partition_idx = i
        client.diag_is_malicious_slot = bool(is_slot_malicious)
        client.diag_n_participations = 0
        clients.append(client)

    random.shuffle(clients)   # 与 main.py:95 一致，但现在是可复现的
    for idx, client in enumerate(clients):
        client.local_model.device = torch_device
        client.cid = idx

    # --- 4. 服务器与 PFL --------------------------------------------------
    global_model = model_factory().to(torch_device)
    server = BasicServer(global_model)
    server.global_model.device = torch_device
    server.agg_rule = "avg"
    if str(cfg.fl.pfl) == "fedbn":
        use_fedbn(server)

    # --- 5. 攻击配置（唯一的模式差异） ------------------------------------
    generator = None
    eval_func = None
    if mode == "attack":
        eval_func = use_our_attack(clients, server, target_class, poison_rate)
        generator = extract_generator(eval_func)

    # --- 6. 参与轮次计数（纯只读埋点，挂在既有事件上） --------------------
    def count_participation(**kwargs):
        clients[kwargs["client_indice"]].diag_n_participations += 1

    fl_event_emitter.on("on_client_begin", count_participation)

    # --- 7. 训练 ----------------------------------------------------------
    select_rule = make_select_rule(
        select_per_round, seed + int(cfg.determinism.select_rule_seed_offset))
    try:
        basic_fl_process(server, clients, local_steps=local_steps,
                         training_rounds=total_round, select_rule=select_rule)
    finally:
        fl_event_emitter.off("on_client_begin", count_participation)

    # --- 8. 保存 ----------------------------------------------------------
    ckpt_root = Path(ckpt_root or cfg.paths.ckpt_root)
    ckpt_dir = ckpt_root / run_dir_name(mode, alpha, seed)

    train_labels_by_client = {i: train_labels[train_indices[i]] for i in range(client_num)}
    test_labels_by_client = {i: test_labels[test_indices[i]] for i in range(client_num)}
    meta = {
        "mode": mode, "alpha": float(alpha), "seed": int(seed),
        "smoke": bool(smoke), "device": str(device),
        "num_classes": int(num_classes), "target_class": target_class,
        "client_num": client_num, "bad_client_num": bad_client_num,
        "select_per_round": select_per_round, "local_steps": local_steps,
        "total_round": total_round, "batch_size": batch_size,
        "poison_rate": poison_rate if mode == "attack" else 0.0,
        "seeds": seed_info,
        "select_rule_seed": seed + int(cfg.determinism.select_rule_seed_offset),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "clients": build_client_meta(clients, train_labels_by_client,
                                     test_labels_by_client, num_classes,
                                     target_class,
                                     train_indices_by_client=train_indices,
                                     test_indices_by_client=test_indices),
    }

    reference_batch = None
    if generator is not None:
        reference_batch = torch.stack(
            [test_dataset[i][0] for i in range(min(16, len(test_dataset)))]
        ).to(torch_device)

    save_run(ckpt_dir, server, clients, meta, generator, reference_batch)
    print(f"[run_fl] mode={mode} alpha={alpha} seed={seed} -> {ckpt_dir}")
    return ckpt_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="诊断用 FL 训练驱动（clean / attack）")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--mode", choices=["clean", "attack"], required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt-root", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="极小规模冒烟配置 + 合成数据（不依赖 torchvision）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_fl(cfg, args.mode, args.alpha, args.seed, smoke=args.smoke,
           device=args.device, ckpt_root=args.ckpt_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
