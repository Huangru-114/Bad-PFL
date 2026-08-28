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
3. **保存 checkpoint 与 meta**：训练结束时保存一次；实验 F 另需按轮次的快照，
   由 ``diag.snapshots.SnapshotRecorder`` 挂在既有的 ``on_client_end`` /
   ``on_round_end`` 事件上（``--snapshot-every``）。两者都只做 I/O ——
   不消耗 RNG、不触碰 optimizer，因此开不开快照的 run 应当逐位一致，
   ``--verify-against`` 就是用来实证这一点的。

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
from .hooks import (attach_generator_checkpoint_hook, build_client_meta,
                    compare_run_checkpoints, extract_generator, run_dir_name,
                    save_generator_meta, save_run)
from .snapshots import (SnapshotRecorder, build_grid, select_snapshot_clients)
from .defenses import ABLATION_VARIANTS, DEFENSES, build_defense, use_defense
from .instrumentation import RoundRecorder
from .track import TrainingTracker

from client import BasicClient, PoisonClient
from event_emitter import fl_event_emitter
from fba import use_our_attack
from fl_process import basic_fl_process
from pfl import use_fedbn

from .schedule import SCHEDULES, AttackSchedule, gate_attack
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


def build_datasets(cfg: Cfg, smoke: bool,
                   dataset_sizes: Optional[Tuple[int, int]] = None
                   ) -> Tuple[Dataset, Dataset, int]:
    """返回 ``(train_dataset, test_dataset, num_classes)``。

    ``smoke=True`` 用合成数据（不依赖 torchvision）；否则加载 CIFAR-10。

    ``dataset_sizes``：显式指定 ``(n_train, n_test)``，仅对合成数据有意义。
    离线分析 checkpoint 时**必须**传入 meta.json 记录的 ``synthetic_sizes``，
    否则重建出的合成数据集与训练时不一致，``train_indices`` 会越界或（更糟）
    指向不同的图片。CIFAR-10 路径固定，不受影响。
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
        if dataset_sizes is not None:
            n_train, n_test = int(dataset_sizes[0]), int(dataset_sizes[1])
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
           ckpt_root: Optional[str] = None,
           snapshot_every: Optional[int] = None,
           verify_against: Optional[str] = None,
           defense: str = "fedavg",
           defense_tau: Optional[float] = None,
           defense_trim_alpha: Optional[float] = None,
           eval_every: int = 0,
           instrument_dir: Optional[str] = None,
           results_dir: Optional[str] = None,
           bad_client_num: Optional[int] = None,
           total_round: Optional[int] = None,
           run_tag: str = "",
           oracle_inner: str = "fedavg",
           eval_include_malicious: bool = False,
           client_num: Optional[int] = None,
           select_per_round: Optional[int] = None,
           local_steps: Optional[int] = None,
           poison_rate: Optional[float] = None,
           model_size: Optional[int] = None,
           layer_metrics: bool = False,
           schedule: Optional["AttackSchedule"] = None,
           generator_online_from: Optional[int] = None,
           freeze_trigger_eval: bool = False) -> Path:
    """跑一次完整的 FL 训练并保存 checkpoint，返回 checkpoint 目录。

    Parameters
    ----------
    snapshot_every:
        实验 F 用的按轮次快照间隔。``None`` 取 ``cfg.exp_f.snapshot_every``，
        ``0`` 或负数表示关闭。开启后每 N 轮把全局模型、抽样客户端的本地模型
        与生成器存到 ``{ckpt_dir}/round_XXXX/``，并写 ``snapshot_manifest.json``。
    verify_against:
        另一个 run 目录。训练结束后逐位比对两边的 ``global.pt`` /
        ``generator.pt`` / ``client_*.pt``，用来实证"快照埋点只做 I/O、
        不改变训练动力学"。比对结果只打印**不中止** —— 分歧本身是要报告的
        结果，不是错误。
    defense:
        服务器端聚合规则（实验 I / J）。``"fedavg"`` 时**完全不接管**
        ``agg_and_update``，走原仓库的 ``agg_avg`` —— 这样"无防御"这一组与
        既有的实验 A–F 结果严格同源，不会因为换了一份等价实现而产生细微差别。
        其余取值见 ``diag.defenses.DEFENSES`` 与 ``ABLATION_VARIANTS``。
    eval_every:
        每多少轮评估一次 ACC/ASR（全局 + 个性化，实验 I §5.5 / 实验 J §5）。
        ``0`` 关闭。全局模型一律用**借 BN** 的版本，另存 ``acc_global_raw``
        暴露 FedBN 下原始全局模型的退化程度。
    instrument_dir:
        逐轮 npz 的落盘根目录（实验 I §5.3 / 实验 J §2.2）。``None`` 关闭。
    """
    if mode not in ("clean", "attack"):
        raise ValueError(f"mode 必须是 'clean' 或 'attack'，收到 '{mode}'")

    # CLI 的显式覆盖优先于 config；两者都没有才用 config 的默认值。
    # 每个覆盖量都先存进 *_override，因为下面的 smoke / 正式两个分支会
    # 无条件地把同名局部变量写掉。
    bad_override, round_override = bad_client_num, total_round
    client_override, select_override = client_num, select_per_round
    steps_override, poison_override = local_steps, poison_rate
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

    # 规模参数要在 bad_client_num 校验之前生效，否则会拿旧的 client_num 去校验
    if client_override is not None:
        client_num = int(client_override)
    if select_override is not None:
        select_per_round = int(select_override)
    if steps_override is not None:
        local_steps = int(steps_override)
    if select_per_round > client_num:
        raise ValueError(
            f"select_per_round={select_per_round} > client_num={client_num}")

    if bad_override is not None:
        if not 0 <= int(bad_override) <= client_num:
            raise ValueError(
                f"bad_client_num={bad_override} 超出 [0, {client_num}]")
        bad_client_num = int(bad_override)
    if round_override is not None:
        total_round = int(round_override)
    if mode == "clean" and bad_client_num and bad_override is not None:
        print(f"[run_fl] 注意：clean 模式下 bad_client_num={bad_client_num} "
              f"只影响'恶意槽位'的标记，不会投毒")

    torch_device = torch.device(device)
    target_class = int(cfg.data.target_class)
    poison_rate = (float(poison_override) if poison_override is not None
                   else float(cfg.fl.poison_rate))
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate={poison_rate} 必须在 [0, 1] 内")
    learning_rate = float(cfg.fl.learning_rate)
    model_size = int(model_size if model_size is not None
                     else cfg.fl.get("model_size", 10))
    schedule = schedule if schedule is not None else AttackSchedule("continuous")

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
        return get_resnet(size=model_size, num_classes=num_classes)

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
    generator_handler = None
    ckpt_root = Path(ckpt_root or cfg.paths.ckpt_root)
    # run_tag 让"只改了 bad_client_num / total_round"的扫描各自落在独立目录，
    # 否则不同恶意密度的 run 会互相覆盖 —— 而且覆盖是静默的。
    base_name = run_dir_name(mode, alpha, seed) + (f"_{run_tag}" if run_tag else "")
    ckpt_dir = ckpt_root / base_name
    if mode == "attack":
        # clean run 完全跳过 use_our_attack —— 不是构造空的攻击对象。
        # 原实现在没有 PoisonClient 时会 UnboundLocalError（eval_func 未赋值），
        # 见 fba.py:60-66；跳过是唯一正确的处理方式。
        eval_func = use_our_attack(clients, server, target_class, poison_rate)
        generator = extract_generator(eval_func)
        every = int(cfg.exp_e.get("generator_ckpt_every", 0)) if "exp_e" in cfg else 0
        if every > 0:
            generator_handler = attach_generator_checkpoint_hook(
                lambda: generator, ckpt_dir / "generator", every_n_rounds=every)

    # --- 6. 参与轮次计数（纯只读埋点，挂在既有事件上） --------------------
    def count_participation(**kwargs):
        clients[kwargs["client_indice"]].diag_n_participations += 1

    fl_event_emitter.on("on_client_begin", count_participation)

    # --- 6b. 实验 F 的按轮次快照（同样只做 I/O，不消耗 RNG） ---------------
    exp_f_cfg = cfg.exp_f if "exp_f" in cfg else None
    if snapshot_every is None:
        snapshot_every = (int(exp_f_cfg.snapshot_every)
                          if exp_f_cfg is not None else 0)
    snapshot_every = int(snapshot_every)
    recorder = None
    snapshot_client_ids: List[int] = []
    if snapshot_every > 0:
        grid = build_grid(total_round, snapshot_every)
        snapshot_client_ids = select_snapshot_clients(
            clients,
            n_benign=int(exp_f_cfg.snapshot_n_benign) if exp_f_cfg else 10,
            n_malicious=int(exp_f_cfg.snapshot_n_malicious) if exp_f_cfg else 2,
            seed=int(exp_f_cfg.snapshot_select_seed) if exp_f_cfg else 4242)
        recorder = SnapshotRecorder(
            ckpt_dir, grid, snapshot_client_ids,
            generator_getter=(lambda: generator) if generator is not None else None)
        recorder.attach()
        print(f"[run_fl] 按轮次快照已开启：{len(grid)} 个网格点 "
              f"× {len(snapshot_client_ids)} 个客户端 -> {ckpt_dir}/round_XXXX/")

    # --- 6c. 防御与逐轮埋点（实验 I / J） ---------------------------------
    run_id = f"{defense}_{base_name}"
    tracker = None
    restore_defense = None
    if defense != "fedavg" or eval_every > 0 or instrument_dir:
        exp_i_cfg = cfg.exp_i if "exp_i" in cfg else None
        tau = (defense_tau if defense_tau is not None
               else (float(exp_i_cfg.tau) if exp_i_cfg else 0.2))
        trim_alpha = (defense_trim_alpha if defense_trim_alpha is not None
                      else (float(exp_i_cfg.trim_alpha) if exp_i_cfg else 0.25))

        benign_ids = sorted(int(c.cid) for c in clients
                            if not bool(getattr(c, "diag_is_malicious", False)))
        malicious_ids = sorted(int(c.cid) for c in clients
                               if bool(getattr(c, "diag_is_malicious", False)))
        n_eval = min(10, len(benign_ids))
        probe = None
        if eval_every > 0:
            from .exp_e import build_exp_e_probe
            probe_cfg = cfg
            if smoke:
                probe_cfg = Cfg({**cfg, "exp_e": {
                    **cfg["exp_e"], "probe_n": int(cfg.smoke.exp_e_probe_n)}})
            probe = build_exp_e_probe(test_dataset, target_class, probe_cfg)
            cfg_for_eval = probe_cfg
        else:
            cfg_for_eval = cfg

        tracker = TrainingTracker(
            cfg=cfg_for_eval,
            recorder=(RoundRecorder(instrument_dir, run_id) if instrument_dir
                      else None),
            device=torch_device, defense_name=defense, variant=defense,
            tau=tau, trim_alpha=trim_alpha, alpha_dirichlet=float(alpha),
            seed=int(seed), bad_client_num=int(bad_client_num),
            poison_rate=float(poison_rate) if mode == "attack" else 0.0,
            schedule_kind=schedule.kind, layer_metrics=bool(layer_metrics),
            model_size=model_size,
            eval_every=int(eval_every),
            eval_client_ids=benign_ids[:n_eval],
            # 正对照：恶意客户端**自己**的模型是被投毒的，其 ASR 应当很高。
            # 少了它，"良性 ASR 在基线"既可能是排除成功、也可能是投毒根本没生效。
            eval_malicious_ids=(malicious_ids if eval_include_malicious else ()),
            generator_getter=((lambda: generator) if generator is not None
                              else None),
            # 论文口径 ASR 的评估函数：use_our_attack 返回的 eval_func
            # （poison_ratio=1.0 的 full_poison_func）。clean run 为 None。
            paper_eval_func=(eval_func if mode == "attack" else None),
            # B 线：停攻点快照触发器图、之后复用（纯权重驻留）。只对 persist 跑开。
            track_frozen_trigger=bool(freeze_trigger_eval and mode == "attack"),
            probe=probe, target_class=target_class, num_classes=num_classes)
        tracker.attach(server, clients)

        # fedavg 默认**不接管** agg_and_update，走原仓库的 agg_avg，
        # 以保证"无防御"这一组与既有的实验 A–F 结果严格同源。
        # 但要逐轮埋点就必须拿到聚合前的各客户端更新，而 `before_update_global`
        # 在 agg_avg **之后**才触发（server.py:32-33）—— 拿不到。
        # 所以开了 --instrument-dir 时 fedavg 也走 FedAvg 类；两者数值等价，
        # 由 test_fedavg_matches_repo_agg_avg 保证。
        needs_takeover = defense != "fedavg" or bool(instrument_dir)
        if needs_takeover:
            # oracle_exclude 要按真实标签剔除，因此需要当轮选中客户端的恶意掩码。
            # 掩码的顺序必须与 fl_process.py:36 上传 state_dict 的顺序一致 ——
            # 两者都来自同一个 client_indices，OracleExclude 内部还会校验长度。
            def _malicious_mask():
                return np.array(
                    [bool(getattr(clients[i], "diag_is_malicious", False))
                     for i in tracker.selected_indices], dtype=bool)

            rule = build_defense(defense, tau=tau, trim_alpha=trim_alpha,
                                 seed=seed, inner=oracle_inner,
                                 malicious_getter=_malicious_mask)
            restore_defense = use_defense(
                server, rule,
                on_round=lambda outcome, states, w_prev: tracker.on_aggregate(
                    outcome, states, w_prev, clients))
            note = ""
            if defense == "fedavg":
                note = "（为逐轮埋点接管聚合；与 agg_avg 数值等价）"
            elif defense == "oracle_exclude":
                note = f"（inner={oracle_inner}；按真实标签完美剔除，不是可实现的防御）"
            # 只打印这个防御**真的会读**的参数。以前无条件打印 tau/trim_alpha，
            # 于是 FLAME / Multi-Krum 的日志看起来像是配了这两个值，
            # 而它们根本不读 —— 日志会被当成配置凭据，不能这样写。
            if defense in ("invariant", "invariant_mask_only",
                           "invariant_trim_only"):
                params = f"（tau={tau}, trim_alpha={trim_alpha}）"
            elif defense == "flame":
                params = f"（min_cluster_size=N//2+1, noise_seed={seed}）"
            else:
                params = ""
            print(f"[run_fl] 防御 = {defense}{params}{note}")

    # --- 6b. 攻击调度（实验 1B） -------------------------------------------
    # 装在 use_our_attack 之后：门控的是它设好的 poison_func 与生成器 hook。
    restore_schedule = None
    schedule_handler = None
    if mode == "attack" and schedule.kind != "continuous":
        if schedule.kind == "after_mta" and int(eval_every) <= 0:
            raise ValueError(
                "after_mta 调度需要 --eval-every > 0：MTA 只在评估轮才算得出来，"
                "关掉评估就永远触发不了，而那会静默地退化成'从不攻击'。")
        _state = {"round": 0}

        def _mta():
            return tracker.last_mta if tracker is not None else None

        # C 线：给生成器一个独立的"在线"窗口（[generator_online_from, ∞)），
        # 让 δ 在衰减段继续更新，而投毒仍只在 schedule 的窗口内。
        gen_schedule = None
        if generator_online_from is not None and int(generator_online_from) > 0:
            gen_schedule = AttackSchedule("late", start=int(generator_online_from))
        restore_schedule = gate_attack(clients, schedule,
                                       lambda: _state["round"], _mta,
                                       generator_schedule=gen_schedule)

        def _on_round_begin_schedule(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            _state["round"] = int(kwargs["cur_round"])
            active = schedule.active_for_round(_state["round"], _mta())
            if tracker is not None:
                tracker.attack_active = active

        fl_event_emitter.on("on_round_begin", _on_round_begin_schedule)
        schedule_handler = _on_round_begin_schedule
        expected = schedule.expected_attack_rounds(total_round)
        print(f"[run_fl] 攻击调度 = {schedule.describe()}"
              + (f"，共 {expected}/{total_round} 轮投毒"
                 if expected is not None else "，投毒轮数取决于训练过程"))

    # --- 7. 训练 ----------------------------------------------------------
    select_rule = make_select_rule(
        select_per_round, seed + int(cfg.determinism.select_rule_seed_offset))
    try:
        basic_fl_process(server, clients, local_steps=local_steps,
                         training_rounds=total_round, select_rule=select_rule)
    finally:
        if schedule_handler is not None:
            fl_event_emitter.off("on_round_begin", schedule_handler)
        if restore_schedule is not None:
            restore_schedule()
        fl_event_emitter.off("on_client_begin", count_participation)
        if generator_handler is not None:
            fl_event_emitter.off("on_round_end", generator_handler)
        if recorder is not None:
            recorder.detach()
            manifest_path = recorder.write_manifest()
            gaps = recorder.missing()
            # missing() 的粒度是 **(客户端, 网格点) 组合**，不是网格点。
            # 旧文案写成"N 个网格点缺快照"，于是 120 格里缺 9 格
            # 会被读成"10 个网格点里缺了 9 个"—— 差一个数量级，
            # 足以让人以为实验 F 没数据可跑。分子分母一起报。
            n_cells = len(recorder.grid) * len(recorder.client_ids)
            print(f"[run_fl] 快照清单 -> {manifest_path}"
                  + (f"（{n_cells} 个 (客户端, 网格点) 组合里缺 {len(gaps)} 个 = "
                     f"覆盖 {1 - len(gaps) / max(n_cells, 1):.1%}，"
                     f"见 manifest 的 'missing'）" if gaps
                     else f"（{n_cells} 个 (客户端, 网格点) 组合全部齐备）"))
        if restore_defense is not None:
            restore_defense()
        if tracker is not None:
            tracker.detach()

    # --- 8. 保存 ----------------------------------------------------------
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
        "synthetic_sizes": ([len(train_dataset), len(test_dataset)]
                            if smoke else None),
        "snapshot_every": snapshot_every,
        "snapshot_client_ids": snapshot_client_ids,
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
    if generator is not None:
        # 让任何一份生成器 checkpoint 都能独立追溯到它的训练条件。
        save_generator_meta(ckpt_dir / "generator", {
            "target_label": target_class,
            "epsilon": float(cfg.perturb.eps_xi),
            "sigma": float(cfg.perturb.eps_delta),
            "total_rounds": total_round,
            "seed": int(seed),
            "alpha": float(alpha),
            "run_id": base_name,
        })
    print(f"[run_fl] mode={mode} alpha={alpha} seed={seed} -> {ckpt_dir}")

    if tracker is not None and tracker.implantation_rows:
        frame = tracker.implantation_frame()
        out_dir = Path(results_dir or (Path(cfg.paths.results_root) / "raw"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"exp_ij_implantation_{run_id}.csv"
        frame.to_csv(out, index=False)
        print(f"[run_fl] 植入过程 {len(frame)} 行 -> {out}")

        # 逐 edge 的行另存一份：均值会掩盖分布，而"每个 edge 的攻击成功率"
        # 正是要看分布的（10 个 1.0 + 80 个 0.0 与全部 0.11 的均值一样）。
        edges = tracker.edge_frame()
        if not edges.empty:
            edge_out = out_dir / f"exp_ij_edge_{run_id}.csv"
            edges.to_csv(edge_out, index=False)
            print(f"[run_fl] 逐 edge {len(edges)} 行 -> {edge_out}")

    if verify_against:
        ok, report = compare_run_checkpoints(Path(verify_against), ckpt_dir)
        print(report)
        if ok:
            print("[run_fl] ✅ 与旧 run 逐位一致 —— 快照埋点确实没有改变训练"
                  "动力学，实验 E 在旧 run 上的结论可直接沿用。")
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
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help="实验 F 的按轮次快照间隔；0 关闭，"
                             "缺省取 config 的 exp_f.snapshot_every")
    parser.add_argument("--verify-against", default=None,
                        help="另一个 run 目录；训练结束后逐位比对 checkpoint，"
                             "用于实证快照埋点未改变训练动力学")
    parser.add_argument("--defense", default="fedavg",
                        choices=list(DEFENSES) + list(ABLATION_VARIANTS),
                        help="服务器端聚合规则；fedavg 走原仓库的 agg_avg")
    parser.add_argument("--defense-tau", type=float, default=None,
                        help="Invariant Aggregator 的 τ；缺省取 config")
    parser.add_argument("--defense-trim-alpha", type=float, default=None,
                        help="trimmed-mean 的 α；缺省取 config")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="每 N 轮评估 ACC/ASR（全局 + 个性化）；0 关闭")
    parser.add_argument("--instrument-dir", default=None,
                        help="逐轮 npz 的落盘根目录；不给则不记录")
    parser.add_argument("--results-dir", default=None,
                        help="植入过程 CSV 的输出目录；缺省 results/raw")
    parser.add_argument("--bad-client-num", type=int, default=None,
                        help="覆盖 config 的恶意客户端数（实验 I §4.3 / "
                             "论文的恶意密度消融）。改了它务必同时给 --run-tag，"
                             "否则不同密度的 run 会静默互相覆盖")
    parser.add_argument("--total-round", type=int, default=None,
                        help="覆盖 config 的总轮数")
    parser.add_argument("--run-tag", default="",
                        help="附加到 checkpoint 目录名与 run_id 后的后缀，"
                             "例如 bad1 -> attack_a0.5_s0_bad1")
    parser.add_argument("--oracle-inner", default="fedavg",
                        choices=[d for d in DEFENSES if d != "oracle_exclude"],
                        help="--defense oracle_exclude 剔除恶意后用哪个规则聚合"
                             "剩下的；缺省 fedavg（攻击者已完美剔除，"
                             "再叠一层防御会把两件事混在一起）")
    parser.add_argument("--eval-include-malicious", action="store_true",
                        help="额外评估恶意客户端**自己**的模型（正对照）。"
                             "静默恶意那一组必须开 —— 否则'良性 ASR 在基线'"
                             "既可能是排除成功、也可能是投毒没生效")
    # --- 攻击剂量与规模（实验 1） ---
    parser.add_argument("--client-num", type=int, default=None,
                        help="客户端总数（覆盖 config 的 100）")
    parser.add_argument("--select-per-round", type=int, default=None,
                        help="每轮选中的客户端数")
    parser.add_argument("--local-steps", type=int, default=None,
                        help="本地训练步数（注意是 step 不是 epoch）")
    parser.add_argument("--poison-rate", type=float, default=None,
                        help="投毒比例 ρ_p ∈ [0,1]。注意它是**逐样本的伯努利"
                             "概率**（fba.py:49 的 torch.rand），不是精确比例")
    parser.add_argument("--model-size", type=int, default=None,
                        choices=[10, 18, 34, 50, 101, 152],
                        help="ResNet 规模，默认取 config 的 fl.model_size")
    parser.add_argument("--layer-metrics", action="store_true",
                        help="逐层更新范数与逐层余弦。要多扫一遍全部 key，"
                             "ResNet-18 上明显变慢，默认关闭")
    # --- 攻击调度（实验 1B） ---
    parser.add_argument("--attack-schedule", default="continuous",
                        choices=list(SCHEDULES))
    parser.add_argument("--attack-start", type=int, default=0)
    parser.add_argument("--attack-end", type=int, default=0)
    parser.add_argument("--attack-length", type=int, default=0)
    parser.add_argument("--attack-period", type=int, default=0)
    parser.add_argument("--attack-duty", type=int, default=0)
    parser.add_argument("--attack-mta-threshold", type=float,
                        default=float("nan"))
    # --- 触发器冻结对比（实验 1B 的 B/C 线） ---
    parser.add_argument("--generator-online-from", type=int, default=None,
                        help="C 线：从这一轮起让生成器继续在线更新（投毒仍受 "
                             "--attack-* 窗口约束）。缺省则停攻即冻结 δ（A/B 线）")
    parser.add_argument("--freeze-trigger-eval", action="store_true",
                        help="B 线：停攻点快照 (x+ξ+δ) 评估图、之后复用，另存 "
                             "asr_paper_frozen_*，量纯权重驻留")
    args = parser.parse_args(argv)

    schedule = AttackSchedule(
        kind=args.attack_schedule, start=args.attack_start,
        end=args.attack_end, length=args.attack_length,
        period=args.attack_period, duty=args.attack_duty,
        mta_threshold=args.attack_mta_threshold)

    cfg = load_config(args.config)
    run_fl(cfg, args.mode, args.alpha, args.seed, smoke=args.smoke,
           device=args.device, ckpt_root=args.ckpt_root,
           snapshot_every=args.snapshot_every,
           verify_against=args.verify_against,
           defense=args.defense, defense_tau=args.defense_tau,
           defense_trim_alpha=args.defense_trim_alpha,
           eval_every=args.eval_every, instrument_dir=args.instrument_dir,
           results_dir=args.results_dir, bad_client_num=args.bad_client_num,
           total_round=args.total_round, run_tag=args.run_tag,
           oracle_inner=args.oracle_inner,
           eval_include_malicious=args.eval_include_malicious,
           client_num=args.client_num, select_per_round=args.select_per_round,
           local_steps=args.local_steps, poison_rate=args.poison_rate,
           model_size=args.model_size, layer_metrics=args.layer_metrics,
           schedule=schedule,
           generator_online_from=args.generator_online_from,
           freeze_trigger_eval=args.freeze_trigger_eval)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
