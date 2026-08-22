"""实验 F：冻结生成器的时间衰减矩阵。

# 要回答什么

实验 E 已经证伪"δ 是目标类自然特征"的**强版本**：在独立训练的干净模型上
``delta_only`` 的 ASR 是 2.72%，干净基线 2.02%。剩下两个互斥的弱假说：

======================  ========================================  =================
假说                     内容                                       预测
======================  ========================================  =================
**H_stable**            δ 稳定存在于**这条轨迹**的共享参数中        冻结的生成器在后续轮次仍有效
**H_coadapt**           δ 是与当时那个 θ_g 协同适应出的非鲁棒方向    随模型漂移快速失效
======================  ========================================  =================

区分二者的唯一办法：**把生成器冻结，让模型继续走。** 这在真实攻击中永远不会
发生 —— ``fba.trigger_gen_trainer`` 挂在 ``before_local_training`` 上
（fba.py:62），恶意客户端每次被选中都会对着刚下发的 θ_g 重新训练 30 步。

若 H_coadapt 成立，论文 Section 4.4 / Appendix D 关于"后门持久性源于自然特征"
的解释被推翻，换成"生成器持续追踪 θ_g 漂移"即可同时解释 Table 3 与 Table 21，
且不需要"自然特征"这个前提。**两个结果都有价值，本模块不得偏向任何一个。**

# 矩阵

对所有 ``t >= s`` 的组合（上三角，含对角线）：用生成器快照 ``G_s`` 生成 δ
（**冻结，不做任何再优化**），在模型快照 ``θ_t`` 上评估 ``asr_targeted``。
``t < s`` 的格子无意义（用未来的生成器打过去的模型），**不计算、不填 0**。

# ξ 的处理（§2.2）

主指标只测 ``delta_only``。ξ 是每次从当前模型现算的闭式解
（``pgd_attack``，``num_iter=1``，无跨轮状态），不存在"冻结"这个概念；
若让它在 θ_t 上现算，测到的就是"δ 漂移"与"ξ 重算"的混合效应。

=================  ==========================  ==================================
模式                ξ 来源                       用途
=================  ==========================  ==================================
``delta_only``     无                           **主指标**
``full_xi_frozen`` 由 θ_s 算出后冻结             次要：完整触发器的整体漂移
``full_xi_fresh``  由 θ_t 现算                   参照：最接近真实攻击条件
``none``           —                            每个 θ_t 的基线，**必须逐 t 重算**
``random_eps8``    —                            搭车记录的负对照
=================  ==========================  ==================================

``none`` 与 ``random_eps8`` 与 s 无关，只随 t 变，因此每个 t 只算一次，
CSV 里 ``gen_round_s`` / ``gap`` 留空。

# 评估对象（两种，分开报告）

- ``personalized``：抽样客户端的本地模型 —— 与论文的 ASR 定义、与实验 E 的
  协议都一致。**主实验。**
- ``global_bn_borrowed``：全局共享参数 + 固定良性客户端的 BN。

**不能直接用 ``global.pt``**：FedBN 下 ``pfl.py:5-12`` 把全部 BN 键从
``server.update`` 里 pop 掉，全局模型的 BN 从初始化起一位没变
（``exp_f_precheck.diagnose_global_bn`` 会实证这一点）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import REPO_ROOT  # noqa: F401
from .config import Cfg, load_config
from .exp_common import resolve_device
from .fedbn import is_fedbn_private_key
from .exp_e import _metrics, _predict, build_exp_e_probe
from .hooks import load_checkpoint, load_meta, run_dir_name, state_dict_hash
from .perturb import apply_perturbation, make_xi_fn
from .probe import make_eval_loader
from .run_fl import build_datasets
from .snapshots import available_rounds, load_manifest, round_dir_name

from generator import Autoencoder
from resnet import get_resnet

__all__ = ["DELTA_SCALE", "EXP_F_MODES", "MODES_PER_T", "FrozenGenerator",
           "compute_delta", "make_frozen_xi_fn", "load_snapshot_model",
           "run_matrix", "main"]

# 陷阱 6：δ 的缩放系数**硬编码**，不从 config 读。
# 数值来自 fba.py:54 的 `trigger_gen(data) / 255. * 4.`。加载旧生成器快照后
# 这个系数必须与它训练时一致；从 config 读会让配置漂移静默改变历史结果的含义。
# 启动时仍会与 config 比对，不一致就**响亮报错**（见 assert_delta_scale）。
DELTA_SCALE = 4.0 / 255.0

EXP_F_MODES: Tuple[str, ...] = ("delta_only", "full_xi_frozen", "full_xi_fresh")
MODES_PER_T: Tuple[str, ...] = ("none", "random_eps8")


def assert_delta_scale(cfg: Cfg) -> None:
    configured = float(cfg.perturb.eps_delta)
    if abs(configured - DELTA_SCALE) > 1e-12:
        raise ValueError(
            f"config 的 perturb.eps_delta={configured!r} 与 fba.py:54 硬编码的 "
            f"4/255={DELTA_SCALE!r} 不一致。实验 F 使用硬编码值（陷阱 6），"
            f"但这个分歧说明配置已经漂移，必须先查清楚再跑。")


# ---------------------------------------------------------------------------
# 冻结的生成器
# ---------------------------------------------------------------------------
class FrozenGenerator:
    """加载某一轮的生成器快照，并保证它在整个使用过程中一位都不变。

    陷阱 1 要求：``eval()`` + ``requires_grad_(False)``，且使用前后做参数哈希
    校验，不一致立即中止。这里把校验做成上下文管理器，**不依赖调用方记得校验**。
    """

    def __init__(self, path: Path, device):
        self.path = Path(path)
        self.round = _round_of(self.path)
        module = Autoencoder()
        module.load_state_dict(load_checkpoint(self.path), strict=True)
        module.to(device)
        module.eval()
        module.requires_grad_(False)
        self.module = module
        self.hash = state_dict_hash(module)

    def __enter__(self) -> "FrozenGenerator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # 已经有异常在往外抛时不要再抛：那会把真正的错因替换成这里的校验失败。
        if exc_type is None:
            self.verify()

    def verify(self) -> None:
        current = state_dict_hash(self.module)
        if current != self.hash:
            raise RuntimeError(
                f"生成器快照 {self.path} 在评估过程中被改动了！\n"
                f"  载入时: {self.hash}\n  当前:   {current}\n"
                f"实验 F 的全部前提是生成器**冻结**，此处必须中止（陷阱 1）。")
        if self.module.training:
            raise RuntimeError(
                f"生成器 {self.path} 处于 train 模式 —— BatchNorm 会用 batch "
                f"统计量并更新 running stats，δ 不再是冻结的（陷阱 1）。")


def _round_of(path: Path) -> int:
    """从快照路径里解析轮次。支持两种落盘布局。"""
    if path.parent.name.startswith("round_"):
        return int(path.parent.name.split("_")[1])
    stem = path.stem
    if stem.startswith("generator_round"):
        return int(stem.replace("generator_round", ""))
    raise ValueError(f"无法从 {path} 解析轮次")


def generator_snapshot_path(run_dir: Path, round_number: int) -> Path:
    """优先新布局 ``round_XXXX/generator.pt``，回退到旧布局。"""
    new = run_dir / round_dir_name(round_number) / "generator.pt"
    if new.exists():
        return new
    legacy = run_dir / "generator" / f"generator_round{int(round_number):04d}.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"轮次 {round_number} 的生成器快照不存在（找过 {new} 与 {legacy}）。"
        f"实验 F **不允许插值填补**缺失的快照轮次（陷阱 8）。")


# ---------------------------------------------------------------------------
# δ：每个 s 只算一次
# ---------------------------------------------------------------------------
def compute_delta(generator: nn.Module, loader, device) -> torch.Tensor:
    """在整个探针集上算出 δ 并缓存成一个张量。

    δ = ``generator(x) * DELTA_SCALE``，与 fba.py:54 数值等价
    （``/255.*4.`` 就是 ``*(4/255)``）。生成器以 ``Tanh`` 收尾
    （generator.py:34），故 ``|δ| <= DELTA_SCALE`` 逐元素成立。

    **缓存的意义不只是省算力**：同一个 s 的整行 t、以及所有评估对象，
    用的必须是**逐位相同**的 δ，否则跨格差异里会混进 δ 本身的抖动。
    探针 loader 是 ``shuffle=False, drop_last=False``，顺序确定，
    因此可以按批次顺序拼接。

    生成器前向不套 ``autocast``（陷阱 5），全程 fp32。
    """
    was_training = generator.training
    generator.eval()
    chunks = []
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                chunks.append((generator(images) * DELTA_SCALE).detach())
    finally:
        if was_training:
            generator.train()
    return torch.cat(chunks, dim=0)


def _slicer(cache: torch.Tensor) -> Callable[[], Callable[[torch.Tensor], torch.Tensor]]:
    """把一个整集张量变成"按批次顺序切片"的取用器。

    返回一个工厂：每次调用得到一个新的、从头开始计数的切片函数。
    每个 (模式, 模型) 的评估都必须用**新鲜的**切片函数，否则偏移量会串。
    """

    def factory() -> Callable[[torch.Tensor], torch.Tensor]:
        offset = {"n": 0}

        def take(images: torch.Tensor) -> torch.Tensor:
            start = offset["n"]
            stop = start + images.shape[0]
            if stop > cache.shape[0]:
                raise RuntimeError(
                    f"缓存耗尽：请求 [{start}, {stop}) 但只有 {cache.shape[0]} 条。"
                    f"探针 loader 的顺序/长度与缓存不一致。")
            chunk = cache[start:stop]
            if chunk.shape[1:] != images.shape[1:]:
                raise RuntimeError(
                    f"缓存形状 {tuple(chunk.shape[1:])} 与当前批次 "
                    f"{tuple(images.shape[1:])} 不匹配。")
            offset["n"] = stop
            return chunk.to(images.device, images.dtype)

        return take

    return factory


def make_frozen_xi_fn(cache: torch.Tensor) -> Callable[[], Callable]:
    """把预先算好的 ``x + ξ_s`` 包成 ``perturb.apply_perturbation`` 要的 ``xi_fn``。

    签名是 ``(images, labels) -> images + xi``，但这里**忽略两个参数的内容**，
    只按批次顺序返回缓存 —— 这正是"ξ 被冻结在 θ_s"的含义：它不再依赖当前模型，
    也不再重新求梯度。

    这样 ``full_xi_frozen`` 可以直接走既有的 ``delta_plus_xi`` 代码路径，
    ``perturb.py`` **零改动**。
    """
    factory = _slicer(cache)

    def build() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        take = factory()

        def frozen_xi_fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return take(images)

        return frozen_xi_fn

    return build


def compute_frozen_xi(model: nn.Module, loader, device, cfg: Cfg) -> torch.Tensor:
    """用 θ_s 在整个探针集上算出 ``x + ξ_s`` 并缓存。

    ξ 需要**真实标签**（无目标 PGD 对真实标签做梯度上升，fba.py:14），
    所以这里必须遍历带标签的 loader。``make_xi_fn`` 会在调用后清零模型梯度。
    """
    xi_fn = make_xi_fn(model, eps=float(cfg.perturb.eps_xi),
                       alpha=float(cfg.perturb.pgd_alpha),
                       num_iter=int(cfg.perturb.pgd_num_iter),
                       seed=int(cfg.perturb.xi_seed))
    chunks = []
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        chunks.append(xi_fn(images, labels).detach())
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
# 判定规则的**唯一定义**在 diag/fedbn.py（陷阱：各处各写一份必然静默漂移）
_is_bn_key = is_fedbn_private_key


def load_snapshot_model(run_dir: Path, round_number: int, *, kind: str,
                        client_id: Optional[int], num_classes: int, device,
                        bn_donor_id: Optional[int] = None,
                        model_size: int = 10) -> nn.Module:
    """加载某一轮的模型快照。

    ``kind='personalized'`` 直接读 ``round_XXXX/client_{cid}.pt``。
    ``kind='global_bn_borrowed'`` 读 ``round_XXXX/global.pt`` 并把 BN 键换成
    供体客户端在**同一轮次**的 BN —— 不能跨轮借，否则共享参数与 BN 统计量
    来自不同时刻，评估对象本身就不自洽。
    """
    directory = run_dir / round_dir_name(round_number)
    if kind == "personalized":
        if client_id is None:
            raise ValueError("kind='personalized' 需要 client_id")
        state = load_checkpoint(directory / f"client_{int(client_id)}.pt")
    elif kind == "global_bn_borrowed":
        if bn_donor_id is None:
            raise ValueError("kind='global_bn_borrowed' 需要 bn_donor_id")
        state = load_checkpoint(directory / "global.pt")
        donor = load_checkpoint(directory / f"client_{int(bn_donor_id)}.pt")
        state = {k: (donor[k].clone() if _is_bn_key(k) and k in donor
                     else v.clone()) for k, v in state.items()}
    else:
        raise ValueError(f"未知 kind '{kind}'")

    model = get_resnet(size=int(model_size), num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.device = device
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 单格评估
# ---------------------------------------------------------------------------
def _evaluate_cell(model: nn.Module, probe, mode: str, *, device, cfg: Cfg,
                   delta_cache: Optional[torch.Tensor] = None,
                   frozen_xi_builder: Optional[Callable] = None,
                   batch_size: int = 256) -> Dict[str, Any]:
    """在共享探针集上评估一个 (模型, 模式)，返回一行指标。"""
    pixel_min, pixel_max = float(cfg.perturb.pixel_min), float(cfg.perturb.pixel_max)
    loader = probe.loader(batch_size)

    if mode == "none":
        preds, labels, realized = _predict(model, loader, device)
        row = _metrics(preds, labels, probe.target_class)
        row.update({"perturb_mode": mode, "linf_realized": realized})
        return row

    if mode == "random_eps8":
        repeats = int(cfg.exp_e.random_repeats)
        collected, realized_max = [], 0.0
        for repeat in range(repeats):
            counter = {"n": 0}
            base_seed = int(cfg.perturb.noise_seed) + repeat * 100003

            def transform(images, labels, _seed=base_seed, _counter=counter):
                rng = torch.Generator(device=images.device)
                rng.manual_seed(_seed + _counter["n"])
                _counter["n"] += 1
                return apply_perturbation(images, "random_eps8", rng=rng,
                                          noise_dist="uniform",
                                          pixel_min=pixel_min, pixel_max=pixel_max)

            preds, labels, realized = _predict(model, loader, device, transform)
            realized_max = max(realized_max, realized)
            collected.append(_metrics(preds, labels, probe.target_class))
        row = {key: float(np.nanmean([entry[key] for entry in collected]))
               for key in collected[0]}
        for key in ("n_eval_samples", "n_eval_samples_unfiltered"):
            row[key] = int(collected[0][key])
        row.update({"perturb_mode": mode, "linf_realized": realized_max})
        return row

    if delta_cache is None:
        raise ValueError(f"模式 '{mode}' 需要 delta_cache")

    delta_factory = _slicer(delta_cache)
    take_delta = delta_factory()

    if mode == "delta_only":
        def transform(images, labels):
            return apply_perturbation(images, "delta_only",
                                      delta=take_delta(images),
                                      pixel_min=pixel_min, pixel_max=pixel_max)
    elif mode == "full_xi_frozen":
        if frozen_xi_builder is None:
            raise ValueError("模式 'full_xi_frozen' 需要 frozen_xi_builder")
        xi_fn = frozen_xi_builder()

        def transform(images, labels):
            return apply_perturbation(images, "delta_plus_xi",
                                      delta=take_delta(images), xi_fn=xi_fn,
                                      labels=labels,
                                      pixel_min=pixel_min, pixel_max=pixel_max)
    elif mode == "full_xi_fresh":
        xi_fn = make_xi_fn(model, eps=float(cfg.perturb.eps_xi),
                           alpha=float(cfg.perturb.pgd_alpha),
                           num_iter=int(cfg.perturb.pgd_num_iter),
                           seed=int(cfg.perturb.xi_seed))

        def transform(images, labels):
            return apply_perturbation(images, "delta_plus_xi",
                                      delta=take_delta(images), xi_fn=xi_fn,
                                      labels=labels,
                                      pixel_min=pixel_min, pixel_max=pixel_max)
    else:
        raise ValueError(f"未知模式 '{mode}'；可选 "
                         f"{list(EXP_F_MODES) + list(MODES_PER_T)}")

    preds, labels, realized = _predict(model, loader, device, transform)
    row = _metrics(preds, labels, probe.target_class)
    row.update({"perturb_mode": mode, "linf_realized": realized})
    return row


# ---------------------------------------------------------------------------
# 矩阵驱动
# ---------------------------------------------------------------------------
@dataclass
class _Target:
    """一个被评估的对象：(kind, client_id)。"""
    kind: str
    client_id: Optional[int]

    @property
    def label(self) -> str:
        return (f"{self.kind}" if self.client_id is None
                else f"{self.kind}:{self.client_id}")


def run_matrix(ckpt_root: Path, alpha: float, seed: int, *, cfg: Cfg, device,
               grid: Optional[Sequence[int]] = None,
               modes: Optional[Sequence[str]] = None,
               per_t_modes: Optional[Sequence[str]] = None,
               eval_targets: Optional[Sequence[str]] = None,
               client_ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """跑上三角矩阵，返回每 (s, t, mode, eval_target, client) 一行的 DataFrame。"""
    assert_delta_scale(cfg)

    run_dir = ckpt_root / run_dir_name("attack", alpha, seed)
    meta = load_meta(run_dir / "meta.json")
    if str(meta["mode"]) != "attack":
        raise ValueError(f"实验 F 必须跑 attack run，但 meta 显示 mode={meta['mode']}")

    num_classes = int(meta["num_classes"])
    target_class = int(meta["target_class"])
    batch_size = int(cfg.probe.batch_size)

    smoke = bool(meta.get("smoke", False))
    _, test_dataset, _ = build_datasets(
        cfg, smoke, dataset_sizes=meta.get("synthetic_sizes"))
    if smoke:
        # 合成测试集只有几百张，正式的 probe_n=2000 会超出容量。
        # **只在冒烟 run 上生效**，且明确打印 —— 不对真实数据做任何静默降级。
        cfg = Cfg({**cfg, "exp_e": {**cfg["exp_e"],
                                    "probe_n": int(cfg.smoke.exp_e_probe_n)}})
        print(f"[exp_f] 冒烟 run：探针集规模临时改为 {cfg.exp_e.probe_n}")
    probe = build_exp_e_probe(test_dataset, target_class, cfg)

    # --- 网格：以磁盘为准，取生成器与模型快照的交集 ---------------------
    model_rounds = set(available_rounds(run_dir, "global"))
    gen_rounds = set(available_rounds(run_dir, "generator"))
    gen_rounds |= {int(path.stem.replace("generator_round", ""))
                   for path in (run_dir / "generator").glob("generator_round*.pt")}
    usable = sorted(model_rounds & gen_rounds)
    if grid is not None:
        requested = sorted(int(r) for r in grid)
        missing = [r for r in requested if r not in set(usable)]
        if missing:
            raise FileNotFoundError(
                f"请求的轮次 {missing} 缺少成对的快照（可用: {usable}）。"
                f"实验 F **不允许插值填补**（陷阱 8）。")
        usable = requested
    if len(usable) < 2:
        raise RuntimeError(
            f"只有 {len(usable)} 个成对的快照轮次（{usable}），无法构造矩阵。"
            f"需要用 `--snapshot-every` 重跑一次 attack run。")

    # --- 评估对象 -------------------------------------------------------
    manifest = load_manifest(run_dir)
    staleness_by_key = {(int(r["client_id"]), int(r["grid_round"])): int(r["staleness"])
                        for r in manifest["records"] if r["kind"] == "client"}
    actual_by_key = {(int(r["client_id"]), int(r["grid_round"])): int(r["actual_round"])
                     for r in manifest["records"] if r["kind"] == "client"}

    malicious_ids = {int(r["client_id"]) for r in meta["clients"]
                     if bool(r["is_malicious"])}
    snapshot_ids = [int(c) for c in manifest["snapshot_client_ids"]]
    benign_ids = [c for c in snapshot_ids if c not in malicious_ids]
    if client_ids is not None:
        wanted = {int(c) for c in client_ids}
        benign_ids = [c for c in benign_ids if c in wanted]
        snapshot_ids = [c for c in snapshot_ids if c in wanted]

    donor = cfg.exp_f.get("bn_donor_client", None)
    bn_donor_id = int(donor) if donor is not None else (
        min(benign_ids) if benign_ids else None)

    eval_targets = list(eval_targets or cfg.exp_f.eval_targets)
    targets: List[_Target] = []
    if "personalized" in eval_targets:
        targets += [_Target("personalized", cid) for cid in benign_ids]
    if "global_bn_borrowed" in eval_targets:
        if bn_donor_id is None:
            raise ValueError("没有可用的 BN 供体客户端")
        targets.append(_Target("global_bn_borrowed", None))

    matrix_modes = [m for m in (modes or cfg.exp_f.modes_p2) if m in EXP_F_MODES]
    # ``none`` 无条件加入：它既是负基线，也是 ``baseline_asr`` / ``clean_acc``
    # 两列的来源，缺了它整张表的相对量就无从计算（陷阱 3）。
    per_t = list(per_t_modes if per_t_modes is not None else cfg.exp_f.modes_per_t)
    per_t = ["none"] + [m for m in per_t if m != "none" and m in MODES_PER_T]

    rows: List[Dict[str, Any]] = []
    common = {"alpha": float(alpha), "seed": int(seed),
              "probe_set": "shared", "n_probe": len(probe.indices)}

    def _provenance(target: _Target, t: int) -> Dict[str, Any]:
        """该 (对象, 网格轮次) 的快照溯源信息。

        ``global_bn_borrowed`` 的共享参数取自轮次 t 的 ``global.pt``（无滞后），
        但它借的 BN 来自供体客户端在同一轮的快照，**滞后随供体走**。
        """
        cid = target.client_id if target.kind == "personalized" else bn_donor_id
        if cid is None:
            return {"snapshot_actual_round": int(t), "staleness": 0}
        return {"snapshot_actual_round": int(actual_by_key.get((cid, t), t)),
                "staleness": int(staleness_by_key.get((cid, t), 0))}

    # --- 第一遍：逐 t 的基线（陷阱 3：不能用单一常数）--------------------
    baseline: Dict[Tuple[str, int], float] = {}
    clean_acc: Dict[Tuple[str, int], float] = {}
    pending: List[Tuple[Dict[str, Any], str, int]] = []
    for target in targets:
        for t in usable:
            model = load_snapshot_model(
                run_dir, t, kind=target.kind, client_id=target.client_id,
                num_classes=num_classes, device=device, bn_donor_id=bn_donor_id)
            for mode in per_t:
                row = _evaluate_cell(model, probe, mode, device=device, cfg=cfg,
                                     batch_size=batch_size)
                if mode == "none":
                    baseline[(target.label, t)] = float(row["asr_targeted"])
                    clean_acc[(target.label, t)] = 1.0 - float(row["error_rate"])
                record = {
                    **common, **row,
                    "gen_round_s": np.nan, "model_round_t": int(t), "gap": np.nan,
                    "eval_target": target.kind, "client_id": target.client_id,
                    **_provenance(target, t),
                }
                rows.append(record)
                pending.append((record, target.label, t))
            del model
        print(f"[exp_f] 基线完成: {target.label}")

    # 基线在整个第一遍跑完后才齐全，所以回填而不是边算边填。
    for record, label, t in pending:
        record["baseline_asr"] = baseline.get((label, t), np.nan)
        record["clean_acc"] = clean_acc.get((label, t), np.nan)

    # --- 第二遍：上三角矩阵 ----------------------------------------------
    needs_frozen_xi = "full_xi_frozen" in matrix_modes
    for s in usable:
        with FrozenGenerator(generator_snapshot_path(run_dir, s), device) as frozen:
            for target in targets:
                # δ 与探针集绑定，与模型无关 → 每个 s 只算一次，整行 t 复用。
                delta_cache = compute_delta(
                    frozen.module, probe.loader(batch_size), device)

                frozen_xi_builder = None
                if needs_frozen_xi:
                    model_s = load_snapshot_model(
                        run_dir, s, kind=target.kind, client_id=target.client_id,
                        num_classes=num_classes, device=device,
                        bn_donor_id=bn_donor_id)
                    xi_cache = compute_frozen_xi(
                        model_s, probe.loader(batch_size), device, cfg)
                    frozen_xi_builder = make_frozen_xi_fn(xi_cache)
                    del model_s

                for t in usable:
                    if t < s:
                        continue    # 陷阱 7：下三角无意义，不计算也不填 0
                    model = load_snapshot_model(
                        run_dir, t, kind=target.kind, client_id=target.client_id,
                        num_classes=num_classes, device=device,
                        bn_donor_id=bn_donor_id)
                    for mode in matrix_modes:
                        row = _evaluate_cell(
                            model, probe, mode, device=device, cfg=cfg,
                            delta_cache=delta_cache,
                            frozen_xi_builder=frozen_xi_builder,
                            batch_size=batch_size)
                        rows.append({
                            **common, **row,
                            "gen_round_s": int(s), "model_round_t": int(t),
                            "gap": int(t - s),
                            "eval_target": target.kind,
                            "client_id": target.client_id,
                            "baseline_asr": baseline.get((target.label, t), np.nan),
                            "clean_acc": clean_acc.get((target.label, t), np.nan),
                            **_provenance(target, t),
                            "generator_hash": frozen.hash,
                        })
                    del model
                del delta_cache
                print(f"[exp_f] s={s} {target.label} 完成")
        # 退出 with 时 FrozenGenerator.verify() 校验哈希，被改动即抛异常

    frame = pd.DataFrame(rows)
    frame["asr_targeted_rel"] = frame["asr_targeted"] - frame["baseline_asr"]
    frame = frame.rename(columns={"asr_targeted_unfiltered": "asr_unfiltered"})
    column_order = ["gen_round_s", "model_round_t", "gap", "perturb_mode",
                    "eval_target", "client_id", "asr_targeted",
                    "asr_targeted_rel", "asr_unfiltered", "error_rate",
                    "clean_acc", "baseline_asr", "snapshot_actual_round",
                    "staleness", "alpha", "seed", "n_eval_samples"]
    rest = [c for c in frame.columns if c not in column_order]
    return frame[[c for c in column_order if c in frame.columns] + rest]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="实验 F：冻结生成器的时间衰减矩阵")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--grid", type=int, nargs="*", default=None,
                        help="快照轮次；缺省用磁盘上全部成对的轮次")
    parser.add_argument("--sparse", action="store_true",
                        help="用 config 的 exp_f.grid_sparse（P2 的 5 个点）")
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"矩阵模式，可选 {list(EXP_F_MODES)}；"
                             f"缺省用 config 的 exp_f.modes_p2")
    parser.add_argument("--per-t-modes", nargs="*", default=None,
                        help=f"只随 t 变的模式，可选 {list(MODES_PER_T)}；"
                             f"'none' 恒被包含（它是 baseline_asr 的来源）")
    parser.add_argument("--eval-targets", nargs="*", default=None)
    parser.add_argument("--client-ids", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    grid = args.grid
    if args.sparse:
        if grid:
            raise SystemExit("--sparse 与 --grid 互斥")
        grid = list(cfg.exp_f.grid_sparse)

    frame = run_matrix(Path(args.ckpt_root), args.alpha, args.seed, cfg=cfg,
                       device=device, grid=grid, modes=args.modes,
                       per_t_modes=args.per_t_modes,
                       eval_targets=args.eval_targets, client_ids=args.client_ids)

    out = Path(args.out) if args.out else (
        Path(cfg.paths.results_root) / "raw" /
        f"exp_f_matrix_a{args.alpha}_s{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[exp_f] {len(frame)} 行 -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
