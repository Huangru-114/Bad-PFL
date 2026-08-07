"""四个实验脚本共用的加载与装配逻辑。

职责：解析 CLI、加载 checkpoint 与 meta、构造共享探针集、拼装 CSV 的公共列。
不含任何指标计算 —— 那些都在 ``diag.metrics``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

from . import REPO_ROOT  # noqa: F401
from .config import Cfg, load_config
from .hooks import load_checkpoint, load_meta
from .perturb import make_delta_fn, make_xi_fn
from .probe import ProbeSet, build_probe_set
from .run_fl import build_datasets

from generator import Autoencoder
from resnet import get_resnet

__all__ = ["COMMON_COLUMNS", "add_common_args", "resolve_device", "RunBundle",
           "load_bundle", "common_row", "write_csv"]

# 每个实验的 CSV 都必须带上这些列。
# 注意：``n_target_samples`` 已按修订更名为 ``n_target_samples_local``，
# 且它现在是**协变量**而不是过滤条件 —— 改用共享探针集后所有客户端都能算出
# 有效指标，包括本地一张目标类图片都没有的客户端。那些客户端恰恰最有信息量。
COMMON_COLUMNS = ["client_id", "is_malicious", "is_malicious_slot",
                  "poison_ratio", "n_local_samples", "n_target_samples_local",
                  "n_participations", "alpha", "seed", "mode"]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--ckpt-dir", default=None,
                        help="checkpoint 目录（{mode}_a{alpha}_s{seed}）")
    parser.add_argument("--out", default=None, help="输出 CSV 路径")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="极小规模冒烟：合成数据 + 冒烟探针集；"
                             "未提供 --ckpt-dir 时使用随机初始化的假模型")
    return parser


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RunBundle:
    """一次 run 的全部可加载产物 + 共享探针集。

    模型按需加载（``client_model(cid)``），避免一次性把上百个 resnet 读进内存。
    """

    def __init__(self, cfg: Cfg, ckpt_dir: Optional[Path], device: torch.device,
                 smoke: bool, meta: Dict[str, Any], probe: ProbeSet,
                 num_classes: int, train_dataset: Any = None):
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.smoke = smoke
        self.meta = meta
        self.probe = probe
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.target_class = int(meta["target_class"])
        self._cache: Dict[str, nn.Module] = {}

    def local_train_loader(self, record: Dict[str, Any]):
        """该客户端**本地训练数据**的评估 loader。

        ⚠️ 只有 ``observable_score`` 的类原型该用它 —— 那是为了忠实模拟防御方
        在真实部署中能拿到的东西。其余所有指标一律用共享探针集。

        用 ``make_eval_loader``（``drop_last=False``, ``shuffle=False``），
        不是训练用的那个 loader。
        """
        from torch.utils.data import Subset

        from .probe import make_eval_loader

        if self.train_dataset is None:
            raise RuntimeError("bundle 未携带 train_dataset")
        indices = record.get("train_indices")
        if indices is None:
            raise KeyError(
                f"meta 中 client {record['client_id']} 缺少 'train_indices'；"
                f"该 checkpoint 由旧版 run_fl 生成，请重新训练。")
        batch_size = (int(self.cfg.smoke.probe.batch_size) if self.smoke
                      else int(self.cfg.probe.batch_size))
        return make_eval_loader(Subset(self.train_dataset, list(indices)),
                                batch_size=batch_size)

    # -- 模型 ---------------------------------------------------------------
    def _new_model(self) -> nn.Module:
        return get_resnet(size=10, num_classes=self.num_classes)

    def _load(self, name: str) -> nn.Module:
        if name in self._cache:
            return self._cache[name]
        model = self._new_model()
        if self.ckpt_dir is not None:
            path = self.ckpt_dir / f"{name}.pt"
            if not path.exists():
                raise FileNotFoundError(f"找不到 checkpoint: {path}")
            model.load_state_dict(load_checkpoint(path), strict=True)
        # ckpt_dir 为 None 时保持随机初始化 —— 仅冒烟测试路径。
        model.to(self.device)
        model.device = self.device      # utils.evaluate_accuracy 依赖此属性
        model.eval()
        self._cache[name] = model
        return model

    def client_model(self, client_id: int) -> nn.Module:
        return self._load(f"client_{int(client_id)}")

    def global_model(self) -> nn.Module:
        return self._load("global")

    def client_state(self, client_id: int) -> Dict[str, torch.Tensor]:
        if self.ckpt_dir is None:
            return {k: v.detach().cpu()
                    for k, v in self.client_model(client_id).state_dict().items()}
        return load_checkpoint(self.ckpt_dir / f"client_{int(client_id)}.pt")

    def global_state(self) -> Dict[str, torch.Tensor]:
        if self.ckpt_dir is None:
            return {k: v.detach().cpu()
                    for k, v in self.global_model().state_dict().items()}
        return load_checkpoint(self.ckpt_dir / "global.pt")

    # -- 生成器 -------------------------------------------------------------
    def generator(self, required: bool = True) -> Optional[nn.Module]:
        """加载生成器；clean run 没有生成器。

        ``required=True`` 且找不到时抛异常 —— 实验 A/B/C 需要 δ，
        δ 必须来自**攻击 run** 的生成器。
        """
        if "generator" in self._cache:
            return self._cache["generator"]
        model = Autoencoder()
        if self.ckpt_dir is not None:
            path = self.ckpt_dir / "generator.pt"
            if not path.exists():
                if required:
                    raise FileNotFoundError(
                        f"找不到 {path}。δ 只存在于攻击 run；"
                        f"请用 --gen-ckpt-dir 指向对应的 attack run 目录。")
                return None
            model.load_state_dict(load_checkpoint(path), strict=True)
        model.to(self.device)
        model.eval()
        self._cache["generator"] = model
        return model

    # -- 扰动闭包 -----------------------------------------------------------
    def delta_fn(self, generator: Optional[nn.Module] = None) -> Callable:
        gen = generator if generator is not None else self.generator()
        return make_delta_fn(gen, eps=float(self.cfg.perturb.eps_delta))

    def xi_fn(self, model: nn.Module) -> Callable:
        """构造绑定到指定模型的 ξ 闭包。

        保持 fba.py:53 的原有语义：ξ 由**被评估的那个模型**生成。
        """
        return make_xi_fn(model, eps=float(self.cfg.perturb.eps_xi),
                          alpha=float(self.cfg.perturb.pgd_alpha),
                          num_iter=int(self.cfg.perturb.pgd_num_iter),
                          seed=int(self.cfg.perturb.xi_seed))

    def probe_loaders(self) -> Dict[str, Any]:
        batch_size = (int(self.cfg.smoke.probe.batch_size) if self.smoke
                      else int(self.cfg.probe.batch_size))
        return self.probe.loaders(batch_size=batch_size)

    def knn_k(self) -> int:
        return (int(self.cfg.smoke.probe.knn_k) if self.smoke
                else int(self.cfg.metrics.knn_k))

    def client_records(self) -> List[Dict[str, Any]]:
        return list(self.meta["clients"])


def _fake_meta(cfg: Cfg) -> Dict[str, Any]:
    """冒烟测试用的假 meta（没有 checkpoint 时）。"""
    smoke_cfg = cfg.smoke
    num_clients = int(smoke_cfg.num_clients)
    bad = int(smoke_cfg.bad_client_num)
    num_classes = int(smoke_cfg.num_classes)
    clients = []
    for cid in range(num_clients):
        is_bad = cid >= num_clients - bad
        clients.append({
            "client_id": cid, "partition_idx": cid,
            "is_malicious": bool(is_bad), "is_malicious_slot": bool(is_bad),
            "poison_ratio": float(cfg.fl.poison_rate) if is_bad else 0.0,
            "n_local_samples": int(smoke_cfg.samples_per_client),
            "n_local_test_samples": int(smoke_cfg.test_samples_per_client),
            "label_histogram": [int(smoke_cfg.samples_per_client) // num_classes] * num_classes,
            "test_label_histogram": [int(smoke_cfg.test_samples_per_client) // num_classes] * num_classes,
            "n_target_samples_local": int(smoke_cfg.samples_per_client) // num_classes,
            "n_participations": int(smoke_cfg.rounds),
        })
    return {"mode": "smoke", "alpha": float(smoke_cfg.alpha),
            "seed": int(smoke_cfg.seed), "smoke": True,
            "num_classes": num_classes,
            "target_class": int(cfg.data.target_class),
            "client_num": num_clients, "bad_client_num": bad,
            "clients": clients}


def load_bundle(args: argparse.Namespace, cfg: Optional[Cfg] = None) -> RunBundle:
    """从 CLI 参数装配 ``RunBundle``（加载 meta、建探针集）。"""
    cfg = cfg if cfg is not None else load_config(args.config)
    device = resolve_device(args.device)
    smoke = bool(args.smoke)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

    if ckpt_dir is not None:
        meta = load_meta(ckpt_dir / "meta.json")
    elif smoke:
        meta = _fake_meta(cfg)
    else:
        raise ValueError("必须提供 --ckpt-dir（或使用 --smoke）")

    num_classes = int(meta["num_classes"])
    train_dataset, test_dataset, _ = build_datasets(cfg, smoke)

    probe_cfg = cfg.smoke.probe if smoke else cfg.probe
    probe_cfg = dict(probe_cfg)
    probe_cfg.setdefault("seed", int(cfg.probe.seed))   # 探针 seed 始终与实验 seed 解耦
    probe = build_probe_set(test_dataset, int(meta["target_class"]),
                            {"probe": probe_cfg})

    return RunBundle(cfg, ckpt_dir, device, smoke, meta, probe, num_classes,
                     train_dataset=train_dataset)


def common_row(bundle: RunBundle, record: Dict[str, Any]) -> Dict[str, Any]:
    """CSV 的公共列。"""
    return {
        "client_id": int(record["client_id"]),
        "is_malicious": bool(record["is_malicious"]),
        "is_malicious_slot": bool(record.get("is_malicious_slot", record["is_malicious"])),
        "poison_ratio": float(record.get("poison_ratio", 0.0)),
        "n_local_samples": int(record.get("n_local_samples", 0)),
        "n_target_samples_local": int(record.get("n_target_samples_local", 0)),
        "n_participations": int(record.get("n_participations", 0)),
        "alpha": float(bundle.meta["alpha"]),
        "seed": int(bundle.meta["seed"]),
        "mode": str(bundle.meta["mode"]),
    }


def write_csv(frame: pd.DataFrame, out_path, default_name: str,
              cfg: Cfg) -> Path:
    """写 CSV，自动建目录。``out_path`` 为 None 时落到 ``results/raw/``。"""
    if out_path is None:
        out_path = Path(cfg.paths.results_root) / "raw" / default_name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"[write_csv] {len(frame)} 行 -> {out_path}")
    return out_path
