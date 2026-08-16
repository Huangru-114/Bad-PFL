"""训练期的逐轮埋点与周期评估（实验 I §5 / 实验 J §2、§5）。

挂在既有事件上（``on_round_begin`` / ``on_round_end``）与 ``use_defense`` 的
聚合回调上，**对原仓库文件零改动**。

# 两件必须分开报的事（实验 I §4.2 / 实验 J §5.1）

服务器端防御只保护**全局模型**，而 Bad-PFL 的 ASR 定义在**个性化模型**上。
只报一个会得出错误结论，所以两个都算。

# 全局模型必须借 BN

FedBN 下 ``server.global_model`` 的 BatchNorm 从初始化起从未更新过
（``pfl.py:5-12`` 把 BN 键从 ``server.update`` pop 掉）。直接评估它得到的是一个
失配模型。因此 ``acc_global`` / ``asr_global_*`` 一律在**借 BN 的全局模型**上算
（全局共享参数 + 一个固定良性客户端的 BN），同时**另存** ``acc_global_raw``
把这个退化程度显式暴露出来，而不是藏起来。

# 个性化模型的取样与陈旧度

FedBN 下客户端只在被选中的轮次更新本地模型。评估的是一组**固定**的良性客户端，
它们的模型停在各自上次参与的那一轮 —— 这正是"该客户端此刻的个性化模型"，
也是论文 ASR 的定义。同时记录本轮这组里有几个参与过，供分析时作协变量。
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import Cfg
from .defenses import DefenseOutcome
from .fedbn import is_fedbn_private_key
from .instrumentation import RoundRecord, RoundRecorder, round_signals
from .invariant_agg import trim_count

__all__ = ["TrainingTracker", "IMPLANTATION_COLUMNS", "preserve_rng_state"]


@contextlib.contextmanager
def preserve_rng_state():
    """保存并恢复 torch / numpy / random 三条随机流。

    **这不是防御性编程，是修一个实测到的 bug。**

    周期评估里为了构造"借 BN 的全局模型"要 ``get_resnet()`` 新建模型，
    而权重初始化会消耗全局 torch RNG。后果是：良性客户端看不出变化
    （它们的 DataLoader 迭代器在 ``BasicClient.__init__`` 时就建好了），
    但恶意客户端的投毒掩码（``fba.py:49`` 的 ``torch.rand``）、
    PGD 的随机起点（``fba.py:8``）与生成器训练全部现取 RNG，于是被整体推移。

    实测症状：开 ``--eval-every 1`` 与关掉相比，``client_{恶意}.pt`` /
    ``generator.pt`` / ``global.pt`` 三个 checkpoint 哈希不同，
    而三个良性客户端**完全一致** —— 这种局部差异极易被误读成"防御生效了"。

    所以整个评估过程都包在这里面，而不是只包新建模型那几行：
    评估路径上任何一处碰随机流都会被吸收掉。
    """
    torch_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)

IMPLANTATION_COLUMNS = [
    "round", "defense", "tau", "alpha", "alpha_dirichlet", "seed", "variant",
    "acc_global", "acc_global_raw", "acc_personalized",
    "asr_global_targeted", "asr_personalized_targeted", "asr_unfiltered",
    "mask_keep_ratio", "effective_trim_n", "zero_sign_ratio",
    "n_malicious_this_round", "rounds_since_last_malicious",
    "n_eval_clients", "n_eval_clients_participated_this_round",
]


@dataclass
class TrainingTracker:
    """逐轮 npz + 周期性 ACC/ASR 评估。

    Parameters
    ----------
    eval_every:
        每多少轮评估一次；``<=0`` 关闭评估（只记逐轮标量）。
    eval_client_ids:
        用于个性化模型评估的良性客户端。BN 供体取其中 ``client_id`` 最小者。
    generator_getter:
        返回攻击生成器；clean run 传 ``None``，此时只评估 ACC，ASR 记为 nan。
    """

    cfg: Cfg
    recorder: Optional[RoundRecorder]
    device: Any
    defense_name: str = ""
    variant: str = ""
    tau: float = float("nan")
    trim_alpha: float = float("nan")
    alpha_dirichlet: float = float("nan")
    seed: int = 0
    eval_every: int = 0
    eval_client_ids: Sequence[int] = field(default_factory=tuple)
    generator_getter: Optional[Any] = None
    probe: Any = None
    target_class: int = 0
    num_classes: int = 10

    # 运行时状态
    cur_round: int = 0
    selected_indices: Sequence[int] = field(default_factory=tuple)
    last_malicious_round: Optional[int] = None
    implantation_rows: List[Dict[str, Any]] = field(default_factory=list)
    _handlers: List[Any] = field(default_factory=list)

    # -- 事件 -------------------------------------------------------------
    def attach(self, server: Any, clients: Sequence[Any]) -> "TrainingTracker":
        from event_emitter import fl_event_emitter

        def _on_round_begin(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            self.cur_round = int(kwargs["cur_round"])
            self.selected_indices = list(kwargs.get("selected_client_indices", []))

        def _on_round_end(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            self._maybe_evaluate(server, clients)

        self._handlers = [("on_round_begin", _on_round_begin),
                          ("on_round_end", _on_round_end)]
        for event, handler in self._handlers:
            fl_event_emitter.on(event, handler)
        return self

    def detach(self) -> None:
        from event_emitter import fl_event_emitter

        for event, handler in self._handlers:
            fl_event_emitter.off(event, handler)
        self._handlers = []

    # -- 聚合回调（由 defenses.use_defense 调用） -------------------------
    def on_aggregate(self, outcome: DefenseOutcome,
                     state_dicts: List[Dict[str, torch.Tensor]],
                     w_prev: Dict[str, torch.Tensor],
                     clients: Sequence[Any]) -> None:
        """在 ``load_state_dict`` 之前被调用，用同一份 ``w_prev`` 算逐轮标量。"""
        indices = list(self.selected_indices)
        if len(indices) != len(state_dicts):
            indices = indices[:len(state_dicts)]
        client_ids = np.array([int(clients[i].cid) for i in indices], dtype=int)
        malicious = np.array(
            [bool(getattr(clients[i], "diag_is_malicious", False))
             for i in indices], dtype=bool)
        if malicious.any():
            self.last_malicious_round = self.cur_round

        if self.recorder is None:
            return

        k = (trim_count(len(state_dicts), self.trim_alpha)
             if np.isfinite(self.trim_alpha) else None)
        signals = round_signals(w_prev, state_dicts, trim_k=k)

        self.recorder.write(RoundRecord(
            round_index=self.cur_round,
            client_ids=client_ids,
            is_malicious=malicious,
            update_norm=signals["update_norm"],
            l2_to_median=signals["l2_to_median"],
            cos_to_median=signals["cos_to_median"],
            influence=np.asarray(outcome.influence, dtype=float),
            trim_survival_rate=outcome.trim_survival_rate,
            mask_keep_ratio=float(outcome.mask_keep_ratio),
            zero_sign_ratio=float(outcome.zero_sign_ratio),
            effective_trim_n=int(outcome.effective_trim_n),
            n_malicious_in_round=int(malicious.sum()),
            n_selected=len(state_dicts),
            defense=self.defense_name,
            extra={**outcome.extra,
                   "selected": (outcome.selected.tolist()
                                if outcome.selected is not None else None),
                   "median_influence": signals["median_influence"].tolist()}))

    # -- 周期评估 ---------------------------------------------------------
    def _borrowed_bn_state(self, server: Any, donor: Any
                           ) -> Dict[str, torch.Tensor]:
        """全局共享参数 + 供体客户端的 BN（见模块 docstring）。"""
        global_state = server.global_model.state_dict()
        donor_state = donor.local_model.state_dict()
        return {key: (donor_state[key].detach().clone()
                      if is_fedbn_private_key(key) and key in donor_state
                      else value.detach().clone())
                for key, value in global_state.items()}

    def _evaluate_model(self, model: Any) -> Dict[str, float]:
        from .exp_e import _metrics, _predict, evaluate_mode

        batch_size = int(self.cfg.probe.batch_size)
        clean = evaluate_mode(model, self.probe, "none", device=self.device,
                              cfg=self.cfg, batch_size=batch_size)
        row = {"acc": 1.0 - float(clean["error_rate"]),
               "asr_targeted": float("nan"),
               "asr_unfiltered": float("nan")}
        generator = (self.generator_getter() if self.generator_getter is not None
                     else None)
        if generator is None:
            return row

        from .perturb import make_delta_fn, make_xi_fn

        delta_fn = make_delta_fn(generator, eps=float(self.cfg.perturb.eps_delta))
        xi_fn = make_xi_fn(model, eps=float(self.cfg.perturb.eps_xi),
                           alpha=float(self.cfg.perturb.pgd_alpha),
                           num_iter=int(self.cfg.perturb.pgd_num_iter),
                           seed=int(self.cfg.perturb.xi_seed))
        attacked = evaluate_mode(model, self.probe, "full", device=self.device,
                                 cfg=self.cfg, delta_fn=delta_fn, xi_fn=xi_fn,
                                 batch_size=batch_size)
        row["asr_targeted"] = float(attacked["asr_targeted"])
        row["asr_unfiltered"] = float(attacked["asr_targeted_unfiltered"])
        return row

    def _maybe_evaluate(self, server: Any, clients: Sequence[Any]) -> None:
        if self.eval_every <= 0 or self.probe is None:
            return
        if self.cur_round % int(self.eval_every) != 0:
            return
        # 评估绝不能推进任何随机流，否则恶意客户端的投毒与生成器训练会被悄悄
        # 改变，而良性客户端看不出差别 —— 见 preserve_rng_state 的 docstring。
        with preserve_rng_state():
            self._evaluate_now(server, clients)

    def _evaluate_now(self, server: Any, clients: Sequence[Any]) -> None:
        from resnet import get_resnet

        by_id = {int(c.cid): c for c in clients}
        eval_clients = [by_id[int(i)] for i in self.eval_client_ids
                        if int(i) in by_id]
        if not eval_clients:
            return
        donor = eval_clients[0]

        # --- 全局模型（借 BN，以及原样的对照） ---
        model = get_resnet(size=10, num_classes=self.num_classes)
        model.load_state_dict(self._borrowed_bn_state(server, donor), strict=True)
        model.to(self.device)
        model.device = self.device
        global_row = self._evaluate_model(model)
        del model

        raw = get_resnet(size=10, num_classes=self.num_classes)
        raw.load_state_dict(server.global_model.state_dict(), strict=True)
        raw.to(self.device)
        raw.device = self.device
        acc_global_raw = self._evaluate_model(raw)["acc"]
        del raw

        # --- 个性化模型 ---
        rows = [self._evaluate_model(client.local_model) for client in eval_clients]
        selected_ids = {int(clients[i].cid) for i in self.selected_indices}
        participated = sum(1 for c in eval_clients if int(c.cid) in selected_ids)

        def _mean(key: str) -> float:
            values = [r[key] for r in rows if np.isfinite(r[key])]
            return float(np.mean(values)) if values else float("nan")

        malicious_now = sum(
            1 for i in self.selected_indices
            if bool(getattr(clients[i], "diag_is_malicious", False)))
        self.implantation_rows.append({
            "round": self.cur_round, "defense": self.defense_name,
            "tau": self.tau, "alpha": self.trim_alpha,
            "alpha_dirichlet": self.alpha_dirichlet, "seed": self.seed,
            "variant": self.variant or self.defense_name,
            "acc_global": global_row["acc"],
            "acc_global_raw": acc_global_raw,
            "acc_personalized": _mean("acc"),
            "asr_global_targeted": global_row["asr_targeted"],
            "asr_personalized_targeted": _mean("asr_targeted"),
            "asr_unfiltered": _mean("asr_unfiltered"),
            "mask_keep_ratio": float("nan"), "effective_trim_n": -1,
            "zero_sign_ratio": float("nan"),
            "n_malicious_this_round": malicious_now,
            "rounds_since_last_malicious": (
                self.cur_round - self.last_malicious_round
                if self.last_malicious_round is not None else -1),
            "n_eval_clients": len(eval_clients),
            "n_eval_clients_participated_this_round": participated,
        })

    # -- 产出 -------------------------------------------------------------
    def implantation_frame(self):
        import pandas as pd

        frame = pd.DataFrame(self.implantation_rows)
        if frame.empty:
            return pd.DataFrame(columns=IMPLANTATION_COLUMNS)
        # 逐轮的聚合统计从 npz 回填，避免两处各记一份而漂移
        if self.recorder is not None:
            by_round = {int(r.round_index): r for r in self.recorder.records}
            for column, attribute in (("mask_keep_ratio", "mask_keep_ratio"),
                                      ("effective_trim_n", "effective_trim_n"),
                                      ("zero_sign_ratio", "zero_sign_ratio")):
                frame[column] = [getattr(by_round[int(r)], attribute)
                                 if int(r) in by_round else np.nan
                                 for r in frame["round"]]
        rest = [c for c in frame.columns if c not in IMPLANTATION_COLUMNS]
        return frame[[c for c in IMPLANTATION_COLUMNS if c in frame.columns] + rest]
