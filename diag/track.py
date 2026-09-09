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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Cfg
from .defenses import DefenseOutcome
from .fedbn import is_fedbn_private_key, split_keys
from .instrumentation import (RoundRecord, RoundRecorder, gram_matrix,
                              group_cosines, layer_signals, round_signals)
from .invariant_agg import trim_count

__all__ = ["TrainingTracker", "IMPLANTATION_COLUMNS", "EDGE_COLUMNS",
           "preserve_rng_state"]


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
    # bad_client_num 必须落在数据里。跨密度并表时若只能从文件名解析 `_bad5`,
    # 一次改名或一个没打 tag 的 run 就会被静默归到错误的密度上。
    "round", "defense", "bad_client_num", "poison_rate",
    "schedule", "attack_active_this_round",
    "tau", "alpha", "alpha_dirichlet", "seed", "variant",
    "acc_global", "acc_global_raw", "acc_personalized",
    "asr_global_targeted", "asr_personalized_targeted", "asr_unfiltered",
    # 论文口径 ASR（原始 full_poison_func，各客户端自己 test loader，不过滤）。
    # 这是 exp1 的**主** ASR；上面 asr_personalized_targeted 是 perturb 分解口径,
    # 只对良性求平均且经 δ/ξ 重建，两者不可混。asr_paper_all 才对齐 main.py 的
    # "Avg ASR"（全体客户端均值）。
    "asr_paper_benign", "asr_paper_malicious", "asr_paper_all",
    # 同一批前向里算出的 **filtered** 口径：排除真实标签已经是目标类的样本，
    # 与 tf-dpfl 的 attack/backdoor_eval.py:41 同分母，也与报告 §2 写的定义一致。
    # 此前只有 recompute_asr_final 能出 filtered，而它只能算最终轮
    # （逐轮的 per-client 模型没有存）。CIFAR-10 下 filtered ≈ (unfiltered−0.1)/0.9。
    "asr_paper_filtered_benign", "asr_paper_filtered_malicious",
    "asr_paper_filtered_all",
    # B 线：停攻点冻结触发器后的 ASR（仅 --freeze-trigger-eval 的 persist 跑有值，
    # 停攻前为 nan）。asr_paper_all 是在线 ξ（A/C 线），这个是完全固定触发器。
    "asr_paper_frozen_benign", "asr_paper_frozen_malicious",
    "asr_paper_frozen_all",
    # 主任务：mta 是全部样本上的准确率，acc_* 是非目标类样本上的（≠ mta）
    "mta_personalized", "mta_global",
    # FedRep 定义的个性化模型 = [当前共享表示, 私有头]（见 _fedrep_shared_pm）。
    # 与上面那一列**并排**量，不替换它：哪种组合才对由数据判，不是改口径宣布。
    # **仅 FedRep 臂有值**，fedbn / 无 pfl 的 run 全是 nan（不是 0，铁律 #5）。
    "mta_personalized_shared", "asr_paper_shared_benign",
    "asr_paper_filtered_shared_benign",
    # 逐客户端**自己的**留出分片上的干净准确率（= tf-dpfl 的 pm_acc 口径）。
    # mta_* 测在共享的**类别均衡**探针上，那是比较 PFL 方法的错误仪器 ——
    # 它系统性惩罚 FedRep 的私有头而不惩罚 FedBN 的全局头。见 _local_acc。
    "acc_local_personalized", "acc_local_malicious",
    "acc_local_personalized_shared",
    "clean_loss_personalized", "clean_loss_global",
    "acc_target_class_personalized", "acc_target_class_global",
    "mask_keep_ratio", "effective_trim_n", "zero_sign_ratio",
    "n_malicious_this_round", "rounds_since_last_malicious",
    "n_eval_clients", "n_eval_clients_participated_this_round",
    # 正对照（--eval-include-malicious）：恶意客户端自己的模型
    "acc_malicious_own", "asr_malicious_own", "n_eval_malicious",
]

# 逐 edge（本仓库是扁平 FL，edge == 客户端）的评估行
EDGE_COLUMNS = [
    "round", "defense", "seed", "bad_client_num", "client_id", "is_malicious",
    "participated_this_round", "acc", "mta", "clean_loss",
    "acc_target_class", "asr_targeted", "asr_unfiltered",
    # 论文口径 ASR（原始触发器 + 本客户端自己 test loader，不过滤目标类）
    "asr_paper",
    # B 线：停攻点冻结触发器后的逐客户端 ASR（停攻前 / 非 persist 跑为 nan）
    "asr_paper_frozen",
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
    # 全局恶意客户端总数（不是当轮选中的恶意数，后者是 n_malicious_this_round）。
    # −1 表示调用方没传，跨密度分析会明确报错而不是猜。
    bad_client_num: int = -1
    poison_rate: float = float("nan")
    schedule_kind: str = "continuous"
    # 逐层指标要多扫一遍全部 key（ResNet-18 上约 60 个），默认关闭
    layer_metrics: bool = False
    eval_every: int = 0
    eval_client_ids: Sequence[int] = field(default_factory=tuple)
    eval_malicious_ids: Sequence[int] = field(default_factory=tuple)
    generator_getter: Optional[Any] = None
    # 论文口径 ASR 的评估函数：``fba.use_our_attack`` 返回的 ``eval_func``
    # （即 poison_ratio=1.0 的 ``full_poison_func``，main.py:117/131）。设了它,
    # exp1 的植入 ASR 就用**原始触发器**在每个客户端自己的 test loader 上算,
    # 与论文完全同尺（见 _paper_asr）。clean run / 未提供时记为 nan。
    # 注意：perturb 的 δ/ξ 分解路径（_evaluate_model）仍在，但**只服务实验 E**
    # （它需要 δ-only/ξ-only 拆分）；exp1 的 ASR 结论一律看 asr_paper_*。
    paper_eval_func: Optional[Any] = None
    # B 线：停攻点把 (x+ξ+δ) 评估图快照下来、之后复用 —— 触发器完全固定，量的是
    # 权重是否真记住了后门（vs asr_paper_* 那种 ξ 每评估现算的"在线"版本）。
    track_frozen_trigger: bool = False
    probe: Any = None
    target_class: int = 0
    num_classes: int = 10
    model_size: int = 10

    # 运行时状态
    cur_round: int = 0
    selected_indices: Sequence[int] = field(default_factory=tuple)
    last_malicious_round: Optional[int] = None
    implantation_rows: List[Dict[str, Any]] = field(default_factory=list)
    # 逐 edge（= 客户端）的评估行，一行一个 (round, client)
    edge_rows: List[Dict[str, Any]] = field(default_factory=list)
    last_mta: Optional[float] = None
    # 本轮攻击是否开启，由 run_fl 每轮从 AttackSchedule 写入
    attack_active: bool = True
    _handlers: List[Any] = field(default_factory=list)
    # B 线运行时状态：停攻点的触发器图快照（cid -> [(perturbed_cpu, labels_cpu)]）
    _frozen_batches: Dict[int, Any] = field(default_factory=dict)
    _frozen_taken: bool = False
    _attack_ever_active: bool = False
    # 墙钟（Stage B 标定）：train_wall_s = 本轮训练+聚合；eval_wall_s = 本轮评估。
    # 两段分开量：标定要回答的「降轮数还是提 local_steps 更划算」完全取决于
    # 二者的比例，而合成一个总数就答不了。None = 还没量到（不是 0）。
    _t_round_begin: Optional[float] = None
    _t_train_s: Optional[float] = None

    # -- 事件 -------------------------------------------------------------
    def attach(self, server: Any, clients: Sequence[Any]) -> "TrainingTracker":
        from event_emitter import fl_event_emitter

        def _on_round_begin(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            self.cur_round = int(kwargs["cur_round"])
            self.selected_indices = list(kwargs.get("selected_client_indices", []))
            self._t_round_begin = time.perf_counter()

        def _on_round_end(**kwargs):
            if kwargs.get("server", None) is not server:
                return
            # 本轮训练+聚合的墙钟，在评估开始之前就定下来（评估本身另计）。
            # 用单调钟：time.time() 会被 NTP 校时拽走。
            t0 = time.perf_counter()
            self._t_train_s = (None if self._t_round_begin is None
                               else t0 - self._t_round_begin)
            self._maybe_evaluate(server, clients)
            # 评估耗时只能事后知道，而植入行是在 _evaluate_now 里写的 ——
            # 所以回填最后一行。它存在 <=> 本轮确实评估过，正是这个数有意义的时候。
            if (self.implantation_rows
                    and self.implantation_rows[-1].get("round") == self.cur_round):
                self.implantation_rows[-1]["eval_wall_s"] = round(
                    time.perf_counter() - t0, 3)

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

        # 防御自己算过 trim 存活率就别再算一遍：argsort 在 [N, 2.4M] 上不便宜，
        # 而且同一个量算两次也是一个潜在的不一致来源。
        k = None
        if outcome.trim_survival_rate is None and np.isfinite(self.trim_alpha):
            k = trim_count(len(state_dicts), self.trim_alpha)
        signals = round_signals(w_prev, state_dicts, trim_k=k)

        # 分层与分组的参数指标。逐层要多扫一遍全部 key，ResNet-18 上不便宜，
        # 所以由 --layer-metrics 显式打开，默认关闭。
        layer: Dict[str, Any] = {}
        groups: Dict[str, float] = {}
        if self.layer_metrics:
            layer = layer_signals(w_prev, state_dicts, malicious)
            shared_float, _, _ = split_keys(state_dicts[0])
            groups = group_cosines(
                gram_matrix(w_prev, state_dicts, shared_float), malicious)

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
            # 全局更新范数 = 各客户端伪梯度均值的范数（= 本轮实际迈出的一步）
            global_update_norm=float(
                np.linalg.norm(layer["layer_global_update_norm"])
                if layer else np.nan),
            layer_names=layer.get("layer_names"),
            layer_update_norm=layer.get("layer_update_norm"),
            layer_global_update_norm=layer.get("layer_global_update_norm"),
            layer_cos_centroid=layer.get("layer_cos_centroid"),
            extra={**outcome.extra, **groups,
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

    def _clean_pass(self, model: Any) -> Dict[str, float]:
        """一次前向，同时给出 clean loss / MTA / 目标类准确率。

        **MTA 与既有的 ``acc`` 不是同一个数。** ``acc = 1 − error_rate``
        算的是**非目标类**样本上的准确率（``exp_e._metrics`` 的口径，为了让
        ASR 的分母与它一致）；MTA 按惯例是**全部样本**上的准确率。两者相差
        约一个类的权重。这里两个都记、各自命名 —— 把 ``acc`` 悄悄改成 MTA
        会让此前所有实验的数字不可比。
        """
        loader = self.probe.loader(int(self.cfg.probe.batch_size))
        target = int(self.target_class)
        was_training = model.training
        model.eval()
        total_loss, n_seen, n_correct = 0.0, 0, 0
        n_target, n_target_correct = 0, 0
        try:
            with torch.no_grad():
                for batch in loader:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    logits = model(images)
                    # sum 而不是 mean：最后一个 batch 通常更小，
                    # 按 batch 平均再平均会给它过高的权重
                    total_loss += float(torch.nn.functional.cross_entropy(
                        logits, labels, reduction="sum").item())
                    predicted = logits.argmax(dim=1)
                    n_correct += int((predicted == labels).sum())
                    n_seen += int(labels.numel())
                    is_target = labels == target
                    n_target += int(is_target.sum())
                    n_target_correct += int(
                        (predicted[is_target] == target).sum())
        finally:
            if was_training:
                model.train()
        return {
            "clean_loss": total_loss / n_seen if n_seen else float("nan"),
            "mta": n_correct / n_seen if n_seen else float("nan"),
            # 目标类准确率：ASR 上升时它通常也上升（后门把别的类推向目标类
            # 不影响这个数），所以它是"模型是否只是变得偏爱目标类"的对照
            "acc_target_class": (n_target_correct / n_target if n_target
                                 else float("nan")),
            "n_clean_eval_samples": n_seen,
        }

    def _paper_asr(self, model: Any, test_loader: Any) -> "Tuple[float, float]":
        """论文口径 ASR，同时返回 ``(unfiltered, filtered)``。

        逐行复刻 ``main.py:131``：给该客户端**自己的** test loader 里的每张图打
        完整触发器（``self.paper_eval_func`` = poison_ratio=1.0 的原始
        ``full_poison_func``，δ 用训练中的真生成器、ξ 用原始 ``pgd_attack``），
        标签全改 ``target_class``，统计 ``argmax == target_class`` 的比例。

        - ``unfiltered``：**不过滤目标类样本**（分母是全部样本），逐行复刻
          ``main.py:131``，是本仓库此前唯一的口径，列名 ``asr_paper_*``；
        - ``filtered``：**排除真实标签已经是目标类的样本**（分母是非目标类样本），
          即社区通行口径，列名 ``asr_paper_filtered_*``。

        为什么两个都要：本仓库的主 ASR 一直是 unfiltered，而 tf-dpfl 那边
        （``attack/backdoor_eval.py:41``）是 filtered，报告 §2 写的定义也是
        「排除目标类」——**两库的数字此前不在同一个分母上，却被并排讨论**。
        两个口径在**同一次前向**里就能同时算出来（原始标签本来就在手边），
        所以补这一列是零额外机时；此前只有 ``recompute_asr_final`` 能出 filtered，
        而它只能算最终轮（逐轮的 per-client 模型没有存）。

        CIFAR-10 均匀分布下目标类约占 1/10，所以
        ``filtered ≈ (unfiltered − 0.1) / 0.9``，两者会有约 10 个百分点的系统差 ——
        这正是「口径不一致」能造成的量级。
        - ``model`` 强制 ``eval()``（``utils.evaluate_accuracy`` 也是 eval），
          返回前恢复；
        - ``full_poison_func`` 内部的 PGD 需要梯度,故**不**套 ``no_grad``;
          仅前向预测套 ``no_grad``。整个调用在 ``preserve_rng_state`` 下发生
          （见 _maybe_evaluate），PGD 的随机起点不会泄漏进训练随机流。
        - 返回 ``[0, 1]`` 的比例（不是百分比）；缺 loader / 无攻击时返回 nan。
        """
        if self.paper_eval_func is None or test_loader is None:
            return float("nan"), float("nan")
        target = int(self.target_class)
        was_training = model.training
        model.eval()
        n_hit, n_total = 0, 0                 # unfiltered：分母 = 全部样本
        n_hit_f, n_total_f = 0, 0             # filtered：分母 = 真实标签非目标类
        try:
            for batch in test_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                # 原始触发器：pgd(client.local_model) + trigger_gen(data)。
                # 返回 (poison_data, all-target-label)，标签这里用不到——直接
                # 比 target，与 evaluate_accuracy 的 pred==transformed_label 等价。
                perturbed, _ = self.paper_eval_func(images, labels)
                with torch.no_grad():
                    preds = model(perturbed).argmax(dim=1)
                hit = (preds == target)
                n_hit += int(hit.sum())
                n_total += int(labels.numel())
                # 同一次前向里顺手算 filtered —— 零额外开销
                eligible = (labels != target)
                n_hit_f += int((hit & eligible).sum())
                n_total_f += int(eligible.sum())
        finally:
            if was_training:
                model.train()
        unfiltered = n_hit / n_total if n_total else float("nan")
        filtered = n_hit_f / n_total_f if n_total_f else float("nan")
        return unfiltered, filtered

    def _snapshot_frozen_trigger(self, clients_to_cache: Sequence[Any]) -> None:
        """停攻点：把每个客户端的 (x+ξ+δ) 评估图快照到 CPU，之后复用。

        快照发生的**当轮**，模型正是停攻时的状态，所以此刻 frozen 与 online 相等
        （k=0 对齐）；之后模型继续干净训练，frozen 用固定图、online 每轮重算 ξ,
        两者分叉 —— 分叉量 = ASR 里有多少靠 ξ 在线对抗撑着、多少是权重记住的。
        """
        self._frozen_batches = {}
        for client in clients_to_cache:
            loader = getattr(client, "test_dataloader", None)
            if self.paper_eval_func is None or loader is None:
                continue
            model = client.local_model
            was_training = model.training
            model.eval()
            batches: List[Any] = []
            try:
                for batch in loader:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    perturbed, _ = self.paper_eval_func(images, labels)
                    batches.append((perturbed.detach().cpu(),
                                    labels.detach().cpu()))
            finally:
                if was_training:
                    model.train()
            self._frozen_batches[int(client.cid)] = batches
        self._frozen_taken = True

    def _paper_asr_frozen(self, model: Any, cid: int) -> float:
        """用快照的固定触发器图评估当前 model 的 ASR（不过滤目标类）。"""
        batches = self._frozen_batches.get(int(cid))
        if not batches:
            return float("nan")
        target = int(self.target_class)
        was_training = model.training
        model.eval()
        n_correct, n_total = 0, 0
        try:
            with torch.no_grad():
                for perturbed_cpu, labels_cpu in batches:
                    preds = model(perturbed_cpu.to(self.device)).argmax(dim=1)
                    n_correct += int((preds == target).sum())
                    n_total += int(labels_cpu.numel())
        finally:
            if was_training:
                model.train()
        return n_correct / n_total if n_total else float("nan")

    def _maybe_frozen_asr(self, eval_clients: Sequence[Any],
                          malicious_clients: Sequence[Any]) -> Dict[int, float]:
        """B 线：需要时在停攻点快照，返回逐客户端的冻结触发器 ASR。

        没开 ``track_frozen_trigger`` 或还没停攻时返回空 dict（列记 nan）。
        """
        if not self.track_frozen_trigger:
            return {}
        both = list(eval_clients) + list(malicious_clients)
        stopped = self._attack_ever_active and not bool(self.attack_active)
        self._attack_ever_active = (self._attack_ever_active
                                    or bool(self.attack_active))
        if stopped and not self._frozen_taken:
            self._snapshot_frozen_trigger(both)
            # 快照里的 PGD 会把梯度累加进被绑定的恶意客户端模型，清掉
            for client in both:
                model = getattr(client, "local_model", None)
                if model is not None:
                    model.zero_grad(set_to_none=True)
        if not self._frozen_taken:
            return {}
        return {int(c.cid): self._paper_asr_frozen(c.local_model, int(c.cid))
                for c in both}

    def _local_acc(self, model: Any, loader: Any) -> float:
        """在**该客户端自己的** test loader 上的干净准确率。

        # 为什么必须另立一列（2026-09-09 实测逼出来的）

        `mta_personalized` 测在 `self.probe` —— 一份**共享的、类别均衡的**探针集
        （`config.yaml:164`，`n_other_per_class: 200` 是刻意配平的）。那份探针对
        实验 A 的归因是对的（只让模型变、数据不变），但**它是比较 PFL 方法的错误
        仪器**：

            FedBN  的个性化模型带的是**全局聚合的分类头** → 本来就是全局分类器，
                   在均衡集上表现好；
            FedRep 的个性化模型带的是**该客户端的私有头** → Dirichlet α=0.5 下
                   只见过 3–4 个主类，被要求在 10 类均衡集上判 → 塌。

        FedRep 优化的恰恰是「在**自己的**分布上准」，用全局均衡集去量它，
        量的是它没在优化的东西。而 `asr_paper_*` 一直用的是客户端自己的
        test loader —— **两个数字此前不在同一个 population**
        （tf-dpfl 陷阱 #11 的同一类错误）。

        本列与 tf-dpfl 的 `pm_acc` 同口径（逐客户端留出分片），两库这才能同框。
        缺 loader → nan，不是 0（铁律 #5）。
        """
        if loader is None:
            return float("nan")
        from utils import evaluate_accuracy      # main.py 自己用的那一个
        was_training = model.training
        try:
            # ⚠️ `utils.evaluate_accuracy` 返回的是**百分数**
            # （`utils.py:47` 是 `100 * correct / total`），而本 CSV 里其它列
            # 一律是 [0, 1] 的比例。不换算的话这一列会比邻列大 100 倍 ——
            # 第一版就漏了，实测印出 59.3056 / 73.2986。
            return float(evaluate_accuracy(model, loader)) / 100.0
        finally:
            model.train(was_training)

    def _fedrep_shared_pm(self, server: Any, client: Any):
        """FedRep 定义的个性化模型：**[当前共享表示 φ, 该客户端的私有头 h]**。

        # 为什么需要它（2026-09-09，探针 B1/B2/B3 逼出来的）

        上游一轮的顺序是：``distribute_model()`` → 客户端 ``local_update()`` ×
        local_steps → ``agg_and_update()`` → ``on_round_end``（**评估在这里**）。
        于是 ``client.local_model`` 在评估时是 **[阶段 2 漂移过的 φ′, 阶段 1 针对
        φ_t 训出来的 h]** —— 头和骨干**不配套**。

        FedBN 臂没有这个问题：它的 15 步**同时**训头和骨干，两者共适应。
        FedRep 臂的头是对着**收到的** φ_t 拟合的，之后骨干被移走了。
        实测征兆（100/10/300 轮/ρ=0.2 的探针）：

            fedbn  15 步          MTA 0.6412
            fedrep 15 步  (1:1)   MTA 0.3966
            fedrep 90 步  (5:1)   MTA 0.5334
            fedrep 165 步 (10:1)  MTA 0.4874   ← 比 5:1 还低

        三个 fedrep 全部低于 fedbn，**尽管 B2/B3 的骨干拿到的步数与 fedbn 一样多、
        头还额外多拿 75/150 步**；而且头训得越久反而越差 —— 头对 φ_t 拟合得越紧，
        骨干一动惩罚越大。这两件只有「失配」解释得通。

        FedRep 里漂移后的 φ′ **只是上传物**，不是个性化模型。tf-dpfl 侧一直是
        对的（``client/hier_fedrep.py``：``pm_weights = [w.copy() for w in
        self.edge_weights]``，用**收到的** backbone）。

        # 为什么用 server.global_model 而不是缓存收到的 φ_t

        评估在聚合之后，``server.global_model`` 就是当前的共享表示 φ_{t+1}；
        缓存 φ_t 要给 100 个客户端各存一份 backbone（约 20 MB × 100）。
        两者只差一个聚合步，而 φ′ 是**朝着该客户端自己的少数类**漂移的，
        差距不是一个量级。

        **只在 FedRep 臂上有定义**：FedBN 臂的全局 BN 停在初始化值（README §4b），
        拼出来的不是可用模型 → 返回 None，对应的列留 nan 而不是 0（铁律 #5）。
        """
        if not hasattr(client, "_fedrep_head_steps"):
            return None
        # `get_resnet` 在本模块是**函数内局部 import**（见 _evaluate_now:586 与
        # 模块 docstring 的说明），不在模块作用域 —— 新方法必须自己 import。
        from resnet import get_resnet

        from diag.pfl_fedrep import merge_shared_and_private

        merged = merge_shared_and_private(server.global_model.state_dict(),
                                          client.local_model.state_dict())
        model = get_resnet(size=self.model_size, num_classes=self.num_classes)
        model.load_state_dict(merged, strict=True)
        model.to(self.device)
        model.device = self.device
        return model

    def _evaluate_model(self, model: Any) -> Dict[str, float]:
        from .exp_e import _metrics, _predict, evaluate_mode

        batch_size = int(self.cfg.probe.batch_size)
        clean = evaluate_mode(model, self.probe, "none", device=self.device,
                              cfg=self.cfg, batch_size=batch_size)
        row = {"acc": 1.0 - float(clean["error_rate"]),
               "asr_targeted": float("nan"),
               "asr_unfiltered": float("nan")}
        row.update(self._clean_pass(model))
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
        model = get_resnet(size=self.model_size, num_classes=self.num_classes)
        model.load_state_dict(self._borrowed_bn_state(server, donor), strict=True)
        model.to(self.device)
        model.device = self.device
        global_row = self._evaluate_model(model)
        del model

        raw = get_resnet(size=self.model_size, num_classes=self.num_classes)
        raw.load_state_dict(server.global_model.state_dict(), strict=True)
        raw.to(self.device)
        raw.device = self.device
        acc_global_raw = self._evaluate_model(raw)["acc"]
        del raw

        # --- 个性化模型 ---
        rows = [self._evaluate_model(client.local_model) for client in eval_clients]

        # --- 并行的第二种组合：[当前共享表示, 私有头]（见 _fedrep_shared_pm）---
        #   **不替换上面那一列**，而是并排量一遍：哪种组合才是 FedRep 的个性化模型
        #   要由数据判，不是由我改个口径就宣布对了（铁律 #2）。
        #   非 FedRep 臂上 _fedrep_shared_pm 返回 None → 整组留空。
        shared_rows, shared_paper, shared_paper_filtered = [], {}, {}
        shared_local_acc: Dict[int, float] = {}
        for client in eval_clients:
            pm = self._fedrep_shared_pm(server, client)
            if pm is None:
                continue
            loader = getattr(client, "test_dataloader", None)
            shared_rows.append(self._evaluate_model(pm))
            shared_local_acc[int(client.cid)] = self._local_acc(pm, loader)
            u, f = self._paper_asr(pm, loader)
            shared_paper[int(client.cid)] = u
            shared_paper_filtered[int(client.cid)] = f
            del pm
        selected_ids = {int(clients[i].cid) for i in self.selected_indices}
        participated = sum(1 for c in eval_clients if int(c.cid) in selected_ids)

        def _mean(key: str, source=rows) -> float:
            values = [r[key] for r in source if np.isfinite(r[key])]
            return float(np.mean(values)) if values else float("nan")

        # 正对照：恶意客户端自己的个性化模型。它是直接在投毒数据上训练的，
        # ASR 应当很高 —— 若它也低，说明投毒本身没生效，而不是防御起了作用。
        malicious_clients = [by_id[int(i)] for i in self.eval_malicious_ids
                             if int(i) in by_id]
        malicious_rows = [self._evaluate_model(c.local_model)
                          for c in malicious_clients]

        malicious_now = sum(
            1 for i in self.selected_indices
            if bool(getattr(clients[i], "diag_is_malicious", False)))

        # --- 论文口径 ASR（原始触发器 + 各客户端自己的 test loader） ---
        # 对已取到的良性评估子集 + 恶意客户端逐个算。asr_paper_all 是两组合并的
        # 均值，近似 main.py 遍历全体客户端的 "Avg ASR"（这里是评估子集上的近似,
        # 不是全 40 个客户端；要全量就把 eval_client_ids 放到全部良性）。
        paper_by_cid: Dict[int, float] = {}
        paper_filtered_by_cid: Dict[int, float] = {}
        local_acc_by_cid: Dict[int, float] = {}
        for client in eval_clients + malicious_clients:
            loader = getattr(client, "test_dataloader", None)
            unfiltered, filtered = self._paper_asr(client.local_model, loader)
            paper_by_cid[int(client.cid)] = unfiltered
            paper_filtered_by_cid[int(client.cid)] = filtered
            # 同一个 loader 上的干净准确率 —— 与 ASR 同 population（见 _local_acc）
            local_acc_by_cid[int(client.cid)] = self._local_acc(
                client.local_model, loader)
        # full_poison_func 内部 PGD 会把梯度累加进被绑定的恶意客户端模型；
        # client.py 在下次本地训练取数前会 zero_grad，本无副作用，但按诊断惯例
        # 主动清掉,避免与 layer_metrics 等其它读操作相互干扰。
        for client in clients:
            model = getattr(client, "local_model", None)
            if model is not None:
                model.zero_grad(set_to_none=True)

        def _mean_over(source: Dict[int, float], cids: Sequence[int]) -> float:
            values = [source[i] for i in cids
                      if np.isfinite(source.get(i, float("nan")))]
            return float(np.mean(values)) if values else float("nan")

        benign_cids = [int(c.cid) for c in eval_clients]
        malicious_cids = [int(c.cid) for c in malicious_clients]
        asr_paper_benign = _mean_over(paper_by_cid, benign_cids)
        asr_paper_malicious = _mean_over(paper_by_cid, malicious_cids)
        asr_paper_all = _mean_over(paper_by_cid, benign_cids + malicious_cids)
        # 同一批前向里算出的 filtered 口径（排除真实标签已是目标类的样本）。
        # 与 tf-dpfl 的 backdoor_eval.py:41 同分母，两库这才能同框比较。
        asr_paper_filtered_benign = _mean_over(paper_filtered_by_cid, benign_cids)
        asr_paper_filtered_malicious = _mean_over(paper_filtered_by_cid, malicious_cids)
        asr_paper_filtered_all = _mean_over(paper_filtered_by_cid,
                                            benign_cids + malicious_cids)

        # [当前共享表示, 私有头] 这一组合下的同一批指标（仅 FedRep 臂）
        def _mean_shared(key: str) -> float:
            values = [r[key] for r in shared_rows if np.isfinite(r[key])]
            return float(np.mean(values)) if values else float("nan")

        mta_shared = _mean_shared("mta")
        acc_local_personalized = _mean_over(local_acc_by_cid, benign_cids)
        acc_local_malicious = _mean_over(local_acc_by_cid, malicious_cids)
        acc_local_shared = _mean_over(shared_local_acc, benign_cids)
        asr_paper_shared_benign = _mean_over(shared_paper, benign_cids)
        asr_paper_filtered_shared_benign = _mean_over(shared_paper_filtered,
                                                      benign_cids)

        # --- B 线：冻结触发器 ASR（停攻点快照后才有值） ---
        frozen_by_cid = self._maybe_frozen_asr(eval_clients, malicious_clients)

        def _frozen_mean(cids: Sequence[int]) -> float:
            values = [frozen_by_cid[i] for i in cids
                      if np.isfinite(frozen_by_cid.get(i, float("nan")))]
            return float(np.mean(values)) if values else float("nan")

        asr_paper_frozen_benign = _frozen_mean(benign_cids)
        asr_paper_frozen_malicious = _frozen_mean(malicious_cids)
        asr_paper_frozen_all = _frozen_mean(benign_cids + malicious_cids)

        # --- 逐 edge 的行 ---
        # 本仓库是扁平 FL，没有中间聚合层，所以 "edge" 就是客户端。真有层级
        # 结构时这里要多一列 edge_id，均值也要改成先组内再组间。
        for client, row in list(zip(eval_clients, rows)) + \
                list(zip(malicious_clients, malicious_rows)):
            self.edge_rows.append({
                "round": self.cur_round, "defense": self.defense_name,
                "seed": self.seed, "bad_client_num": int(self.bad_client_num),
                "client_id": int(client.cid),
                "is_malicious": bool(getattr(client, "diag_is_malicious", False)),
                "participated_this_round": int(client.cid) in selected_ids,
                "acc": row["acc"], "mta": row["mta"],
                "clean_loss": row["clean_loss"],
                "acc_target_class": row["acc_target_class"],
                "asr_targeted": row["asr_targeted"],
                "asr_unfiltered": row["asr_unfiltered"],
                "asr_paper": paper_by_cid.get(int(client.cid), float("nan")),
                "asr_paper_frozen": frozen_by_cid.get(int(client.cid),
                                                      float("nan")),
            })

        # MTA 供 after_mta 调度读取。**只有评估过的轮次才更新** ——
        # 调度因此最多滞后 eval_every 轮，这一点记在 schedule.py 里。
        self.last_mta = _mean("mta")

        self.implantation_rows.append({
            "round": self.cur_round, "defense": self.defense_name,
            "bad_client_num": int(self.bad_client_num),
            "poison_rate": self.poison_rate,
            "schedule": self.schedule_kind,
            "attack_active_this_round": bool(self.attack_active),
            "tau": self.tau, "alpha": self.trim_alpha,
            "alpha_dirichlet": self.alpha_dirichlet, "seed": self.seed,
            "variant": self.variant or self.defense_name,
            "acc_global": global_row["acc"],
            "acc_global_raw": acc_global_raw,
            "acc_personalized": _mean("acc"),
            "asr_global_targeted": global_row["asr_targeted"],
            "asr_personalized_targeted": _mean("asr_targeted"),
            "asr_unfiltered": _mean("asr_unfiltered"),
            # 论文口径（exp1 的主 ASR）
            "asr_paper_benign": asr_paper_benign,
            "asr_paper_filtered_benign": asr_paper_filtered_benign,
            "asr_paper_filtered_malicious": asr_paper_filtered_malicious,
            "asr_paper_filtered_all": asr_paper_filtered_all,
            "asr_paper_malicious": asr_paper_malicious,
            "asr_paper_all": asr_paper_all,
            # B 线：冻结触发器 ASR（停攻前 / 未开该模式为 nan）
            "asr_paper_frozen_benign": asr_paper_frozen_benign,
            "asr_paper_frozen_malicious": asr_paper_frozen_malicious,
            "asr_paper_frozen_all": asr_paper_frozen_all,
            # 主任务指标。mta 是**全部样本**上的准确率，acc_personalized 是
            # 非目标类样本上的（与 ASR 的分母同口径）—— 两者不是同一个数。
            "mta_personalized": _mean("mta"),
            "mta_global": global_row["mta"],
            "mta_personalized_shared": mta_shared,
            "acc_local_personalized": acc_local_personalized,
            "acc_local_malicious": acc_local_malicious,
            "acc_local_personalized_shared": acc_local_shared,
            "asr_paper_shared_benign": asr_paper_shared_benign,
            "asr_paper_filtered_shared_benign": asr_paper_filtered_shared_benign,
            "clean_loss_personalized": _mean("clean_loss"),
            "clean_loss_global": global_row["clean_loss"],
            "acc_target_class_personalized": _mean("acc_target_class"),
            "acc_target_class_global": global_row["acc_target_class"],
            "mask_keep_ratio": float("nan"), "effective_trim_n": -1,
            "zero_sign_ratio": float("nan"),
            "n_malicious_this_round": malicious_now,
            "rounds_since_last_malicious": (
                self.cur_round - self.last_malicious_round
                if self.last_malicious_round is not None else -1),
            "n_eval_clients": len(eval_clients),
            "n_eval_clients_participated_this_round": participated,
            "acc_malicious_own": _mean("acc", malicious_rows),
            "asr_malicious_own": _mean("asr_targeted", malicious_rows),
            "n_eval_malicious": len(malicious_clients),
            # 墙钟（Stage B 标定）。train 在本行写死，eval 由 _on_round_end 回填
            # —— 它只有等 _evaluate_now 返回才知道。两者都不量时留 nan，不填 0。
            "train_wall_s": (float("nan") if self._t_train_s is None
                             else round(self._t_train_s, 3)),
            "eval_wall_s": float("nan"),
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

    def edge_frame(self):
        """逐 edge（客户端）的评估行 —— "attack success on each Edge"。

        均值会掩盖分布：90 个客户端里 10 个 ASR=1.0、80 个 ASR=0.0 与全部
        ASR=0.11 的均值一模一样，而两者的含义完全相反。
        """
        import pandas as pd

        if not self.edge_rows:
            return pd.DataFrame(columns=EDGE_COLUMNS)
        frame = pd.DataFrame(self.edge_rows)
        rest = [c for c in frame.columns if c not in EDGE_COLUMNS]
        return frame[[c for c in EDGE_COLUMNS if c in frame.columns] + rest]
