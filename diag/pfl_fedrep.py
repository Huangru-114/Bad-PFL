"""FedRep（Collins et al., ICML 2021）—— 旁路实现，上游文件零 diff。

# 为什么需要它

上游 ``pfl.py`` 里**只有 FedBN**（`use_fedbn`，`pfl.py:3-24`），Ditto / FedProx /
FedRep 全都不存在。而 tf-dpfl 那边的 Experiment 3 跑的是 ``hier_fedrep``。
两个库的「个性化」因此几乎是**互补**的，两组 ASR 数字在语义上不可直接比较：

    FedBN  ：个性化**归一化**（BN 的 γ/β 与 running stats 私有），分类头**共享**
    FedRep ：个性化**分类头**（`linear.*` 私有），BN 参与**聚合**

本模块把 FedRep 补进 Bad-PFL 侧，让两个库能在同一个 PFL 框架下对照。

# 铁律：上游文件零 diff

所以这里不改 ``pfl.py`` / ``client.py``，而是：

- 服务器侧：``use_fedrep(server)`` 与 ``pfl.py::use_fedbn`` **同构** —— 挂
  ``before_update_global`` / ``before_distribute_global`` 两个钩子，把私有 key
  从 ``server.update`` 和 ``server.distribute_dict`` 里 pop 掉；
- 客户端侧：``FedRepMixin`` 与客户端类**组合**（不是替换）。
  ``diag/run_fl.py:317-334`` 自己构造客户端，所以换个类即可。

  ⚠️ **组合而不是继承替换**，是 tf-dpfl 陷阱 #1 的教训：那边曾用
  ``use_cls = MalCls or ClientCls`` 让恶意客户端类**顶掉** PFL 方法类，
  于是良性端跑 PFL 方法、恶意端跑朴素 FedAvg，上传语义不同，
  足以单独解释 ASR≈0。这里 MRO 是 ``FedRepMixin → PoisonClient → BasicClient``，
  投毒行为原样保留。

# 私有 key 的选择

``resnet.py:87`` 是 ``self.linear = nn.Linear(...)``，所以头的 key 是
``linear.weight`` / ``linear.bias``。判定写成子串匹配以与 ``pfl.py`` 的风格一致。

**BN 不在私有集合里** —— 这是与 tf-dpfl 对齐的关键点：那边
``models/cnn.py:250-252`` 明确把 BN 的 γ/β 与 moving_mean/variance 都归进
backbone、参与聚合。

一个连带后果：FedRep 下 ``server.global_model`` 的 BN **会**被更新，
于是全局模型是个可用的模型 —— 与 FedBN 臂**完全相反**（那边 BN 从初始化起
一位不变，见 ``diag/README.md`` §4b）。所以两条臂的 ``acc_global`` 不可同框，
评估侧的「借 BN」机制（``diag/track.py:279-287``）在 FedRep 臂上是多余的。

# 两阶段本地训练

FedRep 的本地更新分两段：**先训头（表示冻结），再训表示（头冻结）**。
上游的 ``fl_process.basic_fl_process:29-30`` 是

    for local_step in range(local_steps):
        clients[indice].local_update()

它并不调用 ``local_fine_tuning``，也不把 ``local_steps`` 告诉客户端。
所以相位切换做在 ``local_update`` 内部，靠一个每轮在 ``receive_model`` 里
清零的步计数器；``local_steps`` 由 ``run_fl.py`` 在构造后用
``configure_fedrep()`` 注入。

**头阶段里 backbone 是彻底冻结的**：不只 ``requires_grad=False``，还把 backbone
的 BN 模块切到 ``eval()``。否则 forward 仍会更新 running stats —— 那样「冻结」
就名不副实，而 BN 在本方案里是要聚合的，这点差异会直接进到上传物里。
守卫：``diag/tests/test_pfl_fedrep.py`` 断言头阶段后 backbone 的**参数与
buffer 都逐位不变**。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# torch 是**软依赖**：`FedRepMixin` 的方法体需要它，但 key 判定与相位切分这两段
# 纯逻辑不需要。守着 import 让本模块在没装 torch 的机器上仍可 import，
# 于是 `is_fedrep_private_key` / `split_keys_fedrep` / `default_head_steps`
# 这三个**决定「这到底还是不是 FedRep」**的函数能被本地 L1 覆盖，
# 而不必等一台有 torch 的机器。判错一个 key 不会报错，只会让分类头被悄悄聚合。
#
# （`from __future__ import annotations` 已把类型注解变成字符串，所以签名里
#  出现 torch.Tensor 也不会在 import 期求值。）
try:
    import torch
    import torch.nn as nn
except ImportError:                                   # pragma: no cover
    torch = None                                      # type: ignore[assignment]
    nn = None                                         # type: ignore[assignment]

__all__ = [
    "FEDREP_HEAD_SUBSTRINGS",
    "is_fedrep_private_key",
    "split_keys_fedrep",
    "use_fedrep",
    "FedRepMixin",
    "make_fedrep_class",
    "default_head_steps",
]

# resnet.py:87 —— self.linear = nn.Linear(512*expansion, num_classes)
FEDREP_HEAD_SUBSTRINGS: Tuple[str, ...] = ("linear",)


def is_fedrep_private_key(key: str) -> bool:
    """该 key 在 FedRep 下是否为客户端私有（分类头，不参与聚合）。"""
    return any(token in key for token in FEDREP_HEAD_SUBSTRINGS)


def split_keys_fedrep(state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """把 state_dict 分成 ``(shared, head_private)``。

    与 ``diag/fedbn.split_keys`` 不同，这里**不再区分浮点/非浮点**：
    FedRep 下 ``num_batches_tracked``（整数 buffer）是**共享**的，
    调用方若要做 sign / trimmed-mean 需自行按 dtype 过滤。
    """
    shared, private = [], []
    for key in state:
        (private if is_fedrep_private_key(key) else shared).append(key)
    return sorted(shared), sorted(private)


def default_head_steps(local_steps: int) -> int:
    """默认的头阶段步数 = 总步数的一半（至少 1，且至少给 backbone 留 1 步）。

    tf-dpfl 侧是 ``plocal_epochs=1`` 头 + ``local_epochs=1`` backbone，即 1:1。
    这里 ``local_steps`` 默认 15，对半分成 7 头 + 8 backbone。
    """
    if local_steps <= 1:
        return 0                       # 只有一步时全部给 backbone（否则表示永不更新）
    return max(1, min(local_steps - 1, local_steps // 2))


# ──────────────────────────────────────────────────────────────────────────
# 服务器侧：与 pfl.py::use_fedbn 同构
# ──────────────────────────────────────────────────────────────────────────

def merge_shared_and_private(shared: Dict[str, "torch.Tensor"],
                             private: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """拼出 FedRep 定义的个性化模型：**私有键取 private，其余取 shared**。

    抽成纯字典操作是为了能在**没有 torch 的机器上**跑 L1 —— 拼错一个键不会报错，
    只会让评估悄悄测在错的模型上（这正是 2026-09-09 探针暴露的那类问题）。

    键集合以 ``shared`` 为准：私有键在 private 里必须存在，缺了直接 KeyError
    而不是静默回退到 shared（回退等于把私有头换成聚合过的头，那就不是 FedRep 了）。
    """
    return {k: (private[k] if is_fedrep_private_key(k) else v)
            for k, v in shared.items()}


def use_fedrep(server) -> None:
    """把分类头从聚合与下发中剔除。

    逐字对照 ``pfl.py:3-24`` 的 ``use_fedbn``，只换判定条件。
    挂载点相同（``before_update_global`` / ``before_distribute_global``），
    所以 ``server.py:32-34`` 的 ``load_state_dict(..., strict=False)``
    行为也相同：被 pop 的 key 不会被覆盖。
    """

    def fedrep_update(server):
        delete_keys = [k for k in server.update.keys() if is_fedrep_private_key(k)]
        for key in delete_keys:
            server.update.pop(key)

    def fedrep_distribute(server):
        delete_keys = [k for k in server.distribute_dict.keys()
                       if is_fedrep_private_key(k)]
        for key in delete_keys:
            server.distribute_dict.pop(key)

    server.register_func(fedrep_update, "before_update_global")
    server.register_func(fedrep_distribute, "before_distribute_global")


# ──────────────────────────────────────────────────────────────────────────
# 客户端侧：两阶段本地训练
# ──────────────────────────────────────────────────────────────────────────

class FedRepMixin:
    """两阶段本地更新：先训头（backbone 冻结），再训 backbone（头冻结）。

    与客户端类**组合**使用，不要单独实例化：

        FedRepPoisonClient = make_fedrep_class(PoisonClient)
    """

    #: 由 configure_fedrep() 注入；None 表示还没配置（此时退化为普通训练并告警一次）
    _fedrep_head_steps: Optional[int] = None
    _fedrep_step: int = 0
    _fedrep_warned: bool = False

    # ---- 配置 -------------------------------------------------------------

    def configure_fedrep(self, local_steps: int, head_steps: Optional[int] = None) -> None:
        """注入相位切分。``run_fl.py`` 在构造客户端之后调用。

        分开成一个方法而不是塞进 ``__init__``，是为了不碰上游 ``BasicClient``
        的构造签名（铁律 #1）。
        """
        self._fedrep_local_steps = int(local_steps)
        self._fedrep_head_steps = (default_head_steps(int(local_steps))
                                   if head_steps is None else int(head_steps))
        if not 0 <= self._fedrep_head_steps <= self._fedrep_local_steps:
            raise ValueError(
                f"head_steps={self._fedrep_head_steps} 超出 [0, {self._fedrep_local_steps}]")

    # ---- 参数分组 ---------------------------------------------------------

    def _fedrep_head_params(self) -> List["nn.Parameter"]:
        return [p for n, p in self.local_model.named_parameters()
                if is_fedrep_private_key(n)]

    def _fedrep_body_params(self) -> List["nn.Parameter"]:
        return [p for n, p in self.local_model.named_parameters()
                if not is_fedrep_private_key(n)]

    def _fedrep_body_norm_modules(self) -> List["nn.Module"]:
        """backbone 里所有会更新 running stats 的归一化模块。"""
        if nn is None:                                # pragma: no cover
            raise RuntimeError(
                "FedRepMixin 需要 torch —— 本模块的 torch import 是软依赖，"
                "只为让 key 判定这段纯逻辑能在没有 torch 的机器上被测到。")
        return [m for m in self.local_model.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.SyncBatchNorm))]

    def _fedrep_set_phase(self, phase: str) -> None:
        """``phase`` ∈ {"head", "body"}。"""
        train_head = (phase == "head")
        for p in self._fedrep_head_params():
            p.requires_grad_(train_head)
        for p in self._fedrep_body_params():
            p.requires_grad_(not train_head)
        # 头阶段：backbone 必须**彻底**冻结。只关 requires_grad 不够 ——
        # forward 仍会推进 BN 的 running_mean / running_var，而 BN 在 FedRep 下
        # 是要上传聚合的，那部分改动会静默进到上传物里。
        for m in self._fedrep_body_norm_modules():
            m.train(not train_head)
        self._fedrep_phase = phase

    # ---- 训练循环钩子 -----------------------------------------------------

    def receive_model(self, global_state_dict):
        """每轮开始：重置相位计数器。

        必须先 ``super()``：``BasicClient.receive_model``（``client.py:23-25``）
        会触发 ``before_local_training``，Bad-PFL 的生成器训练就挂在那里
        （``fba.py:62``）。相位设置放在它**之后**，免得生成器的 30 步
        在一个被我们改过 requires_grad 的模型上跑。
        """
        super().receive_model(global_state_dict)
        self._fedrep_step = 0
        if self._fedrep_head_steps is None:
            if not self._fedrep_warned:
                print("[FedRep] ⚠️ 未调用 configure_fedrep()，本客户端退化为普通"
                      "单阶段训练（头与 backbone 一起训）。检查 run_fl.py 的接线。")
                self._fedrep_warned = True
            return
        self._fedrep_set_phase("head" if self._fedrep_head_steps > 0 else "body")

    def local_update(self):
        if self._fedrep_head_steps is None:
            return super().local_update()
        phase = "head" if self._fedrep_step < self._fedrep_head_steps else "body"
        if getattr(self, "_fedrep_phase", None) != phase:
            self._fedrep_set_phase(phase)
        super().local_update()
        self._fedrep_step += 1

    def upload_model(self):
        """上传前把所有参数解冻。

        上传物本身是完整 state_dict（头会在服务器侧被 ``use_fedrep`` pop 掉），
        但把模型停在「一半参数 requires_grad=False」的状态会污染后续任何
        诊断路径（例如快照、特征抽取）。解冻是纯状态清理，不改数值。
        """
        try:
            return super().upload_model()
        finally:
            for p in self.local_model.parameters():
                p.requires_grad_(True)
            for m in self._fedrep_body_norm_modules():
                m.train(True)


def make_fedrep_class(base_cls: type) -> type:
    """把 FedRepMixin 组合进一个客户端类。

    MRO = ``FedRepMixin → base_cls → ...``，所以 ``PoisonClient`` 的
    ``fetch_data``（投毒）原样生效。
    """
    return type(f"FedRep{base_cls.__name__}", (FedRepMixin, base_cls), {})
