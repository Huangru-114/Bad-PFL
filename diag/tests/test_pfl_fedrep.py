"""FedRep 旁路实现的算法不变量（``diag/pfl_fedrep.py``）。

分两层：

* **不需要 torch** —— key 判定、相位步数切分、服务器钩子。这几段决定
  「这到底还不还是 FedRep」：判错一个 key 不会报错，只会让**分类头被悄悄聚合**，
  于是跑的其实是 FedAvg，而 ASR 曲线看起来一切正常。所以它们必须在最低配的
  机器上也能跑到。
* **需要 torch** —— 两阶段冻结的逐位不变量、与 PoisonClient 的组合。
  没有 torch 时整体跳过（打印一行说明，不静默通过）。

对照关系：tf-dpfl 的 ``hier_fedrep`` 把 BN 的 γ/β 与 moving stats 都算作
backbone、参与聚合（``models/cnn.py:250-252``）。本实现刻意保持一致 ——
**BN 不是私有**。这是两个库能同框比较的前提，也是与本仓库 FedBN 臂
（BN 私有、分类头共享）的分界线。
"""

from __future__ import annotations

import unittest

from diag.pfl_fedrep import (
    FEDREP_HEAD_SUBSTRINGS,
    default_head_steps,
    is_fedrep_private_key,
    make_fedrep_class,
    split_keys_fedrep,
    use_fedrep,
)

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None                                      # type: ignore[assignment]
    nn = None                                         # type: ignore[assignment]
    HAVE_TORCH = False


# ResNet-10/18 的真实 key 形状（resnet.py:81,87 与 22,25,32-33）
RESNET_KEYS = [
    "conv1.weight",
    "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var",
    "bn1.num_batches_tracked",
    "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.running_mean",
    "layer1.0.conv2.weight", "layer1.0.bn2.weight",
    "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight",
    "layer2.0.shortcut.1.running_mean",
    "linear.weight", "linear.bias",
]


# ══════════════════════════════════════════════════════════════════════════
# 第一层：不需要 torch
# ══════════════════════════════════════════════════════════════════════════

def test_only_the_classifier_head_is_private():
    shared, private = split_keys_fedrep({k: None for k in RESNET_KEYS})
    assert private == ["linear.bias", "linear.weight"], \
        f"私有集合应恰好是分类头，实际是 {private}"
    assert len(shared) == len(RESNET_KEYS) - 2


def test_batchnorm_is_shared_not_private():
    """
    与 tf-dpfl 的 hier_fedrep 对齐的**关键断言**。

    若 BN 被误判为私有，这条臂就退化成「FedBN + 私有头」，既不是 FedRep
    也不是 FedBN，而两个库的 ASR 数字会被当成可比的来读。
    """
    for key in RESNET_KEYS:
        if "bn" in key or "shortcut.1" in key:
            assert not is_fedrep_private_key(key), \
                f"{key} 被判成了私有；FedRep 下 BN 必须参与聚合"


def test_head_and_body_partition_is_exhaustive_and_disjoint():
    shared, private = split_keys_fedrep({k: None for k in RESNET_KEYS})
    assert set(shared) | set(private) == set(RESNET_KEYS)
    assert not (set(shared) & set(private))


def test_substring_rule_does_not_overmatch():
    """``linear`` 是子串匹配 —— 确认它不会误伤别的层名。"""
    assert FEDREP_HEAD_SUBSTRINGS == ("linear",)
    for key in ("conv1.weight", "bn1.weight", "layer1.0.conv1.weight"):
        assert not is_fedrep_private_key(key), f"{key} 不该被判成头"
    # 反向：真的头 key 必须被判中
    assert is_fedrep_private_key("linear.weight")
    assert is_fedrep_private_key("linear.bias")


def test_head_steps_split_leaves_work_for_both_phases():
    """
    两个相位都必须**至少有一步**，否则不是两阶段训练：
    头永不训 → 退化成 FedAvg + 冻结头；backbone 永不训 → 表示永不更新。
    """
    for total in (2, 3, 4, 15, 16, 100):
        head = default_head_steps(total)
        assert 1 <= head <= total - 1, f"local_steps={total} 时 head_steps={head}"
    # local_steps=1 是退化情形：只有一步，全给 backbone（表示必须能更新）
    assert default_head_steps(1) == 0


def test_head_steps_matches_the_one_to_one_convention():
    """tf-dpfl 侧是 plocal_epochs=1 头 + local_epochs=1 backbone，即 1:1。"""
    assert default_head_steps(15) == 7      # 7 头 + 8 backbone
    assert default_head_steps(16) == 8      # 8 头 + 8 backbone


class _FakeServer:
    """够 use_fedrep 用的最小服务器（上游 BasicServer 的钩子协议）。"""

    def __init__(self):
        self.registered = {}
        self.update = {k: object() for k in RESNET_KEYS}
        self.distribute_dict = {k: object() for k in RESNET_KEYS}

    def register_func(self, func, stage):
        self.registered.setdefault(stage, []).append(func)

    def fire(self, stage):
        for f in self.registered.get(stage, []):
            f(self)


def test_use_fedrep_registers_on_the_same_two_hooks_as_use_fedbn():
    s = _FakeServer()
    use_fedrep(s)
    assert set(s.registered) == {"before_update_global", "before_distribute_global"}, \
        f"挂载点与 pfl.py::use_fedbn 不一致：{sorted(s.registered)}"


def test_head_is_removed_from_aggregation():
    s = _FakeServer()
    use_fedrep(s)
    s.fire("before_update_global")
    assert "linear.weight" not in s.update and "linear.bias" not in s.update, \
        "分类头仍在 server.update 里 —— 它会被聚合，那就不是 FedRep 了"
    assert "bn1.running_mean" in s.update, "BN 被误删；FedRep 下 BN 要聚合"
    assert "conv1.weight" in s.update


def test_head_is_removed_from_distribution():
    """
    下发时也要剔除，否则服务器会把**别人的头**盖到本客户端的私有头上。
    （client.py:24 的 load_state_dict 用 strict=False，pop 掉即可保住私有头。）
    """
    s = _FakeServer()
    use_fedrep(s)
    s.fire("before_distribute_global")
    assert "linear.weight" not in s.distribute_dict
    assert "linear.bias" not in s.distribute_dict
    assert "bn1.weight" in s.distribute_dict


# ══════════════════════════════════════════════════════════════════════════
# 第二层：需要 torch
# ══════════════════════════════════════════════════════════════════════════

def _require_torch():
    """没有 torch 就**真的**跳过 —— 不能 return，那会被记成 PASS（假绿）。"""
    if not HAVE_TORCH:
        raise unittest.SkipTest("需要 torch（本机未安装）")


_NetBase = nn.Module if HAVE_TORCH else object


class _TinyNet(_NetBase):
    """conv+bn+linear 的最小网络，key 命名与 resnet.py 一致。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return self.linear(x.mean(dim=(2, 3)))


class _FakeBaseClient:
    """够 FedRepMixin 用的最小客户端：只提供它 super() 到的三个方法。"""

    def __init__(self, model):
        self.local_model = model
        self.received = 0
        self.updates = 0

    def receive_model(self, sd):
        self.received += 1

    def local_update(self):
        """跑一步真实的 forward/backward/step，这样冻结与否是可观测的。

        ⚠️ **必须逐行复刻 `client.py:local_update` 的相关行为**，特别是
        `self.local_model.train()` —— 第一版漏了它，于是「头阶段 BN 被
        `.train()` 拉回训练模式」这个真实 bug 在本地全绿、在集群上跑了 300 轮
        才从 MTA 上看出来。**假客户端与真客户端的差异 = 测不到的 bug。**
        """
        self.updates += 1
        self.local_model.train()             # ← client.py:local_update 每步都调
        opt = torch.optim.SGD(self.local_model.parameters(), lr=0.5)
        opt.zero_grad()
        x = torch.randn(8, 3, 4, 4)
        y = torch.randint(0, 2, (8,))
        loss = torch.nn.functional.cross_entropy(self.local_model(x), y)
        loss.backward()
        opt.step()

    def upload_model(self):
        return self.local_model.state_dict()


def _make_client(local_steps=4, head_steps=2):
    torch.manual_seed(0)
    cls = make_fedrep_class(_FakeBaseClient)
    c = cls(_TinyNet())
    c.local_model.train()
    c.configure_fedrep(local_steps=local_steps, head_steps=head_steps)
    return c


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_head_phase_leaves_backbone_bitwise_unchanged():
    """
    头阶段：backbone 的**参数与 buffer 都**必须逐位不变。

    buffer 那半条是关键 —— 只设 requires_grad=False 的话，forward 仍会推进
    BN 的 running_mean / running_var，而 BN 在 FedRep 下是要上传聚合的，
    那部分改动会静默进到上传物里。
    """
    _require_torch()
    c = _make_client(local_steps=4, head_steps=2)
    c.receive_model({})
    before = _snapshot(c.local_model)
    for _ in range(2):                       # 只走头阶段
        c.local_update()
    after = _snapshot(c.local_model)
    for k in before:
        if k.startswith("linear"):
            continue
        assert torch.equal(before[k], after[k]), \
            f"头阶段动了 backbone 的 {k}（差 {(after[k]-before[k]).abs().max()}）"
    assert not torch.equal(before["linear.weight"], after["linear.weight"]), \
        "头阶段没有更新分类头 —— 那就没在训练"


def test_body_phase_leaves_head_unchanged():
    _require_torch()
    c = _make_client(local_steps=4, head_steps=2)
    c.receive_model({})
    for _ in range(2):
        c.local_update()                     # 先把头阶段走完
    before = _snapshot(c.local_model)
    for _ in range(2):
        c.local_update()                     # backbone 阶段
    after = _snapshot(c.local_model)
    for k in ("linear.weight", "linear.bias"):
        assert torch.equal(before[k], after[k]), f"backbone 阶段动了 {k}"
    assert not torch.equal(before["conv1.weight"], after["conv1.weight"]), \
        "backbone 阶段没有更新表示"


def test_phase_counter_resets_every_round():
    """
    相位靠 receive_model 清零。不清零的话第二轮开始全是 backbone 阶段，
    私有头从第二轮起再也不训练 —— 而这不会报任何错。
    """
    _require_torch()
    c = _make_client(local_steps=4, head_steps=2)
    for rnd in range(3):
        c.receive_model({})
        assert c._fedrep_step == 0, f"第 {rnd} 轮开始时步计数器是 {c._fedrep_step}"
        before = _snapshot(c.local_model)
        c.local_update()
        after = _snapshot(c.local_model)
        assert not torch.equal(before["linear.weight"], after["linear.weight"]), \
            f"第 {rnd} 轮的第一步不是头阶段"


def test_upload_restores_requires_grad():
    """上传后必须解冻，否则模型停在半冻结状态会污染后续诊断路径。"""
    _require_torch()
    c = _make_client()
    c.receive_model({})
    c.local_update()                          # 此时 backbone 是冻的
    assert not c.local_model.conv1.weight.requires_grad
    c.upload_model()
    assert all(p.requires_grad for p in c.local_model.parameters()), \
        "upload_model 之后仍有参数是冻结的"
    assert c.local_model.bn1.training, "upload_model 之后 BN 仍停在 eval()"


def test_mro_composes_rather_than_replaces():
    """
    tf-dpfl 陷阱 #1 的教训：攻击类曾**顶掉** PFL 方法类，于是良性端跑方法、
    恶意端跑朴素 FedAvg，足以单独解释 ASR≈0。这里必须是组合。
    """
    cls = make_fedrep_class(_FakeBaseClient)
    names = [c.__name__ for c in cls.__mro__]
    assert names[:3] == ["FedRep_FakeBaseClient", "FedRepMixin", "_FakeBaseClient"], \
        f"MRO 不是「Mixin 在前、基类在后」：{names}"
    assert issubclass(cls, _FakeBaseClient), "组合后必须仍是原客户端类的子类"


def test_unconfigured_client_falls_back_and_warns():
    """
    忘了调 configure_fedrep 时必须退化成普通训练 + 告警，而不是静默地
    只训一半参数 —— 后者会伪装成「FedRep 效果不好」。
    """
    _require_torch()
    torch.manual_seed(0)
    c = make_fedrep_class(_FakeBaseClient)(_TinyNet())
    c.local_model.train()
    c.receive_model({})                       # 没有 configure_fedrep
    before = _snapshot(c.local_model)
    c.local_update()
    after = _snapshot(c.local_model)
    assert not torch.equal(before["conv1.weight"], after["conv1.weight"])
    assert not torch.equal(before["linear.weight"], after["linear.weight"]), \
        "未配置时应当头与 backbone 一起训（普通单阶段）"


# ── [当前共享表示, 私有头] 的拼装（2026-09-09 探针逼出来的）──────────────────
def test_merge_takes_the_head_from_private_and_the_rest_from_shared():
    """FedRep 定义的个性化模型 = 共享表示 + 私有头。

    评估时 `client.local_model` 是 [阶段 2 漂移过的 φ′, 阶段 1 对着 φ_t 训的 h]，
    头和骨干不配套 —— 探针实测 fedrep 三档 MTA(0.3966/0.5334/0.4874) 全部低于
    fedbn(0.6412)，且 10:1 比 5:1 更差。漂移后的 φ′ 只是**上传物**。
    """
    from diag.pfl_fedrep import merge_shared_and_private
    shared = {"conv1.weight": "S_conv", "bn1.weight": "S_bn",
              "linear.weight": "S_head_w", "linear.bias": "S_head_b"}
    private = {"conv1.weight": "P_conv", "bn1.weight": "P_bn",
               "linear.weight": "P_head_w", "linear.bias": "P_head_b"}
    merged = merge_shared_and_private(shared, private)
    assert merged["linear.weight"] == "P_head_w"   # 头：私有
    assert merged["linear.bias"] == "P_head_b"
    assert merged["conv1.weight"] == "S_conv"      # 表示：共享
    assert merged["bn1.weight"] == "S_bn", "FedRep 下 BN 参与聚合，属于共享表示"


def test_merge_is_neither_of_its_two_inputs():
    """反向锚点：拼出来的既不是纯共享也不是纯私有 —— 否则这个函数没干活。"""
    from diag.pfl_fedrep import merge_shared_and_private
    shared = {"conv1.weight": "S", "linear.weight": "S"}
    private = {"conv1.weight": "P", "linear.weight": "P"}
    merged = merge_shared_and_private(shared, private)
    assert merged != shared and merged != private
    assert merged == {"conv1.weight": "S", "linear.weight": "P"}


def test_merge_raises_when_the_private_head_is_missing():
    """缺私有头就报错，不静默回退到共享头 —— 那样就不是 FedRep 了。"""
    from diag.pfl_fedrep import merge_shared_and_private
    try:
        merge_shared_and_private({"linear.weight": "S"}, {})
    except KeyError:
        return
    raise AssertionError("private 缺 linear.weight 时应当 KeyError")


def test_key_set_follows_shared_so_load_state_dict_can_be_strict():
    from diag.pfl_fedrep import merge_shared_and_private
    shared = {"conv1.weight": 1, "linear.weight": 2}
    private = {"conv1.weight": 9, "linear.weight": 8, "extra.buf": 7}
    assert set(merge_shared_and_private(shared, private)) == set(shared)


# ── track.py 侧的接线（源码断言：本机没有 torch，跑不了 _evaluate_now）────────
def _track_src() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "track.py").read_text(encoding="utf-8")


def test_the_three_shared_columns_are_registered():
    """列没登记 = 落盘时被丢掉，而 run 一切正常 —— 事后才发现白跑。"""
    src = _track_src()
    for col in ("mta_personalized_shared", "asr_paper_shared_benign",
                "asr_paper_filtered_shared_benign"):
        assert src.count(f'"{col}"') >= 2, \
            f"{col} 要既登记进列表、又出现在落行处（现出现 {src.count(chr(34)+col+chr(34))} 次）"


def test_the_parallel_measurement_does_not_replace_the_original_column():
    """**铁律 #2**：并排量一遍，不是把 mta_personalized 改成新口径。

    哪种组合才是 FedRep 的个性化模型要由数据判。悄悄换掉口径会让这一批
    与此前所有 exp1 结果不可比，而且从 CSV 上完全看不出来。
    """
    src = _track_src()
    assert '"mta_personalized": _mean("mta")' in src, \
        "原来的 mta_personalized 被改了 —— 那不是并排测量，是换口径"


def test_shared_pm_is_gated_on_the_fedrep_arm():
    """FedBN 臂的全局 BN 停在初始化值（README §4b），拼出来不是可用模型。
    非 FedRep 客户端必须返回 None → 列留 nan，不是 0（铁律 #5）。"""
    src = _track_src()
    assert 'hasattr(client, "_fedrep_head_steps")' in src
    assert "return None" in src.split("def _fedrep_shared_pm")[1].split("def ")[0]


# ── BN 冻结的方式（2026-09-09：第一版用 .eval()，被 .train() 每步撤销）────────
def _fedrep_src() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "pfl_fedrep.py").read_text(encoding="utf-8")


def test_norm_freeze_uses_track_running_stats_not_eval_mode():
    """`client.py:local_update` **每一步**都调 `self.local_model.train()`，
    而 `_fedrep_set_phase` 只在相位切换时调用一次 —— 用 `.eval()` 冻 BN 的话，
    头阶段第一步之后就被撤销了，running stats 全程在更新并被上传。

    `.train()` 不碰 `track_running_stats`，所以设一次就稳；而且它更正确 ——
    eval 模式会改变头阶段 forward 的归一化来源。
    """
    body = _fedrep_src().split("def _fedrep_set_phase")[1].split("\n    def ")[0]
    # **只看代码，不看注释** —— 本文件的注释里就写着 `.eval()` 三个字，
    # 拿原文匹配会被自己的说明文字绊倒（第一版就是）。
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#") and '"""' not in ln)
    assert "track_running_stats" in code
    assert ".eval()" not in code and "m.train(" not in code, \
        "又用回模块 train/eval 了 —— 那会被 client.py 每步的 .train() 撤销"


def test_norm_tracking_is_restored_before_upload_and_eval():
    """`track_running_stats=False` 时 BN 在 eval 模式下也用 batch 统计量，
    评估的就不是部署时那个函数。head_steps == local_steps 时相位永不切换，
    所以恢复不能只靠 _fedrep_set_phase("body")。"""
    src = _fedrep_src()
    assert "def _fedrep_restore_norm" in src
    assert "_fedrep_restore_norm()" in src.split("def upload_model")[1]


def test_the_fake_base_client_replicates_the_real_local_update():
    """**假客户端与真客户端的差异 = 测不到的 bug。**

    第一版的 `_FakeBaseClient.local_update` 漏了 `self.local_model.train()`，
    于是上面那个 BN bug 在本地全绿、在集群上跑满 300 轮才从 MTA 上看出来。
    """
    from pathlib import Path
    tests_src = Path(__file__).read_text(encoding="utf-8")
    fake = tests_src.split("class _FakeBaseClient")[1].split("\nclass ")[0]
    assert "self.local_model.train()" in fake

    real = (Path(__file__).resolve().parent.parent.parent / "client.py").read_text(encoding="utf-8")
    real_body = real.split("    def local_update(self):")[1].split("\n    def ")[0]
    assert "self.local_model.train()" in real_body, \
        "上游 local_update 不再调 .train() 了 —— 本测试的前提变了，去核对 _fedrep_set_phase"


# ── 逐客户端分布上的准确率（mta_* 用的是共享均衡探针，是错的仪器）──────────
def test_per_client_accuracy_columns_are_registered():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "track.py").read_text(encoding="utf-8")
    for col in ("acc_local_personalized", "acc_local_malicious",
                "acc_local_personalized_shared"):
        assert src.count(f'"{col}"') >= 2, f"{col} 要既登记进列表、又出现在落行处"


def test_per_client_accuracy_uses_the_clients_own_loader():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "track.py").read_text(encoding="utf-8")
    body = src.split("def _local_acc")[1].split("\n    def ")[0]
    assert "loader is None" in body and 'float("nan")' in body, \
        "缺 loader 要返回 nan，不是 0（铁律 #5）"
    assert "from utils import evaluate_accuracy" in body, \
        "用上游 main.py 自己那把尺，否则两边的准确率不同口径"


def test_per_client_accuracy_is_a_fraction_not_a_percentage():
    """`utils.evaluate_accuracy` 返回百分数，本 CSV 其它列都是 [0,1]。

    不换算的话这一列会比邻列大 100 倍，而 CSV 上看不出单位 —— 第一版漏了，
    实测印出 59.3056 / 73.2986 才发现。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    body = (root / "track.py").read_text(encoding="utf-8") \
        .split("def _local_acc")[1].split("\n    def ")[0]
    assert "/ 100.0" in body, "evaluate_accuracy 的百分数没有换算成比例"

    upstream = (root.parent / "utils.py").read_text(encoding="utf-8")
    assert "100 * correct / total" in upstream, \
        "上游 evaluate_accuracy 不再返回百分数了 —— 那这里的 /100 要去掉，去核对"
