"""``diag.snapshots`` 与 ``diag.exp_f`` 的单元测试。

重点在**算错了也不会报错**的地方：

- 快照的网格对齐与 staleness 记录（错了会让 gap 的语义悄悄偏移）；
- 生成器"冻结"的校验必须真的会失败（否则整个实验的前提是空话）；
- δ / ξ 缓存的按批次切片必须严格对齐（错位不会抛错，只会给出错误的数字）；
- 上三角约束与"不插值"（陷阱 7/8）；
- ``global_bn_borrowed`` 只换 BN，不动共享参数。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from diag.config import load_config
from diag.exp_f import (DELTA_SCALE, FrozenGenerator, _is_bn_key, _round_of,
                        _slicer, assert_delta_scale, compute_delta,
                        generator_snapshot_path, load_snapshot_model,
                        make_frozen_xi_fn)
from diag.hooks import compare_run_checkpoints, save_state_dict
from diag.snapshots import (SnapshotRecorder, available_rounds, build_grid,
                            round_dir_name, select_snapshot_clients)

from generator import Autoencoder
from resnet import get_resnet


class _TinyModel(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.fill_(value)
            self.linear.bias.fill_(value)

    def forward(self, x):
        return self.linear(x)


class _FakeClient:
    def __init__(self, cid: int, malicious: bool = False):
        self.cid = cid
        self.diag_is_malicious = malicious
        self.local_model = _TinyModel(float(cid))


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------
def test_build_grid_is_equidistant_and_drops_partial_tail():
    assert build_grid(1000, 50) == list(range(50, 1001, 50))
    # 1000 不是 300 的整数倍：**不补最后一轮**，否则 gap 会出现不等距的点，
    # F-2 折线图的斜率不再可比。
    assert build_grid(1000, 300) == [300, 600, 900]
    assert build_grid(1000, 0) == []
    assert build_grid(1000, -5) == []


def test_round_dir_name_sorts_lexicographically_like_numbers():
    names = [round_dir_name(r) for r in (5, 50, 500, 1000)]
    assert names == sorted(names)


def test_round_of_handles_both_layouts():
    assert _round_of(Path("run/round_0250/generator.pt")) == 250
    assert _round_of(Path("run/generator/generator_round0250.pt")) == 250


# ---------------------------------------------------------------------------
# 快照客户端选取
# ---------------------------------------------------------------------------
def test_select_snapshot_clients_is_deterministic_and_respects_groups():
    clients = [_FakeClient(i, malicious=i >= 8) for i in range(10)]
    first = select_snapshot_clients(clients, 3, 1, seed=4242)
    second = select_snapshot_clients(clients, 3, 1, seed=4242)
    assert first == second
    assert len(first) == 4
    assert sum(1 for c in first if c >= 8) == 1
    assert sum(1 for c in first if c < 8) == 3


def test_select_snapshot_clients_takes_all_when_pool_is_smaller():
    # 冒烟配置只有 4 个客户端；要求 10 个良性时应取全部而不是报错，
    # 否则冒烟测试覆盖不到这条路径。
    clients = [_FakeClient(i, malicious=i == 3) for i in range(4)]
    picked = select_snapshot_clients(clients, 10, 5, seed=0)
    assert picked == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# SnapshotRecorder
# ---------------------------------------------------------------------------
def _drive(recorder: SnapshotRecorder, clients, schedule):
    """用 (轮次 -> 参与的客户端下标) 的日程模拟 fl_process 的事件流。"""
    from event_emitter import fl_event_emitter

    for cur_round, indices in schedule:
        for indice in indices:
            fl_event_emitter.emit(event_name="on_client_end",
                                  cur_round=cur_round, clients=clients,
                                  client_indice=indice)
        fl_event_emitter.emit(event_name="on_round_end", cur_round=cur_round,
                              clients=clients, server=_FakeServer())


class _FakeServer:
    def __init__(self):
        self.global_model = _TinyModel(9.0)


def test_snapshot_recorder_aligns_to_grid_and_records_staleness():
    clients = [_FakeClient(i) for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        recorder = SnapshotRecorder(tmp, grid=[2, 4], client_ids=[0, 1])
        recorder.attach()
        try:
            # client 0 每轮都参与；client 1 只在第 3、4 轮参与
            _drive(recorder, clients, [(1, [0]), (2, [0]), (3, [0, 1]),
                                       (4, [0, 1])])
        finally:
            recorder.detach()
        recorder.write_manifest()

        manifest = json.loads(
            (Path(tmp) / "snapshot_manifest.json").read_text(encoding="utf-8"))
        by_client = {(r["client_id"], r["grid_round"]): r
                     for r in manifest["records"] if r["kind"] == "client"}
        # client 0 在轮次 2 就参与了 -> staleness 0
        assert by_client[(0, 2)]["staleness"] == 0
        # client 1 到轮次 3 才参与 -> 网格点 2 的快照取自轮次 3，staleness 1
        assert by_client[(1, 2)]["actual_round"] == 3
        assert by_client[(1, 2)]["staleness"] == 1
        assert manifest["missing"] == []
        assert manifest["staleness_summary"]["max"] == 1


def test_snapshot_recorder_catches_up_multiple_grid_points_at_once():
    clients = [_FakeClient(0)]
    with tempfile.TemporaryDirectory() as tmp:
        recorder = SnapshotRecorder(tmp, grid=[1, 2, 3], client_ids=[0])
        recorder.attach()
        try:
            _drive(recorder, clients, [(5, [0])])   # 一次跨过全部三个网格点
        finally:
            recorder.detach()
        rounds = [available_rounds(tmp, "client", client_id=0)]
        assert rounds[0] == [1, 2, 3]
        stale = sorted(r["staleness"] for r in recorder.records
                       if r["kind"] == "client")
        assert stale == [2, 3, 4]


def test_snapshot_recorder_reports_missing_without_filling_them_in():
    clients = [_FakeClient(0), _FakeClient(1)]
    with tempfile.TemporaryDirectory() as tmp:
        recorder = SnapshotRecorder(tmp, grid=[1, 2], client_ids=[0, 1])
        recorder.attach()
        try:
            _drive(recorder, clients, [(1, [0]), (2, [0])])   # client 1 从未参与
        finally:
            recorder.detach()
        gaps = recorder.missing()
        assert {g["grid_round"] for g in gaps} == {1, 2}
        assert all(g["client_id"] == 1 for g in gaps)
        # 缺的就是缺的：不能凭空出现文件
        assert available_rounds(tmp, "client", client_id=1) == []


def test_snapshot_recorder_attach_is_idempotent():
    from event_emitter import fl_event_emitter

    with tempfile.TemporaryDirectory() as tmp:
        recorder = SnapshotRecorder(tmp, grid=[1], client_ids=[0])
        recorder.attach()
        recorder.attach()
        recorder.detach()
        handlers = fl_event_emitter._events["on_round_end"].handlers
        assert recorder._on_round_end not in handlers


# ---------------------------------------------------------------------------
# 生成器冻结
# ---------------------------------------------------------------------------
def _write_generator(path: Path, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    save_state_dict(Autoencoder(), path)
    return path


def test_frozen_generator_detects_modification():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_generator(Path(tmp) / "round_0001" / "generator.pt")
        frozen = FrozenGenerator(path, torch.device("cpu"))
        frozen.verify()                       # 未改动时不应抛
        with torch.no_grad():
            next(frozen.module.parameters()).add_(1e-3)
        try:
            frozen.verify()
        except RuntimeError as exc:
            assert "冻结" in str(exc)
        else:
            raise AssertionError("生成器被改动后 verify() 必须抛异常")


def test_frozen_generator_context_manager_checks_on_exit():
    """``with`` 块正常结束时必须校验；块内已有异常时**不得**覆盖原异常。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_generator(Path(tmp) / "round_0001" / "generator.pt")

        try:
            with FrozenGenerator(path, torch.device("cpu")) as frozen:
                with torch.no_grad():
                    next(frozen.module.parameters()).add_(1e-3)
        except RuntimeError as exc:
            assert "冻结" in str(exc)
        else:
            raise AssertionError("正常退出 with 块时必须校验哈希")

        try:
            with FrozenGenerator(path, torch.device("cpu")) as frozen:
                with torch.no_grad():
                    next(frozen.module.parameters()).add_(1e-3)
                raise ValueError("原始错因")
        except ValueError as exc:
            assert str(exc) == "原始错因"   # 不能被 __exit__ 的校验失败盖掉
        else:
            raise AssertionError("块内异常必须原样抛出")


def test_frozen_generator_rejects_train_mode():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_generator(Path(tmp) / "round_0001" / "generator.pt")
        frozen = FrozenGenerator(path, torch.device("cpu"))
        frozen.module.train()
        try:
            frozen.verify()
        except RuntimeError as exc:
            assert "train" in str(exc)
        else:
            raise AssertionError("train 模式下 verify() 必须抛异常")


def test_frozen_generator_freezes_gradients():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_generator(Path(tmp) / "round_0001" / "generator.pt")
        frozen = FrozenGenerator(path, torch.device("cpu"))
        assert not any(p.requires_grad for p in frozen.module.parameters())


def test_generator_snapshot_path_prefers_new_layout_and_refuses_to_guess():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        _write_generator(run / "generator" / "generator_round0100.pt")
        assert generator_snapshot_path(run, 100).parent.name == "generator"
        _write_generator(run / round_dir_name(100) / "generator.pt")
        assert generator_snapshot_path(run, 100).parent.name == "round_0100"
        try:
            generator_snapshot_path(run, 999)
        except FileNotFoundError as exc:
            assert "插值" in str(exc)     # 陷阱 8：缺快照就是缺，不许补
        else:
            raise AssertionError("缺失的轮次必须抛 FileNotFoundError")


# ---------------------------------------------------------------------------
# δ / ξ 缓存的切片
# ---------------------------------------------------------------------------
def test_slicer_returns_chunks_in_order_and_resets_per_factory():
    cache = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    factory = _slicer(cache)
    take = factory()
    assert torch.equal(take(torch.zeros(2, 2)), cache[0:2])
    assert torch.equal(take(torch.zeros(4, 2)), cache[2:6])
    fresh = factory()     # 新的切片函数必须从头开始，否则偏移会串到下一个模式
    assert torch.equal(fresh(torch.zeros(2, 2)), cache[0:2])


def test_slicer_raises_when_exhausted_or_shape_mismatched():
    cache = torch.zeros(4, 3, 2, 2)
    take = _slicer(cache)()
    take(torch.zeros(4, 3, 2, 2))
    try:
        take(torch.zeros(1, 3, 2, 2))
    except RuntimeError as exc:
        assert "缓存耗尽" in str(exc)
    else:
        raise AssertionError("超出缓存长度必须抛异常，不能静默回绕")

    take2 = _slicer(cache)()
    try:
        take2(torch.zeros(2, 3, 4, 4))
    except RuntimeError as exc:
        assert "形状" in str(exc)
    else:
        raise AssertionError("形状不匹配必须抛异常")


def test_frozen_xi_fn_ignores_model_and_labels():
    cache = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    build = make_frozen_xi_fn(cache)
    xi_fn = build()
    # 传入完全不同的图像与标签，返回的仍是缓存 —— 这正是"ξ 冻结在 θ_s"的含义
    out = xi_fn(torch.full((2, 2), 99.0), torch.tensor([3, 7]))
    assert torch.equal(out, cache[0:2])
    assert torch.equal(xi_fn(torch.zeros(2, 2), torch.tensor([0, 0])), cache[2:4])


# ---------------------------------------------------------------------------
# δ 的定义
# ---------------------------------------------------------------------------
def test_delta_scale_is_hardcoded_and_checked_against_config():
    assert DELTA_SCALE == 4.0 / 255.0        # 对应 fba.py:54 的 `/255.*4.`
    cfg = load_config()
    assert_delta_scale(cfg)                  # 当前配置应当一致
    cfg["perturb"] = {**cfg["perturb"], "eps_delta": 8.0 / 255.0}
    try:
        assert_delta_scale(cfg)
    except ValueError as exc:
        assert "漂移" in str(exc)
    else:
        raise AssertionError("配置漂移必须响亮报错（陷阱 6）")


def test_compute_delta_is_bounded_and_deterministic():
    torch.manual_seed(0)
    generator = Autoencoder().eval()
    images = torch.rand(6, 3, 32, 32)
    loader = [(images[:4], torch.zeros(4)), (images[4:], torch.zeros(2))]
    first = compute_delta(generator, loader, torch.device("cpu"))
    second = compute_delta(generator, loader, torch.device("cpu"))
    assert first.shape == images.shape
    # Tanh 收尾 -> |δ| <= DELTA_SCALE 逐元素成立
    assert float(first.abs().max()) <= DELTA_SCALE + 1e-6
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# BN 与借-BN 全局模型
# ---------------------------------------------------------------------------
def test_is_bn_key_matches_the_repo_rule_on_real_resnet_keys():
    keys = list(get_resnet(size=10).state_dict().keys())
    bn_keys = [k for k in keys if _is_bn_key(k)]
    assert any(k.startswith("bn1.") for k in bn_keys)
    assert any("shortcut.1." in k for k in bn_keys)
    # 卷积与线性层不得被误判 —— 误判会让"借 BN"顺手换掉共享参数
    assert not _is_bn_key("layer1.0.conv1.weight")
    assert not _is_bn_key("linear.weight")


def test_global_bn_borrowed_swaps_only_bn_parameters():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        torch.manual_seed(0)
        global_model = get_resnet(size=10)
        torch.manual_seed(1)
        donor = get_resnet(size=10)
        save_state_dict(global_model, run / round_dir_name(50) / "global.pt")
        save_state_dict(donor, run / round_dir_name(50) / "client_7.pt")

        merged = load_snapshot_model(run, 50, kind="global_bn_borrowed",
                                     client_id=None, num_classes=10,
                                     device=torch.device("cpu"), bn_donor_id=7)
        merged_state = merged.state_dict()
        for key, value in global_model.state_dict().items():
            reference = donor.state_dict()[key] if _is_bn_key(key) else value
            assert torch.equal(merged_state[key].cpu(), reference.cpu()), key


# ---------------------------------------------------------------------------
# 逐位比对
# ---------------------------------------------------------------------------
def test_compare_run_checkpoints_detects_difference():
    with tempfile.TemporaryDirectory() as tmp:
        left, right = Path(tmp) / "a", Path(tmp) / "b"
        torch.manual_seed(0)
        model = get_resnet(size=10)
        save_state_dict(model, left / "global.pt")
        save_state_dict(model, right / "global.pt")
        ok, report = compare_run_checkpoints(left, right)
        assert ok, report

        with torch.no_grad():
            model.linear.weight[0, 0] += 1e-5
        save_state_dict(model, right / "global.pt")
        ok, report = compare_run_checkpoints(left, right)
        assert not ok
        assert "global.pt" in report
