"""``diag.exp_e`` 的单元测试。

重点在那些**算错了也不会报错**的地方：
- 双指标的分子分母（ASR 排除目标类样本，ErrorRate 用同一个分母）；
- ``real_target`` 的空分母边界（该组所有样本都是目标类）；
- E3 现训生成器不得改动 clean 模型；
- E2/E3/E4 必须拒绝 attack run 的模型（否则是循环论证）。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from diag.config import load_config
from diag.exp_e import (EXP_E_MODES, ExpEProbe, _metrics, build_exp_e_probe,
                        evaluate_mode, train_oracle_generator)

TARGET = 0


class _FakeTest(Dataset):
    """每类 300 张的假测试集。"""

    def __init__(self, per_class: int = 300, num_classes: int = 10):
        self.targets = [c for c in range(num_classes) for _ in range(per_class)]
        generator = torch.Generator().manual_seed(0)
        self.data = torch.rand(len(self.targets), 3, 8, 8, generator=generator)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def _cfg():
    cfg = load_config()
    cfg["exp_e"].update({"probe_n": 500, "probe_seed": 999, "random_repeats": 2,
                         "oracle_steps": 3})
    return cfg


# ---------------------------------------------------------------------------
# 双指标
# ---------------------------------------------------------------------------
def test_metrics_excludes_target_class_from_denominator():
    """ASR 的分母必须排除 y_true == y_t 的样本（陷阱清单第 5 条）。"""
    labels = torch.tensor([0, 0, 1, 2, 3])          # 前两个本来就是目标类
    preds = torch.tensor([0, 0, 0, 0, 3])           # 三个被判为目标类
    out = _metrics(preds, labels, TARGET)

    assert out["n_eval_samples"] == 3, "分母应只含 3 个非目标类样本"
    assert out["asr_numerator"] == 2, "其中 2 个被判为目标类"
    assert abs(out["asr_targeted"] - 2 / 3) < 1e-9
    # 不排除时：5 个里有 4 个被判为目标类
    assert abs(out["asr_targeted_unfiltered"] - 4 / 5) < 1e-9


def test_metrics_error_rate_uses_same_denominator():
    labels = torch.tensor([0, 1, 2, 3])
    preds = torch.tensor([0, 0, 2, 9])   # 非目标类中：1→0 错, 2→2 对, 3→9 错
    out = _metrics(preds, labels, TARGET)
    assert out["n_eval_samples"] == 3
    assert out["error_numerator"] == 2
    assert abs(out["error_rate"] - 2 / 3) < 1e-9


def test_metrics_all_target_class_gives_nan_not_zero():
    """real_target 组全是目标类 -> 分母为空 -> 必须是 nan，不能是 0。"""
    labels = torch.tensor([0, 0, 0])
    preds = torch.tensor([0, 0, 1])
    out = _metrics(preds, labels, TARGET)

    assert math.isnan(out["asr_targeted"]), "空分母必须给 nan"
    assert math.isnan(out["error_rate"])
    assert abs(out["asr_targeted_unfiltered"] - 2 / 3) < 1e-9, (
        "该组有意义的数字是不排除版本，即目标类准确率")


def test_metrics_asr_and_error_rate_are_different_quantities():
    """ξ 的预期签名：ErrorRate 高但 ASR 低。指标必须能区分这两者。"""
    labels = torch.tensor([1, 2, 3, 4])
    preds = torch.tensor([5, 6, 7, 8])   # 全错，但没有一个指向目标类
    out = _metrics(preds, labels, TARGET)
    assert out["error_rate"] == 1.0
    assert out["asr_targeted"] == 0.0


# ---------------------------------------------------------------------------
# 探针集
# ---------------------------------------------------------------------------
def test_probe_is_reproducible_and_right_size():
    cfg = _cfg()
    first = build_exp_e_probe(_FakeTest(), TARGET, cfg)
    second = build_exp_e_probe(_FakeTest(), TARGET, cfg)
    assert first.indices == second.indices
    assert len(first.subset) == int(cfg.exp_e.probe_n)


def test_probe_target_subset_is_all_target_class():
    cfg = _cfg()
    dataset = _FakeTest()
    probe = build_exp_e_probe(dataset, TARGET, cfg)
    labels = np.asarray(dataset.targets)[probe.target_indices]
    assert (labels == TARGET).all()
    assert set(probe.target_indices).issubset(set(probe.indices))


def test_probe_does_not_disturb_global_numpy_rng():
    np.random.seed(7)
    expected = np.random.rand(3).tolist()
    np.random.seed(7)
    build_exp_e_probe(_FakeTest(), TARGET, _cfg())
    assert np.random.rand(3).tolist() == expected


def test_probe_rejects_oversized_n():
    cfg = _cfg()
    cfg["exp_e"]["probe_n"] = 10 ** 6
    try:
        build_exp_e_probe(_FakeTest(), TARGET, cfg)
    except ValueError as exc:
        assert "超过测试集大小" in str(exc)
    else:
        raise AssertionError("probe_n 超过数据集大小时必须报错")


# ---------------------------------------------------------------------------
# evaluate_mode
# ---------------------------------------------------------------------------
def _fixture():
    cfg = _cfg()
    dataset = _FakeTest()
    probe = build_exp_e_probe(dataset, TARGET, cfg)
    model = _TinyNet()
    gen = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.Tanh())
    from diag.perturb import make_delta_fn, make_xi_fn
    return (cfg, probe, model,
            make_delta_fn(gen, eps=float(cfg.perturb.eps_delta)),
            make_xi_fn(model, eps=float(cfg.perturb.eps_xi), seed=1))


def test_evaluate_mode_covers_all_seven_modes():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    for mode in EXP_E_MODES:
        row = evaluate_mode(model, probe, mode, device=torch.device("cpu"),
                            cfg=cfg, delta_fn=delta_fn, xi_fn=xi_fn,
                            batch_size=128)
        assert row["perturb_mode"] == mode
        assert row["n_eval_samples_unfiltered"] > 0
        for key in ("asr_targeted_unfiltered",):
            assert 0.0 <= row[key] <= 1.0


def test_evaluate_mode_records_linf_budget():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    budgets = {}
    for mode in ("delta_only", "xi_only", "full", "random_eps4", "random_eps8"):
        row = evaluate_mode(model, probe, mode, device=torch.device("cpu"),
                            cfg=cfg, delta_fn=delta_fn, xi_fn=xi_fn,
                            batch_size=128)
        budgets[mode] = (row["linf_budget"], row["linf_realized"])
        assert row["linf_realized"] <= row["linf_budget"] + 1e-6, (
            f"{mode} 的实测 L∞ 超过标称预算")
    assert budgets["full"][0] > budgets["delta_only"][0], "full 的预算应为 8/255"
    assert abs(budgets["random_eps8"][0] - 2 * budgets["random_eps4"][0]) < 1e-9


def test_evaluate_mode_real_target_has_nan_asr():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    row = evaluate_mode(model, probe, "real_target", device=torch.device("cpu"),
                        cfg=cfg, batch_size=128)
    assert math.isnan(row["asr_targeted"])
    assert row["n_eval_samples"] == 0
    assert 0.0 <= row["asr_targeted_unfiltered"] <= 1.0


def test_evaluate_mode_random_modes_average_over_repeats():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    row = evaluate_mode(model, probe, "random_eps4", device=torch.device("cpu"),
                        cfg=cfg, batch_size=128, noise_dist="uniform")
    assert row["repeats"] == int(cfg.exp_e.random_repeats) > 1
    assert row["noise_dist"] == "uniform"


def test_evaluate_mode_missing_component_raises():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    for mode, missing in (("full", "delta_fn"), ("delta_only", "delta_fn")):
        try:
            evaluate_mode(model, probe, mode, device=torch.device("cpu"),
                          cfg=cfg, xi_fn=xi_fn, batch_size=128)
        except ValueError as exc:
            assert missing in str(exc)
        else:
            raise AssertionError(f"{mode} 缺 {missing} 时应抛 ValueError")


def test_evaluate_mode_restores_model_training_flag():
    cfg, probe, model, delta_fn, xi_fn = _fixture()
    model.train()
    evaluate_mode(model, probe, "none", device=torch.device("cpu"), cfg=cfg,
                  batch_size=128)
    assert model.training is True, "评估必须用 eval，但要恢复调用前的状态"


# ---------------------------------------------------------------------------
# E3 现训生成器
# ---------------------------------------------------------------------------
def test_oracle_training_does_not_modify_clean_model():
    """陷阱清单第 7 条：E3 绝不能污染 clean 模型。"""
    from diag.hooks import state_dict_hash
    from torch.utils.data import DataLoader, TensorDataset

    from resnet import get_resnet

    cfg = _cfg()
    model = get_resnet(size=10, num_classes=10)
    model.eval()
    before = state_dict_hash(model)

    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(
        torch.rand(16, 3, 32, 32, generator=generator),
        torch.randint(0, 10, (16,), generator=generator)), batch_size=8)

    gen, info = train_oracle_generator(model, loader, TARGET, cfg=cfg,
                                       device=torch.device("cpu"))
    assert state_dict_hash(model) == before
    assert info["clean_model_unchanged"] is True
    assert info["oracle_steps"] == int(cfg.exp_e.oracle_steps)


def test_oracle_training_actually_updates_generator():
    """生成器必须真的被训练了 —— 否则 E3 的天花板毫无意义。"""
    from torch.utils.data import DataLoader, TensorDataset

    from diag.hooks import state_dict_hash
    from resnet import get_resnet

    cfg = _cfg()
    model = get_resnet(size=10, num_classes=10)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(
        torch.rand(16, 3, 32, 32, generator=generator),
        torch.randint(0, 10, (16,), generator=generator)), batch_size=8)

    torch.manual_seed(0)
    gen, _ = train_oracle_generator(model, loader, TARGET, cfg=cfg,
                                    device=torch.device("cpu"))
    torch.manual_seed(0)
    fresh_hash = state_dict_hash(__import__("generator", fromlist=["Autoencoder"])
                                 .Autoencoder())
    assert state_dict_hash(gen) != fresh_hash, "训练后的生成器不应等于随机初始化"


def test_oracle_training_leaves_no_gradients_on_clean_model():
    from torch.utils.data import DataLoader, TensorDataset

    from resnet import get_resnet

    cfg = _cfg()
    model = get_resnet(size=10, num_classes=10)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(
        torch.rand(8, 3, 32, 32, generator=generator),
        torch.randint(0, 10, (8,), generator=generator)), batch_size=8)

    train_oracle_generator(model, loader, TARGET, cfg=cfg,
                           device=torch.device("cpu"))
    assert all(p.grad is None for p in model.parameters()), (
        "现训结束后不应在 clean 模型上留下梯度")
