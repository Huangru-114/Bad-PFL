"""``diag.hooks`` 的单元测试，重点是 ``verify_partition_consistency``。

这个校验器是整个对照实验的守门人。它最危险的失效模式不是误报，
而是**漏报** —— 划分其实错位了却返回 True，导致所有跨模式比较静默失效。
所以这里既测"一致时通过"，也逐字段测"不一致时必须捕获"。
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from diag.hooks import (extract_generator, flatten_state_dict, run_dir_name,
                        save_state_dict, verify_partition_consistency)


def _meta(mode: str, num_clients: int = 3):
    return {
        "mode": mode, "alpha": 0.5, "seed": 0, "num_classes": 4,
        "target_class": 0,
        "clients": [{
            "client_id": cid,
            "partition_idx": cid,
            "is_malicious": mode == "attack" and cid == 2,
            "is_malicious_slot": cid == 2,
            "poison_ratio": 0.2 if (mode == "attack" and cid == 2) else 0.0,
            "n_local_samples": 40,
            "n_local_test_samples": 20,
            "label_histogram": [10, 10, 10, 10],
            "test_label_histogram": [5, 5, 5, 5],
            "n_target_samples_local": 10,
            "n_participations": 2,
            "train_indices": list(range(cid * 40, (cid + 1) * 40)),
            "test_indices": list(range(cid * 20, (cid + 1) * 20)),
        } for cid in range(num_clients)],
    }


def _write_pair(clean, attack):
    tmp = Path(tempfile.mkdtemp())
    clean_path, attack_path = tmp / "clean.json", tmp / "attack.json"
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    attack_path.write_text(json.dumps(attack), encoding="utf-8")
    return clean_path, attack_path


def test_identical_partitions_pass():
    clean, attack = _meta("clean"), _meta("attack")
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert ok, report
    assert "通过" in report


def test_poison_flags_are_allowed_to_differ():
    """``is_malicious`` / ``poison_ratio`` 正是两次 run 应该不同的地方。"""
    clean, attack = _meta("clean"), _meta("attack")
    assert clean["clients"][2]["is_malicious"] != attack["clients"][2]["is_malicious"]
    ok, _ = verify_partition_consistency(*_write_pair(clean, attack))
    assert ok


def test_label_histogram_mismatch_is_caught():
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"][1]["label_histogram"] = [40, 0, 0, 0]
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok
    assert "label_histogram" in report


def test_sample_count_mismatch_is_caught():
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"][0]["n_local_samples"] = 39
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok and "n_local_samples" in report


def test_partition_idx_mismatch_is_caught():
    """cid <-> 分片 的映射错位是最隐蔽的一种错误，必须被捕获。"""
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"][0]["partition_idx"] = 2
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok and "partition_idx" in report


def test_index_level_mismatch_is_caught_even_when_histogram_matches():
    """直方图相同但实际样本不同 —— 只比直方图会漏报，所以要比索引。"""
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"][1]["train_indices"] = [i + 1000 for i in
                                             clean["clients"][1]["train_indices"]]
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok
    assert "train_indices" in report
    assert "对称差" in report, "索引不一致应给出摘要而不是刷屏"


def test_global_field_mismatch_is_caught():
    clean, attack = _meta("clean"), _meta("attack")
    attack["alpha"] = 0.1
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok and "alpha" in report


def test_missing_client_is_caught():
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"] = attack["clients"][:2]
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert not ok and "仅出现在 clean run" in report


def test_legacy_meta_without_indices_still_compares():
    """旧格式 meta 没有索引字段时跳过该字段，而不是判为不一致。"""
    clean, attack = _meta("clean"), _meta("attack")
    for record in clean["clients"] + attack["clients"]:
        record.pop("train_indices")
        record.pop("test_indices")
    ok, _ = verify_partition_consistency(*_write_pair(clean, attack))
    assert ok


def test_participation_difference_is_a_note_not_a_failure():
    clean, attack = _meta("clean"), _meta("attack")
    attack["clients"][0]["n_participations"] = 99
    ok, report = verify_partition_consistency(*_write_pair(clean, attack))
    assert ok, "参与轮次差异只应提示，不判为失败"
    assert "提示" in report and "随机流可能未隔离" in report


def test_run_dir_name_format():
    assert run_dir_name("attack", 0.05, 2) == "attack_a0.05_s2"


def test_save_state_dict_is_detached_cpu_copy():
    """必须深拷贝：``server.agg_avg`` 会原地修改 state_dict。"""
    model = nn.Linear(4, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.pt"
        save_state_dict(model, path)
        saved = torch.load(path, map_location="cpu")
        # 保存后原地污染模型权重，落盘的副本不应受影响
        with torch.no_grad():
            model.weight.add_(100.0)
        assert not torch.allclose(saved["weight"], model.weight)


def test_flatten_state_dict_skips_integer_buffers():
    model = nn.BatchNorm1d(4)
    state = model.state_dict()
    assert "num_batches_tracked" in state
    flat = flatten_state_dict(state)
    # weight(4) + bias(4) + running_mean(4) + running_var(4) = 16，整型 buffer 被跳过
    assert flat.shape == (16,)


def test_flatten_state_dict_difference_mode():
    model = nn.Linear(3, 2)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shifted = {k: v + 1.0 for k, v in state.items()}
    diff = flatten_state_dict(shifted, reference=state)
    assert torch.allclose(diff, torch.ones_like(diff))


def test_extract_generator_reports_actual_freevars_on_failure():
    """闭包变量改名时必须给出可定位的错误，而不是静默回退。"""
    def make_bad():
        something_else = nn.Linear(2, 2)

        def inner():
            return something_else
        return inner

    try:
        extract_generator(make_bad())
    except RuntimeError as exc:
        assert "something_else" in str(exc)
    else:
        raise AssertionError("找不到 trigger_gen 时必须抛 RuntimeError")


def test_extract_generator_finds_real_generator():
    """对照 fba.use_our_attack 的闭包结构。"""
    def make_good():
        trigger_gen = nn.Conv2d(3, 3, 1)

        def our_poison_func():
            return trigger_gen
        return our_poison_func

    assert isinstance(extract_generator(make_good()), nn.Module)
