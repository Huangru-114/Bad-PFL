"""训练结束时的 checkpoint 与元数据保存，以及对照组一致性校验。

# 非侵入方式

原仓库的 ``fl_process.basic_fl_process`` 已经通过 ``fl_event_emitter`` 发射了
6 个事件（``on_fl_begin`` / ``on_round_begin`` / ``on_client_begin`` /
``on_client_end`` / ``on_round_end`` / ``on_fl_end``），但**没有任何代码注册
监听器**。本模块把保存逻辑挂在 ``on_fl_end`` 上，因此对
``fl_process.py`` / ``client.py`` / ``server.py`` 的改动为**零**。

# 一个必须注意的陷阱

``BasicClient.upload_model``（client.py:28）返回的是 ``state_dict()`` 的
**引用**，而 ``server.agg_avg``（server.py:4-10）会**原地修改**
``state_dicts[0]``。因此保存时必须深拷贝并搬到 CPU，否则存下来的可能是被
聚合过程污染过的张量。
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from event_emitter import fl_event_emitter

__all__ = ["run_dir_name", "save_state_dict", "save_run", "attach_save_hook",
           "build_client_meta", "verify_partition_consistency",
           "extract_generator", "load_client_model", "load_meta",
           "flatten_state_dict"]


# ---------------------------------------------------------------------------
# 路径约定
# ---------------------------------------------------------------------------
def run_dir_name(mode: str, alpha: float, seed: int) -> str:
    """``{mode}_a{alpha}_s{seed}``，例如 ``attack_a0.5_s0``。"""
    return f"{mode}_a{alpha}_s{seed}"


def save_state_dict(module: nn.Module, path: Path) -> None:
    """深拷贝到 CPU 后保存（见模块 docstring 的原地修改陷阱）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
             for k, v in module.state_dict().items()}
    torch.save(state, path)


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------
def _histogram(labels: np.ndarray, num_classes: int) -> List[int]:
    return np.bincount(np.asarray(labels, dtype=np.int64),
                       minlength=num_classes)[:num_classes].astype(int).tolist()


def build_client_meta(clients: Any, train_labels_by_client: Dict[int, np.ndarray],
                      test_labels_by_client: Dict[int, np.ndarray],
                      num_classes: int, target_class: int,
                      train_indices_by_client: Optional[Dict[int, Any]] = None,
                      test_indices_by_client: Optional[Dict[int, Any]] = None
                      ) -> List[Dict[str, Any]]:
    """为每个客户端构造元数据记录。

    ``partition_idx`` 是该客户端拿到的**原始数据分片编号**。它与 ``client_id``
    不同：main.py 先按分片顺序建客户端，再 ``shuffle`` 打乱、然后才赋
    ``cid``（main.py:89-99）。恶意客户端恒定绑定分片 90..99，
    ``shuffle`` 只改变 cid 标签，不改变"哪个分片是恶意的"。

    ``train_indices`` / ``test_indices`` 保存的是相对原始数据集的绝对索引。
    它们有两个用途：(1) ``exp_c`` 需要重建**本地训练 loader** 来算
    ``observable_score`` 的类原型；(2) 让 ``verify_partition_consistency``
    能做索引级的精确比对，而不只是比直方图。
    """
    records = []
    for client in clients:
        partition_idx = int(getattr(client, "partition_idx"))
        train_hist = _histogram(train_labels_by_client[partition_idx], num_classes)
        test_hist = _histogram(test_labels_by_client[partition_idx], num_classes)
        poison_ratio = float(getattr(client, "diag_poison_ratio", 0.0))
        record = {
            "client_id": int(client.cid),
            "partition_idx": partition_idx,
            "is_malicious": bool(getattr(client, "diag_is_malicious", False)),
            "is_malicious_slot": bool(getattr(client, "diag_is_malicious_slot", False)),
            "poison_ratio": poison_ratio,
            "n_local_samples": int(sum(train_hist)),
            "n_local_test_samples": int(sum(test_hist)),
            "label_histogram": train_hist,
            "test_label_histogram": test_hist,
            "n_target_samples_local": int(train_hist[int(target_class)]),
            "n_participations": int(getattr(client, "diag_n_participations", 0)),
        }
        if train_indices_by_client is not None:
            record["train_indices"] = np.asarray(
                train_indices_by_client[partition_idx], dtype=np.int64).tolist()
        if test_indices_by_client is not None:
            record["test_indices"] = np.asarray(
                test_indices_by_client[partition_idx], dtype=np.int64).tolist()
        records.append(record)
    records.sort(key=lambda record: record["client_id"])
    return records


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def save_run(ckpt_dir, server: Any, clients: Any, meta: Dict[str, Any],
             generator: Optional[nn.Module] = None,
             delta_reference_batch: Optional[torch.Tensor] = None) -> Path:
    """保存全局模型、每个客户端的本地模型、生成器与元数据。

    ``delta.pt`` 的语义说明：δ 是**输入条件化**的（``δ = generator(x)``），
    不存在"最终的 δ"这样一个固定张量。因此 ``delta.pt`` 只保存在一个固定
    参考 batch 上算出的 δ，**仅供存档与可视化，不参与任何指标计算**。
    真正参与计算的是 ``generator.pt``。
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_state_dict(server.global_model, ckpt_dir / "global.pt")
    for client in clients:
        save_state_dict(client.local_model, ckpt_dir / f"client_{int(client.cid)}.pt")

    if generator is not None:
        save_state_dict(generator, ckpt_dir / "generator.pt")
        if delta_reference_batch is not None:
            from .perturb import delta_from_generator
            delta = delta_from_generator(generator, delta_reference_batch)
            torch.save({"delta": delta.detach().cpu(),
                        "note": "仅供存档/可视化；δ 是输入条件化的，"
                                "指标计算一律用 generator.pt 现算"},
                       ckpt_dir / "delta.pt")

    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return ckpt_dir


def attach_save_hook(ckpt_dir, meta_builder: Callable[[], Dict[str, Any]],
                     generator_getter: Optional[Callable[[], Optional[nn.Module]]] = None,
                     delta_reference_batch: Optional[torch.Tensor] = None) -> Callable:
    """把 ``save_run`` 挂到 ``on_fl_end`` 事件上（对原文件零改动）。

    返回注册的 handler，便于测试时 ``fl_event_emitter.off`` 注销。
    """

    def handler(**kwargs):
        server = kwargs["server"]
        clients = kwargs["clients"]
        generator = generator_getter() if generator_getter is not None else None
        save_run(ckpt_dir, server, clients, meta_builder(), generator,
                 delta_reference_batch)

    fl_event_emitter.on("on_fl_end", handler)
    return handler


# ---------------------------------------------------------------------------
# 对照组一致性校验
# ---------------------------------------------------------------------------
def load_meta(path) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_partition_consistency(meta_clean_path, meta_attack_path
                                 ) -> Tuple[bool, str]:
    """逐客户端比对两次 run 的数据划分是否完全一致。

    这**不是可选项**。整个诊断的对照基础是"clean run 与 attack run 只有
    '是否投毒'这一个变量不同"。若两次 run 的划分错位，所有跨模式比较都失效，
    而且是**静默**失效 —— 结果看起来完全正常。

    比对内容（逐 ``client_id``）：
    ``partition_idx`` / ``n_local_samples`` / ``label_histogram`` /
    ``n_local_test_samples`` / ``test_label_histogram``，
    以及全局的 ``alpha`` / ``seed`` / ``num_classes`` / ``target_class``。

    **不**比对 ``is_malicious`` 与 ``poison_ratio`` —— 那正是两次 run 应该
    不同的地方。也不比对 ``n_participations``（若客户端采样已隔离随机流，
    它应当一致，故仅在报告中提示差异，不判为失败）。

    Returns
    -------
    ``(ok, report)``：``ok`` 为 True 表示划分一致；``report`` 是人类可读的
    差异明细，无论成功失败都应写进日志。
    """
    clean = load_meta(meta_clean_path)
    attack = load_meta(meta_attack_path)

    problems: List[str] = []
    notes: List[str] = []

    for key in ("alpha", "seed", "num_classes", "target_class"):
        if clean.get(key) != attack.get(key):
            problems.append(
                f"全局字段 '{key}' 不一致: clean={clean.get(key)} "
                f"attack={attack.get(key)}")

    clean_clients = {record["client_id"]: record for record in clean["clients"]}
    attack_clients = {record["client_id"]: record for record in attack["clients"]}

    only_clean = sorted(set(clean_clients) - set(attack_clients))
    only_attack = sorted(set(attack_clients) - set(clean_clients))
    if only_clean:
        problems.append(f"仅出现在 clean run 的 client_id: {only_clean}")
    if only_attack:
        problems.append(f"仅出现在 attack run 的 client_id: {only_attack}")

    compared_fields = ("partition_idx", "n_local_samples", "label_histogram",
                       "n_local_test_samples", "test_label_histogram",
                       "train_indices", "test_indices")
    bulky_fields = ("train_indices", "test_indices")
    for client_id in sorted(set(clean_clients) & set(attack_clients)):
        left, right = clean_clients[client_id], attack_clients[client_id]
        for field in compared_fields:
            if field in bulky_fields and field not in left and field not in right:
                continue    # 旧格式 meta 没有索引字段，跳过而不是判为不一致
            if left.get(field) != right.get(field):
                if field in bulky_fields:
                    # 索引列表可能有上千项，只报摘要，避免刷屏。
                    left_n = len(left.get(field) or [])
                    right_n = len(right.get(field) or [])
                    diff_n = len(set(left.get(field) or []) ^ set(right.get(field) or []))
                    problems.append(
                        f"client {client_id} 的 '{field}' 不一致: "
                        f"clean 有 {left_n} 项, attack 有 {right_n} 项, "
                        f"对称差 {diff_n} 项")
                else:
                    problems.append(
                        f"client {client_id} 的 '{field}' 不一致: "
                        f"clean={left.get(field)} attack={right.get(field)}")
        if left.get("n_participations") != right.get("n_participations"):
            notes.append(
                f"client {client_id} 参与轮次数不同: "
                f"clean={left.get('n_participations')} "
                f"attack={right.get('n_participations')} "
                f"（客户端采样随机流可能未隔离）")

    ok = not problems
    lines = [
        f"划分一致性校验: {'通过' if ok else '未通过'}",
        f"  clean  meta: {meta_clean_path}",
        f"  attack meta: {meta_attack_path}",
        f"  比对客户端数: {len(set(clean_clients) & set(attack_clients))}",
        f"  比对字段: {', '.join(compared_fields)}",
    ]
    if problems:
        lines.append(f"  发现 {len(problems)} 处不一致:")
        lines.extend(f"    - {item}" for item in problems[:50])
        if len(problems) > 50:
            lines.append(f"    ... 另有 {len(problems) - 50} 处未显示")
    if notes:
        lines.append(f"  提示（不判为失败）共 {len(notes)} 条:")
        lines.extend(f"    - {item}" for item in notes[:10])
    return ok, "\n".join(lines)


# ---------------------------------------------------------------------------
# 生成器提取
# ---------------------------------------------------------------------------
def extract_generator(eval_func: Any) -> nn.Module:
    """从 ``fba.use_our_attack`` 的返回值里取出生成器实例。

    ``use_our_attack``（fba.py:25-66）只返回 ``eval_func``，把 ``trigger_gen``
    留在闭包里。这里通过闭包变量读取，是**只读**操作，不改动 fba.py。

    若原仓库将来重命名了这个自由变量，这里会抛出带有实际变量名的异常，
    便于快速定位 —— 刻意不做静默回退。
    """
    func = eval_func.func if isinstance(eval_func, functools.partial) else eval_func
    freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    if "trigger_gen" not in freevars:
        raise RuntimeError(
            f"无法从 eval_func 的闭包中找到 'trigger_gen'；"
            f"实际的自由变量为 {freevars}。fba.use_our_attack 的实现可能已变更。")
    cell = func.__closure__[freevars.index("trigger_gen")]
    generator = cell.cell_contents
    if not isinstance(generator, nn.Module):
        raise RuntimeError(f"闭包中的 'trigger_gen' 不是 nn.Module，而是 {type(generator)}")
    return generator


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_client_model(ckpt_path, model_factory: Callable[[], nn.Module],
                      device) -> nn.Module:
    """加载一个 checkpoint 到新建的模型实例上。

    加载后会补上 ``model.device`` 属性 —— 原仓库的 ``utils.evaluate_accuracy``
    （utils.py:30）依赖它，但它是 main.py 外部注入的，不是 nn.Module 的原生属性。
    """
    model = model_factory()
    state = torch.load(Path(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.device = device
    model.eval()
    return model


def flatten_state_dict(state: Dict[str, torch.Tensor],
                       reference: Optional[Dict[str, torch.Tensor]] = None
                       ) -> torch.Tensor:
    """把 state_dict 展平成一维向量；给定 ``reference`` 时返回两者之差。

    只纳入浮点张量，跳过 ``num_batches_tracked`` 这类整型 buffer。
    键的遍历顺序按名称排序，保证跨客户端可比。
    """
    parts = []
    for key in sorted(state.keys()):
        value = state[key]
        if not torch.is_tensor(value) or not value.dtype.is_floating_point:
            continue
        vector = value.detach().cpu().float().flatten()
        if reference is not None:
            vector = vector - reference[key].detach().cpu().float().flatten()
        parts.append(vector)
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)
