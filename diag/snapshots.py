"""实验 F 的按轮次快照埋点。

# 为什么需要新的埋点

实验 F 要构造一个 (生成器轮次 s, 模型轮次 t) 的矩阵，因此必须同时拥有**按轮次**
的生成器快照与**按轮次**的模型快照。

- 生成器侧已有：``hooks.attach_generator_checkpoint_hook``（每 N 轮存一次）。
- 模型侧**从来没有实现过**：``hooks.save_run`` 只在 ``run_fl`` 末尾调用一次，
  官方仓库更是连一次 ``torch.save`` 都没有。本模块补上这一半。

# 非侵入

``fl_process.basic_fl_process`` 已经发射 ``on_client_end``（fl_process.py:32）
与 ``on_round_end``（fl_process.py:38），本模块只注册监听器，
**对 ``fl_process.py`` / ``client.py`` / ``server.py`` 的改动为零**。

# 埋点不得改变训练动力学

保存逻辑只做 I/O：不消耗任何 RNG、不触碰 optimizer、不做前向。因此加了埋点的
重跑应当与不加埋点的 run 逐位一致 —— ``run_fl`` 的 ``--verify-against``
就是用来实证这一点的。

# 客户端快照的"网格对齐参与式"策略

FedBN 下客户端的 ``local_model`` 只在它被选中的那一轮才更新（100 个客户端每轮
选 10 个）。若在轮次 t 无差别地保存所有目标客户端，拿到的多半是它上次参与时的
陈旧权重，而陈旧程度不被记录。

这里改用：每个目标客户端维护一个网格指针，在 ``on_client_end`` 时若
``cur_round >= grid[pointer]`` 就保存到 ``round_{grid值:04d}/`` 并推进指针。
于是每个客户端在每个网格点恰好有一份快照，且

    staleness = 网格轮次 − 实际保存轮次   (<= 0，即实际轮次不早于网格轮次)

被逐份记录进 ``snapshot_manifest.json``，作为分析里的协变量 —— 不隐藏、不平滑。

# ``agg_avg`` 的别名问题

``server.agg_avg``（server.py:4-10）的 ``average_dict = state_dicts[0]`` 是引用；
它对条目做的是**重新绑定**（``average_dict[key] = average_dict[key] + ...``
创建新张量），不是张量级原地写，所以客户端 ``local_model`` 的参数不会被污染 ——
被污染的是 ``client.upload_state_dict`` 这个属性所持的 dict。

本模块一律从 ``module.state_dict()`` 现取，并经 ``hooks.save_state_dict``
深拷贝到 CPU，因此两种情况都安全。**任何读 ``upload_state_dict`` 的代码
必须自己深拷贝。**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from event_emitter import fl_event_emitter

from .hooks import save_state_dict

__all__ = ["build_grid", "select_snapshot_clients", "SnapshotRecorder",
           "round_dir_name", "load_manifest", "available_rounds"]


def round_dir_name(grid_round: int) -> str:
    """``round_0250`` 这样的目录名。四位补零保证字典序 == 数值序。"""
    return f"round_{int(grid_round):04d}"


def build_grid(total_round: int, every: int) -> List[int]:
    """快照网格 ``[every, 2*every, ..., <= total_round]``。

    ``every <= 0`` 返回空列表（表示不做按轮次快照）。若 ``total_round`` 不是
    ``every`` 的整数倍，**不额外补上最后一轮** —— 网格必须等距，否则
    实验 F 的 ``gap = t - s`` 会出现不等距的点，折线图的斜率不可比。
    最终轮的模型/生成器由 ``hooks.save_run`` 单独保存在 run 根目录。
    """
    if int(every) <= 0:
        return []
    return list(range(int(every), int(total_round) + 1, int(every)))


def select_snapshot_clients(clients: Sequence[Any], n_benign: int,
                            n_malicious: int, seed: int) -> List[int]:
    """挑选要逐轮保存本地模型的客户端，返回排序后的 ``client_id`` 列表。

    用独立的 ``np.random.RandomState`` 抽样，**不触碰全局 RNG** —— 否则会改变
    后续训练的随机流，让"埋点不改变训练动力学"这条不再成立。

    恶意客户端也抽几个：H_coadapt 的一个自然推论是恶意客户端自己的模型漂移
    最小（它们持续被生成器对齐），所以它们是有信息量的对照，而不是冗余。

    ``clients`` 少于要求的数量时**取全部并如实返回**，不报错也不补齐 ——
    冒烟配置只有 4 个客户端，报错会让冒烟测试无法覆盖这条路径。
    """
    rng = np.random.RandomState(int(seed))
    benign, malicious = [], []
    for client in clients:
        cid = int(client.cid)
        if bool(getattr(client, "diag_is_malicious", False)):
            malicious.append(cid)
        else:
            benign.append(cid)

    def _take(pool: List[int], count: int) -> List[int]:
        pool = sorted(pool)
        count = min(int(count), len(pool))
        if count <= 0:
            return []
        picked = rng.choice(len(pool), size=count, replace=False)
        return [pool[i] for i in sorted(picked.tolist())]

    return sorted(_take(benign, n_benign) + _take(malicious, n_malicious))


class SnapshotRecorder:
    """把全局模型与选中客户端的本地模型按网格保存到 ``ckpt_dir/round_XXXX/``。

    用法::

        recorder = SnapshotRecorder(ckpt_dir, grid, client_ids)
        recorder.attach()
        try:
            basic_fl_process(...)
        finally:
            recorder.detach()
            recorder.write_manifest()

    ``attach`` / ``detach`` 是幂等的，重复调用不会重复注册监听器 —— 单元测试
    会反复 attach，泄漏的监听器会污染后续测试。
    """

    def __init__(self, ckpt_dir, grid: Sequence[int], client_ids: Sequence[int],
                 generator_getter: Optional[Callable[[], Any]] = None):
        self.ckpt_dir = Path(ckpt_dir)
        self.grid: List[int] = [int(r) for r in grid]
        self.client_ids: List[int] = sorted(int(c) for c in client_ids)
        self.generator_getter = generator_getter
        # 每个客户端的网格指针：下一个还没保存的网格点在 self.grid 中的下标
        self._pointer: Dict[int, int] = {cid: 0 for cid in self.client_ids}
        self.records: List[Dict[str, Any]] = []
        self._attached = False
        self._handlers: List[tuple] = []

    # -- 事件处理 ---------------------------------------------------------
    def _on_client_end(self, **kwargs) -> None:
        if not self.grid:
            return
        clients = kwargs["clients"]
        cur_round = int(kwargs["cur_round"])
        client = clients[kwargs["client_indice"]]
        cid = int(client.cid)
        pointer = self._pointer.get(cid)
        if pointer is None or pointer >= len(self.grid):
            return
        grid_round = self.grid[pointer]
        if cur_round < grid_round:
            return

        # 该客户端可能一次跨过多个网格点（长时间未被选中）。为每个被跨过的
        # 网格点都存一份**同样的**权重，并如实记录 staleness —— 这比留空更好，
        # 因为"客户端在轮次 t 的模型"本来就定义为"它上次参与后的模型"。
        while pointer < len(self.grid) and cur_round >= self.grid[pointer]:
            grid_round = self.grid[pointer]
            path = self.ckpt_dir / round_dir_name(grid_round) / f"client_{cid}.pt"
            save_state_dict(client.local_model, path)
            self.records.append({
                "kind": "client", "client_id": cid,
                "grid_round": int(grid_round), "actual_round": int(cur_round),
                "staleness": int(cur_round - grid_round),
                "is_malicious": bool(getattr(client, "diag_is_malicious", False)),
                "path": str(path.relative_to(self.ckpt_dir)),
            })
            pointer += 1
        self._pointer[cid] = pointer

    def _on_round_end(self, **kwargs) -> None:
        if not self.grid:
            return
        cur_round = int(kwargs["cur_round"])
        if cur_round not in set(self.grid):
            return
        server = kwargs["server"]
        path = self.ckpt_dir / round_dir_name(cur_round) / "global.pt"
        save_state_dict(server.global_model, path)
        self.records.append({
            "kind": "global", "client_id": None,
            "grid_round": int(cur_round), "actual_round": int(cur_round),
            "staleness": 0, "is_malicious": None,
            "path": str(path.relative_to(self.ckpt_dir)),
        })
        if self.generator_getter is not None:
            generator = self.generator_getter()
            if generator is not None:
                gen_path = (self.ckpt_dir / round_dir_name(cur_round)
                            / "generator.pt")
                save_state_dict(generator, gen_path)
                self.records.append({
                    "kind": "generator", "client_id": None,
                    "grid_round": int(cur_round), "actual_round": int(cur_round),
                    "staleness": 0, "is_malicious": None,
                    "path": str(gen_path.relative_to(self.ckpt_dir)),
                })

    # -- 生命周期 ---------------------------------------------------------
    def attach(self) -> "SnapshotRecorder":
        if self._attached:
            return self
        self._handlers = [("on_client_end", self._on_client_end),
                          ("on_round_end", self._on_round_end)]
        for event, handler in self._handlers:
            fl_event_emitter.on(event, handler)
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        for event, handler in self._handlers:
            fl_event_emitter.off(event, handler)
        self._handlers = []
        self._attached = False

    # -- 产出 -------------------------------------------------------------
    def missing(self) -> List[Dict[str, Any]]:
        """网格点 × 客户端里没能保存到的组合。

        出现的原因是该客户端在训练结束前再没被选中过。**如实返回，不补齐**
        —— 实验 F 的矩阵允许缺格，但不允许用插值伪造（陷阱 8）。
        """
        gaps = []
        for cid, pointer in self._pointer.items():
            for grid_round in self.grid[pointer:]:
                gaps.append({"client_id": cid, "grid_round": int(grid_round),
                             "reason": "训练结束前未再被选中"})
        return gaps

    def write_manifest(self) -> Path:
        """写 ``snapshot_manifest.json``。"""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / "snapshot_manifest.json"
        staleness = [record["staleness"] for record in self.records
                     if record["kind"] == "client"]
        payload = {
            "grid": self.grid,
            "snapshot_client_ids": self.client_ids,
            "records": self.records,
            "missing": self.missing(),
            "staleness_summary": {
                "n": len(staleness),
                "max": int(max(staleness)) if staleness else None,
                "mean": (float(sum(staleness) / len(staleness))
                         if staleness else None),
            },
            "note": ("staleness = 实际保存轮次 − 网格轮次 >= 0。FedBN 下客户端"
                     "只在参与的轮次更新本地模型，故快照取该网格点之后的首次"
                     "参与。分析时应把它当协变量看待。"),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path


# ---------------------------------------------------------------------------
# 读取侧
# ---------------------------------------------------------------------------
def load_manifest(ckpt_dir) -> Dict[str, Any]:
    path = Path(ckpt_dir) / "snapshot_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在。该 run 没有按轮次的快照，实验 F 的矩阵无法构造；"
            f"需要用 `--snapshot-every` 重跑一次 attack run。")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def available_rounds(ckpt_dir, kind: str = "global",
                     client_id: Optional[int] = None) -> List[int]:
    """列出磁盘上**实际存在**的快照轮次（不读 manifest，直接看文件）。

    manifest 与磁盘可能不一致（例如任务被抢占、文件被清理），因此矩阵的可用性
    以磁盘为准。
    """
    ckpt_dir = Path(ckpt_dir)
    if kind == "client":
        if client_id is None:
            raise ValueError("kind='client' 时必须给 client_id")
        filename = f"client_{int(client_id)}.pt"
    elif kind in ("global", "generator"):
        filename = f"{kind}.pt"
    else:
        raise ValueError(f"未知 kind '{kind}'；可选 global / client / generator")

    rounds = []
    for directory in sorted(ckpt_dir.glob("round_*")):
        if not (directory / filename).exists():
            continue
        try:
            rounds.append(int(directory.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(rounds)
