"""审计脚本共用的发现、统计与判定工具。

**判定规则集中在这里**（``THRESHOLDS``），每条规则在报告中与实测值并列打印。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["THRESHOLDS", "Verdict", "AuditContext", "discover_runs",
           "meta_client_frame", "group_compare", "md_table", "md_kv",
           "fmt", "safe_auc", "valid_class_count", "load_run_bundle",
           "load_run_meta", "assert_partition_matches"]

# ---------------------------------------------------------------------------
# 判定阈值（唯一来源）
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    # 任务 1：AUC 达到多少算"完美分离"
    "perfect_auc": 0.999,
    # 任务 3：有效 t 列数的中位数低于此值 -> A_observable 判为 invalid
    "min_valid_t_median": 2.0,
    # 任务 3：边距矩阵 nan 占比高于此值 -> A_observable 判为 invalid
    "max_nan_fraction": 0.80,
    # 任务 3：|Spearman(A, 有效类数)| 高于此值 -> 判为存在有效类数混杂
    "confound_rho": 0.50,
    # 任务 3：|AUC(仅用有效类数) - 0.5| 高于此值且方向与 A 一致 -> 强化混杂判定
    "confound_auc_margin": 0.15,
    # 任务 4.1：良性 E_k 的 |均值| 高于此值 -> E_k 定义的前提可疑
    "benign_E_tolerance": 0.05,
    # 任务 4.3 / 通用：AUC 偏离 0.5 多少才算有判别力
    "auc_signal_margin": 0.10,
    # 任务 1：两组范围的分离间隙 > 0 即视为完全不重叠
    "separation_gap": 0.0,
}

RUN_DIR_RE = re.compile(r"^(?P<mode>clean|attack)_a(?P<alpha>[0-9.]+)_s(?P<seed>\d+)$")


# ---------------------------------------------------------------------------
# 判定结果
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    """一条判定：规则 + 实测值 + 结论。

    ``conclusion`` 只允许取 "确认" / "赝品" / "真实现象" / "未能确定" 一类的
    短标签；拿不准时必须是 "未能确定"，不允许含糊其辞。
    """

    item: str
    conclusion: str
    rule: str
    measured: str
    action: str = ""
    detail: str = ""

    def as_row(self) -> Dict[str, str]:
        return {"检查项": self.item, "判定": self.conclusion,
                "规则": self.rule, "实测": self.measured, "后续处理": self.action}


@dataclass
class AuditContext:
    """一次审计运行的全部输入与产出位置。"""

    ckpt_root: Path
    raw_dir: Path
    evidence_dir: Path
    cfg: Any
    device: Any
    seeds: Optional[List[int]] = None
    alphas: Optional[List[float]] = None
    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdicts: List[Verdict] = field(default_factory=list)
    # 跨任务传递的小结论（如 task3 判定 invalid 的 alpha 列表）
    notes: Dict[str, Any] = field(default_factory=dict)

    def save_evidence(self, frame: pd.DataFrame, name: str) -> Path:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path

    def add(self, verdict: Verdict) -> Verdict:
        self.verdicts.append(verdict)
        return verdict

    def run_dir(self, mode: str, alpha: float, seed: int) -> Optional[Path]:
        match = self.runs[(self.runs["mode"] == mode)
                          & (self.runs["alpha"] == alpha)
                          & (self.runs["seed"] == seed)]
        if match.empty:
            return None
        return Path(match.iloc[0]["path"])

    def pairs(self) -> List[Tuple[float, int]]:
        """有 checkpoint 的 (alpha, seed) 组合，按 alpha 降序。"""
        if self.runs.empty:
            return []
        found = self.runs[["alpha", "seed"]].drop_duplicates()
        out = [(float(a), int(s)) for a, s in found.itertuples(index=False)]
        if self.alphas is not None:
            out = [p for p in out if p[0] in self.alphas]
        if self.seeds is not None:
            out = [p for p in out if p[1] in self.seeds]
        return sorted(out, key=lambda p: (-p[0], p[1]))


# ---------------------------------------------------------------------------
# 发现
# ---------------------------------------------------------------------------
def discover_runs(ckpt_root: Path) -> pd.DataFrame:
    """清点 ``ckpt_root`` 下所有形如 ``{mode}_a{alpha}_s{seed}`` 的目录。"""
    ckpt_root = Path(ckpt_root)
    rows: List[Dict[str, Any]] = []
    if not ckpt_root.exists():
        return pd.DataFrame(columns=["mode", "alpha", "seed", "path"])

    for entry in sorted(ckpt_root.iterdir()):
        if not entry.is_dir():
            continue
        match = RUN_DIR_RE.match(entry.name)
        if not match:
            continue
        n_clients = len(list(entry.glob("client_*.pt")))
        rows.append({
            "mode": match.group("mode"),
            "alpha": float(match.group("alpha")),
            "seed": int(match.group("seed")),
            "path": str(entry),
            "has_meta": (entry / "meta.json").exists(),
            "has_global": (entry / "global.pt").exists(),
            "has_generator": (entry / "generator.pt").exists(),
            "n_client_ckpt": n_clients,
        })
    return pd.DataFrame(rows)


def meta_client_frame(meta: Dict[str, Any]) -> pd.DataFrame:
    """把 ``meta.json`` 的 clients 段展开成 DataFrame（不含索引大列表）。"""
    rows = []
    for record in meta["clients"]:
        row = {k: v for k, v in record.items()
               if k not in ("train_indices", "test_indices", "label_histogram",
                            "test_label_histogram")}
        row["label_histogram"] = record.get("label_histogram")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["alpha"] = float(meta["alpha"])
    frame["seed"] = int(meta["seed"])
    frame["mode"] = str(meta["mode"])
    return frame


def valid_class_count(histogram: Sequence[int], min_class_count: int) -> int:
    """标签直方图中样本数达到阈值的类数。"""
    return int(np.sum(np.asarray(histogram, dtype=np.int64) >= int(min_class_count)))


def load_run_meta(run_path: Path) -> Dict[str, Any]:
    from ..hooks import load_meta
    return load_meta(Path(run_path) / "meta.json")


def assert_partition_matches(bundle, meta: Dict[str, Any],
                             n_check: int = 3) -> None:
    """校验重建出的数据集与训练时是**同一份**。

    审计是离线重建数据集的：``meta.json`` 只存了 ``train_indices``，
    数据集本身要靠配置重新构造。如果构造出来的数据集与训练时不一致，
    索引要么越界（会崩，还算好），要么**仍然有效但指向不同的图片** ——
    后者是静默错误，所有类原型、散度、边距都会悄悄算错。

    这里按 ``train_indices`` 取出标签，重算直方图，与 meta 中记录的
    ``label_histogram`` 逐项比对。不一致立刻抛异常，绝不带病继续。

    只抽查前 ``n_check`` 个客户端 —— 数据集要么整体对要么整体错，全查是浪费。
    """
    from ..probe import get_targets

    if bundle.train_dataset is None:
        return
    targets = get_targets(bundle.train_dataset)
    num_classes = int(bundle.num_classes)

    for record in meta["clients"][:n_check]:
        indices = record.get("train_indices")
        expected = record.get("label_histogram")
        if indices is None or expected is None:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and int(indices.max()) >= len(targets):
            raise RuntimeError(
                f"重建的训练集只有 {len(targets)} 个样本，但 client "
                f"{record['client_id']} 的 train_indices 最大到 "
                f"{int(indices.max())}。数据集与训练时不一致 —— "
                f"检查 meta.json 的 synthetic_sizes 是否被正确传入 build_datasets。")
        actual = np.bincount(targets[indices], minlength=num_classes)[:num_classes]
        if actual.tolist() != list(expected):
            raise RuntimeError(
                f"client {record['client_id']} 的标签直方图对不上：\n"
                f"  meta 记录: {list(expected)}\n"
                f"  重建得到: {actual.tolist()}\n"
                f"数据集内容与训练时不同（尺寸相同但内容不同是静默错误）。"
                f"审计中止，不带病继续。")


def load_run_bundle(ctx: "AuditContext", run_path: Path,
                    cache_models: bool = False):
    """为审计加载一个 ``RunBundle``。

    默认 ``cache_models=False`` —— 审计要遍历全部客户端，开着无界缓存会让
    100 个 resnet10 常驻（约 2GB/bundle）。
    """
    import argparse

    from ..exp_common import load_bundle

    # smoke 标志必须跟着 checkpoint 走，不能写死：合成数据生成的 run 需要走
    # smoke 路径（否则会去要 torchvision / CIFAR-10），探针规模也不同。
    meta = load_run_meta(run_path)
    smoke = bool(meta.get("smoke", False))
    args = argparse.Namespace(config=None, ckpt_dir=str(run_path), out=None,
                              device=str(ctx.device), smoke=smoke)
    bundle = load_bundle(args, ctx.cfg, cache_models=cache_models,
                         dataset_sizes=meta.get("synthetic_sizes"))
    assert_partition_matches(bundle, meta)
    return bundle


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def safe_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUC，剔除 nan；任一类为空返回 nan。复用 diag.analysis 的实现。"""
    from ..analysis import auc_score
    return auc_score(scores, labels)


def group_compare(values: Sequence[float], is_malicious: Sequence[bool],
                  name: str = "") -> Dict[str, Any]:
    """良性 vs 恶意的分布对比 + 是否完全不重叠。

    ``separated`` 为 True 表示两组取值范围**完全不相交**（存在正的分离间隙），
    这是判断"系统性差异"而非"分布偏移"的关键证据。
    nan 会被剔除；剔除后任一组为空时返回的统计量为 nan。
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(is_malicious).astype(bool)
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]

    benign, malicious = values[~labels], values[labels]
    out: Dict[str, Any] = {"signal": name,
                           "n_benign": int(len(benign)),
                           "n_malicious": int(len(malicious))}
    for tag, block in (("benign", benign), ("malicious", malicious)):
        out[f"{tag}_mean"] = float(block.mean()) if len(block) else float("nan")
        out[f"{tag}_std"] = float(block.std(ddof=1)) if len(block) > 1 else float("nan")
        out[f"{tag}_min"] = float(block.min()) if len(block) else float("nan")
        out[f"{tag}_max"] = float(block.max()) if len(block) else float("nan")

    if len(benign) and len(malicious):
        # 正的间隙 = 两组范围不相交（无论哪一组更高）
        gap = max(float(malicious.min()) - float(benign.max()),
                  float(benign.min()) - float(malicious.max()))
        out["separation_gap"] = gap
        out["separated"] = bool(gap > THRESHOLDS["separation_gap"])
        out["auc"] = safe_auc(values, labels)
        from ..analysis import mannwhitney
        _, p_value = mannwhitney(malicious, benign)
        out["mannwhitney_p"] = p_value
    else:
        out["separation_gap"] = float("nan")
        out["separated"] = False
        out["auc"] = float("nan")
        out["mannwhitney_p"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Markdown 输出（自己实现，避免引入 tabulate 依赖）
# ---------------------------------------------------------------------------
def fmt(value: Any, digits: int = 4) -> str:
    """统一的数值格式化；nan 显示为 ``n/a``，bool 显示为 是/否。

    字符串中的竖线会被转义 —— 判定规则里常含 ``|rho| > 0.5`` 这类写法，
    不转义会把 Markdown 表格切断，整张表错位。换行同理。
    """
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def md_table(frame: pd.DataFrame, digits: int = 4,
             columns: Optional[Sequence[str]] = None) -> str:
    """DataFrame -> Markdown 表格。空表返回一句说明而不是空字符串。"""
    if frame is None or len(frame) == 0:
        return "_（无数据）_"
    if columns is not None:
        frame = frame[[c for c in columns if c in frame.columns]]
    header = list(frame.columns)
    # 表头同样要转义：列名里可能有 ``||mu - mu_g||^2`` 这种写法。
    lines = ["| " + " | ".join(fmt(c, digits) for c in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    # 按列取值而不是用 iterrows()：后者会把一行里的 int / float / bool
    # 统一成同一个 dtype，导致 seed=0 被显示成 0.0000、bool 变成 1.0。
    for position in range(len(frame)):
        lines.append("| " + " | ".join(
            fmt(frame[c].iloc[position], digits) for c in header) + " |")
    return "\n".join(lines)


def md_kv(mapping: Dict[str, Any], digits: int = 4) -> str:
    """dict -> 两列 Markdown 表格。"""
    lines = ["| 项 | 值 |", "|---|---|"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {fmt(value, digits)} |")
    return "\n".join(lines)
