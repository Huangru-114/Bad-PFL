"""静态解析 `diag/*.py` 里每个函数用到的名字是否都能解析到。

# 为什么需要这个

`diag/track.py`、`diag/run_fl.py` 等模块 import torch，**本机跑不起来**，
于是「用了一个不在作用域里的名字」这种错只有在集群上、GPU 排到之后、
跑到第一个评估轮时才会炸成 `NameError`。

2026-09-09 实测发生过一次：`_fedrep_shared_pm` 里用了 `get_resnet`，而它在
`track.py` 是**函数内局部 import**（`_evaluate_now` 里那一句），不在模块作用域。
探针白排了一次队。

这个检查是纯 AST，不 import 被检查的模块，所以在没有 torch 的机器上也能跑。

# 它检查什么

对每个函数：**本函数直接写的** `Name` 读取，减去
（参数 ∪ 本函数内的赋值/import/推导式目标/except 名/lambda 参数 ∪
  外层函数绑定的名字 ∪ 模块级名字 ∪ builtins）。
余下的就是解析不到的名字。

嵌套函数各自检查、并继承外层已绑定的名字（闭包是合法的）；
模块级的 `try: import torch / except ImportError:` 也算模块级名字。
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

DIAG = Path(__file__).resolve().parent.parent


def module_level_names(tree):
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name): out.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name): out.add(n.id)
        elif isinstance(node, (ast.If, ast.Try)):  # TYPE_CHECKING / try: import torch
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names: out.add((a.asname or a.name).split(".")[0])
    return out

def bound_in(fn):
    out = set()
    for a in list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs):
        out.add(a.arg)
    if fn.args.vararg: out.add(fn.args.vararg.arg)
    if fn.args.kwarg: out.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
                out |= bound_in(n)
        elif isinstance(n, ast.Lambda):     # lambda 自带作用域，参数不是自由变量
            for a in (list(n.args.args) + list(n.args.kwonlyargs)
                      + list(n.args.posonlyargs)):
                out.add(a.arg)
            if n.args.vararg: out.add(n.args.vararg.arg)
            if n.args.kwarg: out.add(n.args.kwarg.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal): out |= set(n.names)
    return out

def _own_loads(fn):
    """只看**本函数直接写的**表达式，不下钻进嵌套函数（那些各自检查）。"""
    out, nested = set(), []
    def walk(n, top=False):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not top:
            nested.append(n); return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            walk(c)
    walk(fn, top=True)
    return out, nested


def check(path):
    tree = ast.parse(Path(path).read_text())
    mod = module_level_names(tree) | set(dir(builtins)) | {"__file__", "__name__"}
    bad = []

    def visit(fn, enclosing):
        available = enclosing | bound_in(fn)
        loads, nested = _own_loads(fn)
        free = loads - available - mod
        if free:
            bad.append((fn.name, fn.lineno, sorted(free)))
        for sub in nested:
            visit(sub, available)

    def top(node, enclosing):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(c, enclosing)
            else:
                top(c, enclosing)
    top(tree, set())
    return bad


def _scan(source: str):
    """→ [(函数名, 行号, [解析不到的名字])]"""
    tree = ast.parse(source)
    mod = module_level_names(tree) | set(dir(builtins)) | {"__file__", "__name__"}
    bad = []

    def visit(fn, enclosing):
        available = enclosing | bound_in(fn)
        loads, nested = _own_loads(fn)
        free = loads - available - mod
        if free:
            bad.append((fn.name, fn.lineno, sorted(free)))
        for sub in nested:
            visit(sub, available)

    def top(node, enclosing):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(c, enclosing)
            else:
                top(c, enclosing)

    top(tree, set())
    return bad


def test_every_name_in_diag_resolves():
    problems = []
    for path in sorted(DIAG.glob("*.py")):
        for name, line, free in _scan(path.read_text(encoding="utf-8")):
            problems.append(f"{path.name}:{line} {name}() 用了 {free}，但它不在作用域里")
    assert not problems, ("这些名字在集群上会炸成 NameError（本机 import 不了这些模块，"
                          "所以只能静态查）：\n  " + "\n  ".join(problems))


def test_there_are_files_to_scan():
    """反向自检：扫到零个文件也会「全部通过」。"""
    assert len(list(DIAG.glob("*.py"))) > 10


def test_a_missing_local_import_is_caught():
    """正是 2026-09-09 炸掉探针的那个形状：模块级没有、方法里也没 import。"""
    src = (
        "import ast\n"
        "class T:\n"
        "    def ok(self):\n"
        "        from resnet import get_resnet\n"
        "        return get_resnet(size=10)\n"
        "    def broken(self):\n"
        "        return get_resnet(size=10)\n"
    )
    found = {name: free for name, _, free in _scan(src)}
    assert "broken" in found and found["broken"] == ["get_resnet"]
    assert "ok" not in found, "同名函数里 import 过就不该报"


def test_closures_and_lambdas_are_not_false_positives():
    """闭包捕获外层变量、lambda 参数都是合法的 —— 误报会让这个检查被忽略。"""
    src = (
        "def outer(rows):\n"
        "    total = 0\n"
        "    def inner():\n"
        "        return total + len(rows)\n"
        "    return sorted([inner()], key=lambda d: d)\n"
    )
    assert _scan(src) == []


def test_module_level_try_import_counts_as_in_scope():
    """`try: import torch / except ImportError: torch = None` 是本仓库的常用写法。"""
    src = (
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
        "def f(x):\n"
        "    return torch.as_tensor(x)\n"
    )
    assert _scan(src) == []
