"""`diag/METRICS.md` 里的常数必须与代码一致（回应导师 #1 #2，防陷阱 #14）。

**为什么需要这条**：陷阱 #14 的教训是「`experiments/` 下的文档会比数据旧，
而且骗过了一份报告」。口径说明页是要**直接贴进论文**的东西，它一旦和代码对不上，
错的就是论文里的方法学描述 —— 这类错误审稿人一查就穿，而我们自己看不出来。

所以这里不是「检查文档写了没有」，而是**把文档里的每个数字回头对一遍代码**：
代码改了而文档没改 → 立刻红。

纯 stdlib（正则 + 源码），本地秒级。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOC = ROOT / "diag" / "METRICS.md"
MAIN = ROOT / "main.py"
FBA = ROOT / "fba.py"
GEN = ROOT / "generator.py"

DOC_TEXT = DOC.read_text(encoding="utf-8")


def test_doc_exists_and_covers_both_supervisor_questions():
    assert DOC.exists()
    # 用希腊字母 ρ 而不是 "rho" —— 文档里写的就是 ρ（第一版这条挂在这里）
    for must in ("target", "source class", "trigger", "ρ", "filtered",
                 "unfiltered"):
        assert must.lower() in DOC_TEXT.lower(), f"口径页没有覆盖 {must}"


def test_target_class_matches_the_code_default():
    """文档说 target = 0 (airplane)。代码改了默认值就必须同步改文档。"""
    src = MAIN.read_text(encoding="utf-8")
    m = re.search(r'"--ba_target_label",\s*type=int,\s*default=(\d+)', src)
    assert m, "main.py 里找不到 --ba_target_label 的默认值"
    assert m.group(1) == "0", f"代码默认 target={m.group(1)}，文档写的是 0"
    assert "0 = airplane" in DOC_TEXT


def test_pgd_epsilon_matches_the_code():
    """ξ 的预算：文档写 4/255。"""
    src = FBA.read_text(encoding="utf-8")
    m = re.search(r"def pgd_attack\([^)]*epsilon=([\d.]+)\s*/\s*([\d.]+)", src)
    assert m, "fba.py 里找不到 pgd_attack 的 epsilon"
    assert (m.group(1), m.group(2)) == ("4.", "255."), \
        f"代码 epsilon = {m.group(1)}/{m.group(2)}，文档写的是 4/255"
    assert "4/255" in DOC_TEXT


def test_pgd_is_untargeted_against_the_true_label():
    """文档断言 ξ 是**无目标**的（推离真标签），不是推向 target ——
    ρ=1.0 自毁那段解释完全建立在这一点上。"""
    src = FBA.read_text(encoding="utf-8")
    assert "F.cross_entropy(outputs, labels)" in src, \
        "pgd_attack 的损失不再是对真标签的 CE —— ρ=1.0 的机制解释要重写"
    assert "adv_images + alpha * torch.sign(adv_images.grad)" in src, \
        "不再是梯度上升 —— 那就不是无目标 PGD 了"
    assert "untargeted" in DOC_TEXT.lower() or "无目标" in DOC_TEXT


def test_generator_output_scaling_matches():
    """δ 的预算：Tanh ∈ [-1,1] 再 /255*4 → δ ∈ [-4/255, 4/255]。"""
    src = FBA.read_text(encoding="utf-8")
    assert src.count("trigger_gen(") >= 2
    assert re.search(r"trigger_gen\([^)]*\)\s*/\s*255\.\s*\*\s*4\.", src), \
        "生成器输出的缩放变了，文档里 δ 的预算要跟着改"
    assert "nn.Tanh()" in GEN.read_text(encoding="utf-8"), \
        "生成器末层不再是 Tanh —— δ 的取值范围不再是 [-4/255, 4/255]"


def test_the_two_components_are_added_without_a_final_clip():
    """文档明确写了「δ 加上去之后没有再 clip，所以最坏情况总预算是 8/255」。
    这是要写进论文的一句话，代码一改就得改它。"""
    src = FBA.read_text(encoding="utf-8")
    m = re.search(r"poison_data\s*=\s*poison_mask.*?\(poison_data \+ gen_trigger\)",
                  src, re.S)
    assert m, "两个分量不再是简单相加了"
    tail = src[src.index("(poison_data + gen_trigger)"):]
    stop = tail.index("return")
    assert "clamp" not in tail[:stop], \
        "现在 δ 之后有 clip 了 —— 文档里 8/255 那句要改"
    assert "8/255" in DOC_TEXT


def test_rho_is_a_per_sample_bernoulli():
    """文档强调 ρ 是逐样本伯努利，不是每批固定张数。"""
    src = FBA.read_text(encoding="utf-8")
    # 注意 `torch.rand(label.size(0), device=...)` 里有**嵌套括号**，
    # `[^)]*` 会在内层的 ")" 上提前停住（第一版这条就挂在这里）。
    assert re.search(r"poison_mask\s*=\s*torch\.rand\(.*?\)\s*<=\s*poison_ratio",
                     src), "ρ 的语义变了"
    assert "Bernoulli" in DOC_TEXT


def test_eval_forces_rho_to_one():
    """评估时每张图都带触发器 —— 文档据此说明「分母里每张都投了毒」。"""
    src = FBA.read_text(encoding="utf-8")
    assert re.search(r"eval_func\s*=\s*partial\(our_poison_func[^)]*poison_ratio=1\.",
                     src), "评估时的 poison_ratio 不再是 1.0"


def test_per_client_sample_counts_match():
    """224 = floor(250/32)*32 这个数会被直接写进论文，必须对得上代码。"""
    src = MAIN.read_text(encoding="utf-8")
    m = re.search(r'"--client_batch",\s*type=int,\s*default=(\d+)', src)
    assert m and m.group(1) == "32", f"client_batch 默认变成了 {m and m.group(1)}"
    assert "drop_last=True" in src, "drop_last 关掉了 —— 224 那个数不再成立"
    assert re.search(r"client_test_sample_nums\s*=\s*\[int\(len\(test_dataset\)\s*/\s*"
                     r"args\.client_num\)", src), "每客户端测试样本数的算法变了"
    for number in ("1250", "250", "224"):
        assert number in DOC_TEXT, f"文档里少了 {number}"


def test_train_and_test_are_partitioned_separately():
    """Bad-PFL 侧**不合并** train/test —— 文档据此说「这边没有训练/评估重叠问题」。
    哪天有人照 tf-dpfl 改成合并，这句话就错了。"""
    src = MAIN.read_text(encoding="utf-8")
    assert "client_train_data_indices = client_inner_dirichlet_partition(train_dataset_labels" in src
    assert "client_test_data_indices = client_inner_dirichlet_partition(test_dataset_labels" in src
    assert "官方 train 与 test 从不合并" in DOC_TEXT


def test_benign_eval_scope_matches():
    src = (ROOT / "diag" / "run_fl.py").read_text(encoding="utf-8")
    assert "min(10, len(benign_ids))" in src, "良性端评估口径变了"
    assert "min(10, len(benign_ids))" in DOC_TEXT


def test_doc_states_the_filtered_convention_is_primary():
    """报告正文一律用 filtered —— 这是导师第 2 条问的核心，不能含糊。"""
    assert re.search(r"报告正文一律用", DOC_TEXT)
    assert "(unfiltered − 0.1) / 0.9" in DOC_TEXT


def test_doc_carries_a_ready_to_paste_english_section():
    """导师是英文读者；口径页要能直接贴进报告，不能只有中文。"""
    assert "Threat model." in DOC_TEXT
    assert "ASR convention." in DOC_TEXT
    english = DOC_TEXT[DOC_TEXT.index("Threat model."):]
    cjk = [ch for ch in english if "一" <= ch <= "鿿"]
    assert not cjk, f"英文段里混进了中文：{''.join(cjk[:20])}"


def test_analysis_default_asr_column_matches_the_documented_convention():
    """图默认画的口径必须与文档写的口径一致。

    这一条此前**不成立**：`diag/METRICS.md` 与报告 §2 都写 filtered，
    而 `analysis_exp1` 的 `--asr-column` 默认是 `asr_paper_benign`（unfiltered），
    两者差约 10 个百分点。文档说一套、图画另一套是最难发现的一类错误 ——
    图上没有任何地方会暴露它。
    """
    src = (ROOT / "diag" / "analysis_exp1.py").read_text(encoding="utf-8")
    m = re.search(r'"--asr-column",\s*default="([^"]+)"', src)
    assert m, "找不到 --asr-column 的默认值"
    assert m.group(1).startswith("asr_paper_filtered"), (
        f"默认口径是 {m.group(1)}（unfiltered），而文档写的是 filtered")
    assert "报告正文一律用" in DOC_TEXT


def test_analysis_announces_which_convention_it_used():
    """终端要打出口径，unfiltered 时要给警告 —— 事后判读一批图靠的就是这行。"""
    src = (ROOT / "diag" / "analysis_exp1.py").read_text(encoding="utf-8")
    assert "ASR 口径 =" in src
    assert 'if "filtered" not in ASR_COLUMN:' in src, "unfiltered 时没有警告"
