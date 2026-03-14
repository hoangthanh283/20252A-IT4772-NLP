"""
=============================================================
Bài tập:
  1. Tải Brown Corpus từ NLTK
  2. Gán nhãn POS bằng 2 bộ tagger:
       - Tagger 1: UnigramTagger (dựa trên xác suất từng từ)
       - Tagger 2: BigramTagger  (dựa trên context 2 từ liên tiếp)
  3. Đánh giá: Precision, Recall, Macro-F1 (per-class + macro)
  4. Visualize kết quả so sánh
=============================================================
"""
import random
import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
nltk.download("brown",         quiet=True)
nltk.download("universal_tagset", quiet=True)

from nltk.corpus import brown

tagged_sents = brown.tagged_sents(tagset="universal")
all_tags     = [tag for sent in tagged_sents for (_, tag) in sent]
all_words    = [w   for sent in tagged_sents for (w, _)  in sent]

print(f"  Tổng câu  : {len(tagged_sents):,}")
print(f"  Tổng token: {len(all_words):,}")
print(f"  Số tag    : {len(set(all_tags))}")
print(f"  Ví dụ câu : {tagged_sents[0][:5]} ...\n")

# ─────────────────────────────────────────────────────────────
# Phân tích phân phối nhãn (EDA)
# ─────────────────────────────────────────────────────────────
print("Phân tích phân phối nhãn...")
tag_counts = Counter(all_tags)
print("  Phân phối tag:")
for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / len(all_tags)
    print(f"    {tag:<8} {cnt:>8,}  ({pct:.1f}%)")
print()

# ─────────────────────────────────────────────────────────────
# Vẽ biểu đồ phân phối nhãn
# ─────────────────────────────────────────────────────────────
print("Vẽ biểu đồ phân phối nhãn...")
os.makedirs("figures", exist_ok=True)

tags_sorted  = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])]
counts_sorted = [tag_counts[t] for t in tags_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tags_sorted, counts_sorted,
              color=plt.cm.tab20.colors[:len(tags_sorted)], edgecolor="white")
ax.set_title("Phân phối nhãn POS trong Brown Corpus (Universal tagset)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Nhãn POS", fontsize=11)
ax.set_ylabel("Số lượng token", fontsize=11)
ax.bar_label(bars, fmt="{:,.0f}", fontsize=8, padding=3)
plt.tight_layout()
plt.savefig("figures/tag_distribution.png", dpi=150)
plt.close()
print("  => Lưu: figures/tag_distribution.png\n")

# ─────────────────────────────────────────────────────────────
# Chia tập Train / Test (80 / 20)
# ─────────────────────────────────────────────────────────────
print("Chia tập Train / Test...")
random.seed(42)
shuffled   = list(tagged_sents)
random.shuffle(shuffled)

split       = int(0.8 * len(shuffled))
train_sents = shuffled[:split]
test_sents  = shuffled[split:]

# Flatten test set
test_tokens = [(w, t) for sent in test_sents for (w, t) in sent]
test_words  = [w for (w, _) in test_tokens]
test_true   = [t for (_, t) in test_tokens]

print(f"  Train: {len(train_sents):,} câu")
print(f"  Test : {len(test_sents):,} câu  ({len(test_tokens):,} token)\n")

# ─────────────────────────────────────────────────────────────
# Khởi tạo DefaultTagger (backoff chung)
# ─────────────────────────────────────────────────────────────
print("Khởi tạo DefaultTagger (backoff)...")
from nltk.tag import DefaultTagger
default_tagger = DefaultTagger("NOUN")   # nhãn mặc định khi không biết
print("  DefaultTagger('NOUN') sẵn sàng.\n")

# ─────────────────────────────────────────────────────────────
# Huấn luyện Tagger 1 – UnigramTagger
# ─────────────────────────────────────────────────────────────
print("Huấn luyện Tagger 1 – UnigramTagger...")
from nltk.tag import UnigramTagger
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
print("  => UnigramTagger huấn luyện xong.\n")

# ─────────────────────────────────────────────────────────────
# Huấn luyện Tagger 2 – BigramTagger
# ─────────────────────────────────────────────────────────────
print("Huấn luyện Tagger 2 – BigramTagger...")
from nltk.tag import BigramTagger
# BigramTagger dùng UnigramTagger làm backoff để giảm sparsity
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
print("  => BigramTagger huấn luyện xong (backoff = UnigramTagger → DefaultTagger).\n")

# ─────────────────────────────────────────────────────────────
# Predict bằng Tagger 1 – UnigramTagger
# ─────────────────────────────────────────────────────────────
print("Predict bằng UnigramTagger...")
unigram_pred = []
for sent in test_sents:
    words  = [w for (w, _) in sent]
    tagged = unigram_tagger.tag(words)
    unigram_pred.extend([t for (_, t) in tagged])
print(f"  Đã gán nhãn {len(unigram_pred):,} token.\n")

# ─────────────────────────────────────────────────────────────
# Predict bằng Tagger 2 – BigramTagger
# ─────────────────────────────────────────────────────────────
print("Predict bằng BigramTagger...")
bigram_pred = []
for sent in test_sents:
    words  = [w for (w, _) in sent]
    tagged = bigram_tagger.tag(words)
    bigram_pred.extend([t for (_, t) in tagged])
print(f"  Đã gán nhãn {len(bigram_pred):,} token.\n")

# ─────────────────────────────────────────────────────────────
# Định nghĩa hàm đánh giá
# ─────────────────────────────────────────────────────────────
print("Định nghĩa hàm đánh giá...")

LABELS = sorted(set(test_true))

def evaluate(name: str, y_true: list, y_pred: list):
    """Trả về dict chứa precision/recall/f1 macro + per-class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=LABELS, zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0, digits=4
    )
    return {
        "name": name,
        "macro_p": precision, "macro_r": recall, "macro_f1": f1,
        "p_cls": p_cls, "r_cls": r_cls, "f1_cls": f1_cls,
        "report": report,
    }

print("  => Hàm evaluate() sẵn sàng.\n")

# ─────────────────────────────────────────────────────────────
# Đánh giá Tagger 1 – UnigramTagger
# ─────────────────────────────────────────────────────────────
print("Đánh giá UnigramTagger...")
res1 = evaluate("UnigramTagger", test_true, unigram_pred)
print(f"  Macro Precision : {res1['macro_p']:.4f}")
print(f"  Macro Recall    : {res1['macro_r']:.4f}")
print(f"  Macro F1-Score  : {res1['macro_f1']:.4f}")
print(f"\nPer-class:\n{res1['report']}")

# ─────────────────────────────────────────────────────────────
# Đánh giá Tagger 2 – BigramTagger
# ─────────────────────────────────────────────────────────────
print("Đánh giá BigramTagger...")
res2 = evaluate("BigramTagger", test_true, bigram_pred)
print(f"  Macro Precision : {res2['macro_p']:.4f}")
print(f"  Macro Recall    : {res2['macro_r']:.4f}")
print(f"  Macro F1-Score  : {res2['macro_f1']:.4f}")
print(f"\nPer-class:\n{res2['report']}")

# ─────────────────────────────────────────────────────────────
# Block 15: Bảng tổng hợp kết quả macro
# ─────────────────────────────────────────────────────────────
print("Bảng tổng hợp kết quả macro")
print("=" * 60)
print(f"{'Tagger':<22} {'Precision':>12} {'Recall':>10} {'Macro-F1':>10}")
print("─" * 60)
for r in [res1, res2]:
    print(f"{r['name']:<22} {r['macro_p']:>12.4f} {r['macro_r']:>10.4f} {r['macro_f1']:>10.4f}")
print("=" * 60)
winner = max([res1, res2], key=lambda x: x["macro_f1"])
print(f"\nBộ tagger tốt hơn theo Macro-F1: {winner['name']}\n")

# ─────────────────────────────────────────────────────────────
# Vẽ so sánh Precision per-class
# ─────────────────────────────────────────────────────────────
print("Vẽ so sánh Precision per-class...")

x    = np.arange(len(LABELS))
w    = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, res1["p_cls"], w, label="UnigramTagger",
       color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, res2["p_cls"], w, label="BigramTagger",
       color="#DD8452", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Precision per-class: UnigramTagger vs BigramTagger",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Precision")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/precision_per_class.png", dpi=150)
plt.close()
print("  => Lưu: figures/precision_per_class.png\n")

# ─────────────────────────────────────────────────────────────
# Vẽ so sánh Recall per-class
# ─────────────────────────────────────────────────────────────
print("Vẽ so sánh Recall per-class...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, res1["r_cls"], w, label="UnigramTagger",
       color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, res2["r_cls"], w, label="BigramTagger",
       color="#DD8452", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Recall per-class: UnigramTagger vs BigramTagger",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Recall")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/recall_per_class.png", dpi=150)
plt.close()
print("  => Lưu: figures/recall_per_class.png\n")

# ─────────────────────────────────────────────────────────────
# Vẽ so sánh F1-Score per-class
# ─────────────────────────────────────────────────────────────
print("Vẽ so sánh F1-Score per-class...")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, res1["f1_cls"], w, label="UnigramTagger",
       color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, res2["f1_cls"], w, label="BigramTagger",
       color="#DD8452", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("F1-Score per-class: UnigramTagger vs BigramTagger",
             fontsize=12, fontweight="bold")
ax.set_ylabel("F1-Score")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/f1_per_class.png", dpi=150)
plt.close()
print("  => Lưu: figures/f1_per_class.png\n")

# ─────────────────────────────────────────────────────────────
# Vẽ so sánh Macro metrics (radar / grouped bar)
# ─────────────────────────────────────────────────────────────
print("Vẽ so sánh tổng hợp Macro metrics...")

metrics      = ["Macro Precision", "Macro Recall", "Macro F1"]
uni_vals     = [res1["macro_p"], res1["macro_r"], res1["macro_f1"]]
bi_vals      = [res2["macro_p"], res2["macro_r"], res2["macro_f1"]]
x3           = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x3 - w/2, uni_vals, w, label="UnigramTagger",
            color="#4C72B0", edgecolor="white")
b2 = ax.bar(x3 + w/2, bi_vals,  w, label="BigramTagger",
            color="#DD8452", edgecolor="white")
ax.set_xticks(x3); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.7, 1.05)
ax.set_title("So sánh Macro Precision / Recall / F1",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Score")
ax.bar_label(b1, fmt="%.4f", fontsize=9, padding=3)
ax.bar_label(b2, fmt="%.4f", fontsize=9, padding=3)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/macro_comparison.png", dpi=150)
plt.close()
print("  => Lưu: figures/macro_comparison.png\n")
print("Ghi kết quả phân tích ra README.md...")

def fmt_row(label, p, r, f1):
    return f"| {label:<20} | {p:>9.4f} | {r:>6.4f} | {f1:>8.4f} |"

readme_lines = [
    "# POS Tagging – Brown Corpus",
    "",
    "## Mô tả bài toán",
    "",
    "Gán nhãn từ loại (Part-of-Speech Tagging) cho toàn bộ **Brown Corpus** (NLTK)",
    "sử dụng **Universal Tagset** (12 nhãn). Hai bộ tagger được huấn luyện trên cùng",
    "tập dữ liệu và đánh giá trên tập test riêng biệt (80/20 split).",
    "",
    "## Dữ liệu",
    "",
    f"| Thuộc tính      | Giá trị           |",
    f"|-----------------|-------------------|",
    f"| Tổng câu        | {len(tagged_sents):,}            |",
    f"| Tổng token      | {len(all_words):,}          |",
    f"| Số nhãn POS     | {len(LABELS)}                |",
    f"| Train / Test    | 80% / 20%         |",
    "",
    "## Hai bộ POS Tagger",
    "",
    "### Tagger 1: UnigramTagger",
    "- Gán nhãn dựa trên **xác suất xuất hiện của từng từ** trong tập huấn luyện.",
    "- Với từ chưa gặp, dùng `DefaultTagger('NOUN')` làm backoff.",
    "- **Không** xét ngữ cảnh xung quanh.",
    "",
    "### Tagger 2: BigramTagger",
    "- Gán nhãn dựa trên **cặp (nhãn trước, từ hiện tại)**.",
    "- Backoff chain: `BigramTagger → UnigramTagger → DefaultTagger`.",
    "- **Có** xét ngữ cảnh 1 từ bên trái → kỳ vọng chính xác hơn UnigramTagger.",
    "",
    "## Kết quả đánh giá",
    "",
    "### Macro Precision / Recall / F1",
    "",
    "| Tagger               | Precision | Recall | Macro-F1 |",
    "|----------------------|-----------|--------|----------|",
    fmt_row("UnigramTagger", res1["macro_p"], res1["macro_r"], res1["macro_f1"]),
    fmt_row("BigramTagger",  res2["macro_p"], res2["macro_r"], res2["macro_f1"]),
    "",
    "### Per-class F1: UnigramTagger",
    "",
    "| Tag    | Precision | Recall | F1-Score |",
    "|--------|-----------|--------|----------|",
]
for i, tag in enumerate(LABELS):
    readme_lines.append(
        f"| {tag:<6} | {res1['p_cls'][i]:>9.4f} | {res1['r_cls'][i]:>6.4f} | {res1['f1_cls'][i]:>8.4f} |"
    )

readme_lines += [
    "",
    "### Per-class F1: BigramTagger",
    "",
    "| Tag    | Precision | Recall | F1-Score |",
    "|--------|-----------|--------|----------|",
]
for i, tag in enumerate(LABELS):
    readme_lines.append(
        f"| {tag:<6} | {res2['p_cls'][i]:>9.4f} | {res2['r_cls'][i]:>6.4f} | {res2['f1_cls'][i]:>8.4f} |"
    )

readme_lines += [
    "",
    "## Biểu đồ",
    "",
    "| Biểu đồ | File |",
    "|---------|------|",
    "| Phân phối nhãn POS | ![tag_dist](figures/tag_distribution.png) |",
    "| Precision per-class | ![prec](figures/precision_per_class.png) |",
    "| Recall per-class    | ![rec](figures/recall_per_class.png) |",
    "| F1 per-class        | ![f1](figures/f1_per_class.png) |",
    "| Macro metrics so sánh | ![macro](figures/macro_comparison.png) |",
    "",
    "## Nhận xét",
    "",
    f"- **UnigramTagger** đạt Macro-F1 = **{res1['macro_f1']:.4f}**.",
    f"- **BigramTagger**  đạt Macro-F1 = **{res2['macro_f1']:.4f}**.",
    f"- Bộ tagger tốt hơn theo Macro-F1: **{winner['name']}**.",
    "",
    "- BigramTagger sử dụng ngữ cảnh bigram nên lý thuyết tốt hơn, nhưng cũng",
    "  dễ gặp vấn đề **data sparsity** (nhiều bigram chưa gặp trong train). Backoff",
    "  về UnigramTagger và DefaultTagger giúp giảm thiểu vấn đề này.",
    "- UnigramTagger đơn giản nhưng hiệu quả vì Brown Corpus đủ lớn để ước lượng",
    "  xác suất từng từ khá tốt.",
    "- Nhãn có F1 thấp nhất thường là **X** (từ lạ) và **PRT** (particle) — hai nhóm",
    "  có ít mẫu và phân phối không đều trong corpus.",
    "",
    "## Cách chạy",
    "",
    "```bash",
    "conda activate python3.11",
    "pip install nltk scikit-learn matplotlib numpy",
    "python pos_tagger.py",
    "```",
]

with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))
