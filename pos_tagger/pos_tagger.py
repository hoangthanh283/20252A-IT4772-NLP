"""
POS Tagging Exercise
===================
- Tải Brown Corpus từ NLTK
- Gán nhãn từ loại (POS) bằng 2 bộ tagger:
    1. UnigramTagger (train trên Brown corpus train split)
    2. NLTK averaged_perceptron_tagger (pre-trained)
- Đánh giá bằng Precision, Recall, Macro-F1 trên tập test
"""

import nltk
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ─────────────────────────────────────────────
# 1. Tải dữ liệu
# ─────────────────────────────────────────────
def download_resources():
    resources = [
        "brown",
        "universal_tagset",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for rr in resources:
        nltk.download(rr, quiet=True)

download_resources()

from nltk.corpus import brown
from nltk.tag import UnigramTagger, DefaultTagger

# ─────────────────────────────────────────────
# 2. Chuẩn bị dữ liệu
# ─────────────────────────────────────────────
print("=" * 60)
print("POS TAGGING – BROWN CORPUS")
print("=" * 60)

# Dùng universal tagset để nhãn đồng nhất hơn
tagged_sents = brown.tagged_sents(tagset="universal")

# Xáo trộn và chia train/test (80/20)
random.seed(42)
shuffled = list(tagged_sents)
random.shuffle(shuffled)

split = int(0.8 * len(shuffled))
train_sents = shuffled[:split]
test_sents  = shuffled[split:]

print(f"\nTổng câu trong Brown corpus : {len(shuffled):,}")
print(f"  Train : {len(train_sents):,} câu")
print(f"  Test  : {len(test_sents):,} câu")

# Flatten test set để lấy danh sách (word, true_tag)
test_tokens = [(w, t) for sent in test_sents for (w, t) in sent]
test_words   = [w for (w, _) in test_tokens]
test_true    = [t for (_, t) in test_tokens]

# ─────────────────────────────────────────────
# 3. Bộ tagger 1 – UnigramTagger
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("Tagger 1: UnigramTagger (train trên Brown train split)")
print("─" * 60)

# Dự phòng: nếu chưa gặp từ thì gán "NOUN"
default_tagger = DefaultTagger("NOUN")
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)

# Predict
unigram_pred = []
for sent in test_sents:
    words = [w for (w, _) in sent]
    tagged = unigram_tagger.tag(words)
    unigram_pred.extend([t for (_, t) in tagged])

# ─────────────────────────────────────────────
# 4. Bộ tagger 2 – PerceptronTagger (pre-trained)
# ─────────────────────────────────────────────
print("\nTagger 2: NLTK PerceptronTagger (pre-trained)")
print("─" * 60)

from nltk.tag import PerceptronTagger
perceptron_tagger = PerceptronTagger()

# PTB tags → Universal tags mapping
from nltk.tag import map_tag

perceptron_pred = []
for sent in test_sents:
    words = [w for (w, _) in sent]
    tagged = perceptron_tagger.tag(words)
    # map PTB → universal
    mapped = [map_tag("en-ptb", "universal", t) for (_, t) in tagged]
    perceptron_pred.extend(mapped)

# ─────────────────────────────────────────────
# 5. Đánh giá
# ─────────────────────────────────────────────
def evaluate(name: str, y_true: list, y_pred: list):
    """In precision, recall, macro-F1 của một tagger."""
    labels = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )

    print(f"\n{'─'*60}")
    print(f"KẾT QUẢ – {name}")
    print(f"{'─'*60}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"  Macro F1-Score  : {f1:.4f}")

    # Per-class report
    print(f"\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=labels,
                                zero_division=0, digits=4))

    return precision, recall, f1

p1, r1, f1_1 = evaluate("UnigramTagger", test_true, unigram_pred)
p2, r2, f1_2 = evaluate("PerceptronTagger", test_true, perceptron_pred)

# ─────────────────────────────────────────────
# 6. Kết Quả
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG KẾT SO SÁNH")
print("=" * 60)
header = f"{'Tagger':<22} {'Precision':>12} {'Recall':>10} {'Macro-F1':>10}"
print(header)
print("─" * len(header))
print(f"{'UnigramTagger':<22} {p1:>12.4f} {r1:>10.4f} {f1_1:>10.4f}")
print(f"{'PerceptronTagger':<22} {p2:>12.4f} {r2:>10.4f} {f1_2:>10.4f}")
print("=" * 60)

winner = "PerceptronTagger" if f1_2 > f1_1 else "UnigramTagger"
print(f"\nBộ tagger tốt hơn theo Macro-F1: {winner}")
