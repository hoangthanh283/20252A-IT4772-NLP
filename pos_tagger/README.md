# POS Tagging – Brown Corpus

## Mô tả bài toán

Gán nhãn từ loại (Part-of-Speech Tagging) cho toàn bộ **Brown Corpus** (NLTK)
sử dụng **Universal Tagset** (12 nhãn). Hai bộ tagger được huấn luyện trên cùng
tập dữ liệu và đánh giá trên tập test riêng biệt (80/20 split).

## Dữ liệu

| Thuộc tính      | Giá trị           |
|-----------------|-------------------|
| Tổng câu        | 57,340            |
| Tổng token      | 1,161,192          |
| Số nhãn POS     | 12                |
| Train / Test    | 80% / 20%         |

## Hai bộ POS Tagger

### Tagger 1: UnigramTagger
- Gán nhãn dựa trên **xác suất xuất hiện của từng từ** trong tập huấn luyện.
- Với từ chưa gặp, dùng `DefaultTagger('NOUN')` làm backoff.
- **Không** xét ngữ cảnh xung quanh.

### Tagger 2: BigramTagger
- Gán nhãn dựa trên **cặp (nhãn trước, từ hiện tại)**.
- Backoff chain: `BigramTagger → UnigramTagger → DefaultTagger`.
- **Có** xét ngữ cảnh 1 từ bên trái → kỳ vọng chính xác hơn UnigramTagger.

## Kết quả đánh giá

### Macro Precision / Recall / F1

| Tagger               | Precision | Recall | Macro-F1 |
|----------------------|-----------|--------|----------|
| UnigramTagger        |    0.9285 | 0.8932 |   0.9018 |
| BigramTagger         |    0.9305 | 0.9006 |   0.9083 |

### Per-class F1: UnigramTagger

| Tag    | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| .      |    0.9995 | 1.0000 |   0.9998 |
| ADJ    |    0.9127 | 0.8826 |   0.8974 |
| ADP    |    0.9372 | 0.9095 |   0.9232 |
| ADV    |    0.9253 | 0.8558 |   0.8892 |
| CONJ   |    0.9909 | 0.9984 |   0.9946 |
| DET    |    0.9873 | 0.9840 |   0.9856 |
| NOUN   |    0.9260 | 0.9651 |   0.9452 |
| NUM    |    0.9626 | 0.9341 |   0.9481 |
| PRON   |    0.9991 | 0.9384 |   0.9678 |
| PRT    |    0.6795 | 0.9271 |   0.7843 |
| VERB   |    0.9669 | 0.9304 |   0.9483 |
| X      |    0.8542 | 0.3923 |   0.5377 |

### Per-class F1: BigramTagger

| Tag    | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| .      |    0.9995 | 1.0000 |   0.9998 |
| ADJ    |    0.9324 | 0.8877 |   0.9095 |
| ADP    |    0.9542 | 0.9038 |   0.9283 |
| ADV    |    0.9365 | 0.8788 |   0.9067 |
| CONJ   |    0.9919 | 0.9973 |   0.9946 |
| DET    |    0.9897 | 0.9869 |   0.9883 |
| NOUN   |    0.9370 | 0.9745 |   0.9554 |
| NUM    |    0.9622 | 0.9338 |   0.9478 |
| PRON   |    0.9539 | 0.9827 |   0.9681 |
| PRT    |    0.6970 | 0.8989 |   0.7852 |
| VERB   |    0.9771 | 0.9513 |   0.9640 |
| X      |    0.8350 | 0.4115 |   0.5513 |

## Biểu đồ

| Biểu đồ | File |
|---------|------|
| Phân phối nhãn POS | ![tag_dist](figures/tag_distribution.png) |
| Precision per-class | ![prec](figures/precision_per_class.png) |
| Recall per-class    | ![rec](figures/recall_per_class.png) |
| F1 per-class        | ![f1](figures/f1_per_class.png) |
| Macro metrics so sánh | ![macro](figures/macro_comparison.png) |

## Nhận xét

- **UnigramTagger** đạt Macro-F1 = **0.9018**.
- **BigramTagger**  đạt Macro-F1 = **0.9083**.
- Bộ tagger tốt hơn theo Macro-F1: **BigramTagger**.

- BigramTagger sử dụng ngữ cảnh bigram nên lý thuyết tốt hơn, nhưng cũng
  dễ gặp vấn đề **data sparsity** (nhiều bigram chưa gặp trong train). Backoff
  về UnigramTagger và DefaultTagger giúp giảm thiểu vấn đề này.
- UnigramTagger đơn giản nhưng hiệu quả vì Brown Corpus đủ lớn để ước lượng
  xác suất từng từ khá tốt.
- Nhãn có F1 thấp nhất thường là **X** (từ lạ) và **PRT** (particle) — hai nhóm
  có ít mẫu và phân phối không đều trong corpus.
