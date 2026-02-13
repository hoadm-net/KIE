# LayoutLMv3 for Document Understanding - Token Classification

## 📋 Tổng quan

Pipeline hoàn chỉnh để fine-tune và sử dụng **LayoutLMv3** (Microsoft) cho bài toán **token classification** trên documents (form understanding, information extraction).

### Kiến trúc Model

**LayoutLMv3** là multimodal transformer kết hợp 3 modalities:
- **Text**: Tokenization với RoBERTa tokenizer (Byte-level BPE)
- **Layout**: Bounding box coordinates (normalized 0-1000)
- **Image**: Patch embeddings từ document image

```
Input: Image + Words + Bounding Boxes
         ↓
LayoutLMv3 Encoder (Text + Layout + Vision)
         ↓
Token Classification Head
         ↓
Output: BIO Tags (B-QUESTION, I-ANSWER, B-HEADER, ...)
```

**Key Features:**
- Pre-trained trên 11M document images
- Unified text-image masking
- Word-patch alignment objective
- State-of-the-art trên FUNSD, CORD, SROIE

---

## 🎯 Bài toán: Token Classification trên Forms

### Task Definition
Gán nhãn cho **mỗi word** trong document với BIO tagging scheme:

| Label | Meaning | Example |
|-------|---------|---------|
| B-QUESTION | Begin-Question | "Name:", "Date:", "Address:" |
| I-QUESTION | Inside-Question | "full", "name" trong "Full Name:" |
| B-ANSWER | Begin-Answer | "John" trong answer field |
| I-ANSWER | Inside-Answer | "Smith" trong "John Smith" |
| B-HEADER | Begin-Header | "Application", "Form" |
| I-HEADER | Inside-Header | "Number" trong "Application Number" |
| B-OTHER | Begin-Other | Metadata, page numbers |
| I-OTHER | Inside-Other | Continuation của OTHER |
| O | Outside | Background, không thuộc entity |

### Ứng dụng
- **Form Understanding**: Trích xuất thông tin từ forms, invoices, receipts
- **Document Parsing**: Phân loại các phần trong document
- **Information Extraction**: Tự động điền database từ scanned documents

---

## 📁 Dataset: FUNSD (Form Understanding in Noisy Scanned Documents)

### Thông tin Dataset
- **Train**: 149 annotated forms
- **Test**: 50 annotated forms
- **Total words**: ~30,000 words với BIO labels
- **Nguồn**: RVL-CDIP dataset (scanned documents)

### Cấu trúc Annotation
```json
{
  "form": [
    {
      "id": 0,
      "label": "question",  // entity-level label
      "box": [x0, y0, x1, y1],
      "words": [
        {
          "text": "Full",
          "box": [x0, y0, x1, y1]
        },
        {
          "text": "Name:",
          "box": [x0, y0, x1, y1]
        }
      ]
    }
  ]
}
```

### Xử lý Đặc biệt
1. **Empty Words Filtering**: Loại bỏ words rỗng hoặc chỉ chứa whitespace
2. **Box Normalization**: Normalize tọa độ về scale 0-1000
3. **BIO Conversion**: Convert entity labels sang token-level BIO tags

---

## 🔧 Pipeline Components

### 1. Preprocessing (`preprocess_funsd.py`)
**FUNSDDataset class** - Load và preprocess data cho training

**Key Functions:**
- `convert_to_bio_labels()`: Convert entity labels → BIO tags
- `normalize_box()`: Normalize bounding boxes to 0-1000 scale
- `__getitem__()`: Return processed sample với processor alignment

**Critical Implementation:**
```python
# Filter empty words (MUST match training & inference!)
if not word_text or not word_text.strip():
    continue

# Use processor with word_labels for automatic alignment
encoding = processor(
    image, 
    words, 
    boxes=boxes, 
    word_labels=word_labels,  # Processor auto-aligns labels to tokens
    padding="max_length",
    truncation=True
)
```

**Processor Behavior:**
- Tokenizes words into subword tokens (BPE)
- **only_label_first_subword=True**: Chỉ first token nhận label
- Remaining subword tokens → label = -100 (ignored in loss)

---

### 2. Training (`train_layoutlmv3.py`)
**LayoutLMv3Trainer class** - Fine-tune model trên FUNSD

**Hyperparameters (recommended):**
```python
learning_rate = 5e-5
num_epochs = 20
batch_size = 2
warmup_ratio = 0.1
gradient_accumulation_steps = 4
```

**Training Process:**
1. Load pre-trained `microsoft/layoutlmv3-base`
2. Initialize token classification head (9 classes)
3. AdamW optimizer với linear warmup schedule
4. Mixed precision training (FP16)
5. Best model checkpoint based on eval F1

**Usage:**
```bash
python train_layoutlmv3.py \
  --data_dir data/FUNSD \
  --output_dir outputs/experiment_name \
  --num_epochs 20 \
  --batch_size 2 \
  --learning_rate 5e-5
```

**Expected Results (20 epochs):**
- F1 Score: ~0.85-0.90
- Accuracy: ~88-92%
- Training time: ~2-3 hours on single GPU

---

### 3. Evaluation (`evaluate_layoutlmv3.py`)
**LayoutLMv3Evaluator class** - Đánh giá model trên test set

**Metrics:**
- **Accuracy**: Token-level accuracy
- **Precision/Recall/F1**: Per-class và macro average
- **Confusion Matrix**: Visualization of class predictions
- **Classification Report**: Detailed per-class metrics

**Usage:**
```bash
python evaluate_layoutlmv3.py \
  --model_path outputs/experiment/best_model \
  --data_dir data/FUNSD \
  --output_dir outputs/evaluation
```

**Outputs:**
- `confusion_matrix.png`: Visual confusion matrix
- `classification_report.txt`: Detailed metrics
- `evaluation_results.json`: Machine-readable results

---

### 4. Inference (`inference_layoutlmv3.py`)
**LayoutLMv3Predictor class** - Predict labels trên documents mới

**Key Implementation - Approach 2 (Dummy Labels):**
```python
# CRITICAL: Use dummy labels để processor align giống training!
dummy_labels = [0] * len(words)

encoding = processor(
    image,
    words,
    boxes=boxes,
    word_labels=dummy_labels,  # Dummy để có alignment info
    padding="max_length",
    truncation=True
)

# Extract predictions from positions where label != -100
predictions = model(**encoding).logits.argmax(-1)
label_positions = encoding['labels']  # Processor đã align!

word_predictions = []
for i, label_val in enumerate(label_positions):
    if label_val != -100:  # First token của một word
        word_predictions.append(id2label[predictions[i]])
```

**Tại sao dùng Approach 2:**
1. **Consistency**: Training và inference dùng CÙNG alignment logic
2. **Automatic Handling**: Processor tự động xử lý subword tokenization
3. **Edge Cases**: Empty words, special chars, Unicode đều handle đúng
4. **Tiếng Việt Ready**: Tokenization consistent cho dấu và ký tự đặc biệt

**Usage:**
```bash
python inference_layoutlmv3.py \
  --model_path outputs/experiment/best_model \
  --image_path data/test_image.png \
  --annotation_path data/test_annotation.json \
  --output_dir predictions
```

**Outputs:**
- `{image_name}_prediction.png`: Visualization với bounding boxes
- `{image_name}_predictions.json`: Machine-readable predictions

---

## 🇻🇳 Tương lai: Dataset Tiếng Việt

### Challenges với Tiếng Việt

#### 1. Tokenization Issues
**Vấn đề:** RoBERTa tokenizer (trained on English) không tối ưu cho tiếng Việt

**Example:**
```python
# Word tiếng Việt
word = "đậu phộng"

# RoBERTa BPE tokenization (BAD)
tokens = ["Ä", "##á", "##º", "##u", " ", "ph", "á", "##»", "##£", "ng"]
# → 10 tokens cho 2 words! Dấu bị split thành bytes
```

**Impacts:**
- Alignment phức tạp hơn
- Nhiều subword tokens → exceed max_length dễ hơn
- Model khó học patterns với rare byte sequences

#### 2. Word Boundary Ambiguity
Tiếng Việt không có spaces giữa syllables trong compound words:
- "đậu phộng" (2 words) vs "đậu_phộng" (1 compound)
- "hướng dương" (2 words) vs "hướng_dương" (sunflower - 1 concept)

---

### Giải pháp đề xuất

#### Option 1: Fine-tune Tokenizer (Khuyến nghị)
**Cách làm:**
1. Thu thập Vietnamese corpus (10M+ sentences)
2. Train custom BPE tokenizer:
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["vietnamese_corpus.txt"],
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
)

# Replace trong LayoutLMv3Processor
processor.tokenizer = LayoutLMv3TokenizerFast(tokenizer_object=tokenizer.tokenizer)
```

**Advantages:**
- Tối ưu cho Vietnamese text patterns
- Giảm số lượng subword tokens
- Dấu và ký tự đặc biệt được handle tốt hơn

**Training time:** ~2-4 hours on CPU

---

#### Option 2: Use LayoutXLM (Khuyến nghị cho production)
**LayoutXLM** = LayoutLMv3 architecture + **XLM-RoBERTa tokenizer**

**Advantages:**
- XLM-RoBERTa trained trên 100 languages (including Vietnamese!)
- Multilingual support out-of-the-box
- Better tokenization cho tiếng Việt

**Cách sử dụng:**
```python
from transformers import LayoutXLMProcessor, LayoutXLMForTokenClassification

processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
model = LayoutXLMForTokenClassification.from_pretrained(
    "microsoft/layoutxlm-base",
    num_labels=9  # Your BIO tags
)

# Everything else giống LayoutLMv3!
```

**Note:** LayoutXLM có architecture giống LayoutLMv3 nhưng:
- Larger model (278M params vs 133M)
- Slower inference (~1.5x)
- Better multilingual performance

---

#### Option 3: Pre-tokenize với Vietnamese Word Segmentation
**Cách làm:**
1. Use `underthesea` hoặc `pyvi` để word segmentation:
```python
from underthesea import word_tokenize

text = "đậu phộng hạt hướng dương"
words = word_tokenize(text)
# → ["đậu_phộng", "hạt", "hướng_dương"]
```

2. Treat compound words as single tokens trong annotation
3. Mỗi word = 1 label trong BIO scheme

**Advantages:**
- Control word boundaries explicitly
- Easier annotation
- More interpretable results

**Disadvantages:**
- Requires pre-processing step
- Word segmentation errors propagate

---

### Data Annotation Guide cho Tiếng Việt

#### Format giống FUNSD
```json
{
  "form": [
    {
      "label": "question",
      "words": [
        {"text": "Họ", "box": [x0, y0, x1, y1]},
        {"text": "và", "box": [x0, y0, x1, y1]},
        {"text": "tên:", "box": [x0, y0, x1, y1]}
      ]
    },
    {
      "label": "answer",
      "words": [
        {"text": "Nguyễn", "box": [x0, y0, x1, y1]},
        {"text": "Văn", "box": [x0, y0, x1, y1]},
        {"text": "A", "box": [x0, y0, x1, y1]}
      ]
    }
  ]
}
```

#### Annotation Best Practices
1. **Consistency**: Quyết định word boundary rules (compound words)
2. **OCR Integration**: Sử dụng VietOCR hoặc similar cho text extraction
3. **Box Accuracy**: Bounding boxes chính xác quan trọng cho layout features
4. **Entity Granularity**: Quyết định entity types phù hợp với domain

#### Sample Entity Types cho Vietnamese Forms
- `B-HO_TEN / I-HO_TEN`: Họ và tên
- `B-NGAY_SINH / I-NGAY_SINH`: Ngày sinh
- `B-DIA_CHI / I-DIA_CHI`: Địa chỉ
- `B-SO_DIEN_THOAI / I-SO_DIEN_THOAI`: Số điện thoại
- `B-SO_CMND / I-SO_CMND`: Số CMND/CCCD
- `B-NOI_CAP / I-NOI_CAP`: Nơi cấp
- `B-NGAY_CAP / I-NGAY_CAP`: Ngày cấp

---

## 📊 Performance Benchmarks

### FUNSD Dataset (English)

| Model | Epochs | F1 Score | Accuracy | Training Time |
|-------|--------|----------|----------|---------------|
| LayoutLMv3 (paper) | - | 90.59% | - | - |
| Our implementation | 1 | 58.6% | 74% | ~10 min |
| Our implementation | 20 | ~85-90% | ~88-92% | ~2-3 hours |

### Per-Class Performance (1 epoch)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| B-QUESTION | 0.80 | 0.83 | 0.81 | 1046 |
| I-QUESTION | 0.82 | 0.59 | 0.68 | 1426 |
| B-ANSWER | 0.75 | 0.86 | 0.80 | 803 |
| I-ANSWER | 0.73 | 0.84 | 0.78 | 2476 |
| **B-HEADER** | 0.64 | 0.29 | **0.40** | 119 ⚠️ |
| I-HEADER | 0.80 | 0.44 | 0.57 | 255 |
| **B-OTHER** | 0.88 | 0.35 | **0.51** | 257 ⚠️ |
| I-OTHER | 0.69 | 0.76 | 0.72 | 1974 |

**Notes:**
- HEADER và OTHER classes có ít data → performance thấp hơn
- Cần augmentation hoặc class weighting để improve
- 20 epochs sẽ improve đáng kể

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repo
git clone <your-repo>
cd KIE

# Install dependencies
pip install -r requirements.txt

# Download FUNSD dataset
python funsd_download.py
```

### 2. Train Model
```bash
# Quick test (1 epoch)
python train_layoutlmv3.py \
  --data_dir data/FUNSD \
  --output_dir outputs/test \
  --num_epochs 1 \
  --batch_size 2

# Full training (20 epochs)
python train_layoutlmv3.py \
  --data_dir data/FUNSD \
  --output_dir outputs/full_training \
  --num_epochs 20 \
  --batch_size 2 \
  --learning_rate 5e-5
```

### 3. Evaluate
```bash
python evaluate_layoutlmv3.py \
  --model_path outputs/full_training/run_*/best_model \
  --data_dir data/FUNSD \
  --output_dir outputs/evaluation
```

### 4. Inference
```bash
python inference_layoutlmv3.py \
  --model_path outputs/full_training/run_*/best_model \
  --image_path data/FUNSD/testing_data/images/example.png \
  --annotation_path data/FUNSD/testing_data/annotations/example.json \
  --output_dir predictions
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Solution:**
```bash
# Giảm batch size
python train_layoutlmv3.py --batch_size 1 --gradient_accumulation_steps 8

# Hoặc reduce max_length
python train_layoutlmv3.py --max_length 256
```

#### 2. Low Accuracy on Inference
**Checklist:**
- ✅ Empty words được filter ở cả training và inference
- ✅ Box normalization consistent (0-1000 scale)
- ✅ Dùng dummy labels trong inference (Approach 2)
- ✅ Eval accuracy có consistent với inference không

#### 3. Class Imbalance
**Solution:**
```python
# Sử dụng class weights trong loss
from torch.nn import CrossEntropyLoss

class_weights = torch.tensor([0.5, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 2.0, 2.0])
loss_fct = CrossEntropyLoss(weight=class_weights)
```

---

## 📚 References

1. **LayoutLMv3 Paper**: [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)
2. **FUNSD Dataset**: [Form Understanding in Noisy Scanned Documents](https://guillaumejaume.github.io/FUNSD/)
3. **Hugging Face Docs**: [LayoutLMv3 Documentation](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
4. **LayoutXLM**: [Multimodal Pre-training for Multilingual Document Understanding](https://arxiv.org/abs/2104.08836)

---

## 📝 License & Citation

```bibtex
@article{huang2022layoutlmv3,
  title={LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  journal={arXiv preprint arXiv:2204.08387},
  year={2022}
}
```

---

## 👥 Contact & Contribution

For questions, issues, or contributions, please open an issue on GitHub.

**Maintained by**: [Your Name]  
**Last Updated**: February 2026
