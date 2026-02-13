# LayoutXLM Pipeline cho FUNSD và Vietnamese Documents

## Giới thiệu

Pipeline này triển khai LayoutXLM (multilingual LayoutLMv2) để xử lý documents đa ngôn ngữ, đặc biệt hỗ trợ tiếng Việt. LayoutXLM được pre-trained trên 53 ngôn ngữ và phù hợp cho việc trích xuất thông tin từ tài liệu có cấu trúc phức tạp.

## Tại sao chọn LayoutXLM?

### LayoutXLM
- **Đa ngôn ngữ**: Pre-trained trên 53 ngôn ngữ bao gồm tiếng Việt
- **Model**: `microsoft/layoutxlm-base` (LayoutLMv2 architecture)
- **Tokenizer**: XLM-RoBERTa (hỗ trợ 100+ ngôn ngữ)
- **Visual Backbone**: Detectron2 ResNeXt-101 FPN
- **Ưu điểm**: Hỗ trợ multilingual out-of-the-box, không cần fine-tune tokenizer

### So với LayoutLMv3
| Đặc điểm | LayoutXLM | LayoutLMv3 |
|----------|-----------|------------|
| Ngôn ngữ | 53 languages | Mainly English |
| Tokenizer | XLM-RoBERTa (~250K vocab) | RoBERTa (~50K vocab) |
| Vietnamese | ✓ Native support | ✗ Cần adaptation |
| Visual Features | Detectron2 | ResNet |

## Yêu cầu hệ thống

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (recommended)
- 16GB+ RAM
- 8GB+ GPU VRAM (for training)

## Cài đặt

### 1. Cài đặt dependencies cơ bản

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Detectron2 (BẮT BUỘC)

LayoutXLM yêu cầu Detectron2 cho visual backbone. Cài đặt theo các bước sau:

**Bước 2.1: Cài build tools**
```bash
pip install ninja setuptools wheel
```

**Bước 2.2: Clone và build Detectron2**
```bash
cd /tmp
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e . --no-build-isolation
```

**Note**: Flag `--no-build-isolation` quan trọng để Detectron2 sử dụng PyTorch đã cài.

**Bước 2.3: Verify installation**
```bash
python -c "import detectron2; print('✓ Detectron2 installed:', detectron2.__version__)"
```

### 3. Download FUNSD dataset

```bash
python funsd_download.py
```

Dataset structure:
```
data/FUNSD/
├── training_data/
│   ├── images/
│   └── annotations/
└── testing_data/
    ├── images/
    └── annotations/
```

## Cấu trúc Pipeline

```
├── preprocess_funsd_layoutxlm.py    # Dataset preprocessing
├── train_layoutxlm.py               # Training pipeline
├── evaluate_layoutxlm.py            # Evaluation với metrics
├── inference_layoutxlm.py           # Inference và visualization
└── LayoutXLM_README.md              # Documentation
```

## Sử dụng

### 1. Training

Train model trên FUNSD dataset:

```bash
python train_layoutxlm.py \
    --data_dir data/FUNSD \
    --num_epochs 20 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --output_dir outputs_layoutxlm
```

**Hyperparameters:**
- `--num_epochs`: Số epochs (khuyến nghị: 10-20)
- `--batch_size`: Batch size (2-4 tùy GPU memory)
- `--learning_rate`: Learning rate (5e-5 mặc định)
- `--max_length`: Max sequence length (512 mặc định)
- `--gradient_accumulation_steps`: Gradient accumulation (1 mặc định)
- `--warmup_ratio`: Warmup ratio (0.1 = 10% steps)

**Output:**
```
outputs_layoutxlm/
└── run_YYYYMMDD_HHMMSS/
    ├── best_model/          # Model với F1 cao nhất
    ├── final_model/         # Model cuối epoch
    ├── training_history.json
    └── config.json
```

### 2. Evaluation

Đánh giá model trên test set:

```bash
python evaluate_layoutxlm.py \
    --data_dir data/FUNSD \
    --model_path outputs_layoutxlm/run_*/best_model \
    --batch_size 4 \
    --plot_cm \
    --output_dir evaluation_results_layoutxlm
```

**Outputs:**
- `evaluation_results.json`: Overall metrics (accuracy, precision, recall, F1)
- `classification_report.txt`: Per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization

### 3. Inference

Chạy inference trên document mới:

```bash
python inference_layoutxlm.py \
    --model_path outputs_layoutxlm/run_*/best_model \
    --image_path data/FUNSD/testing_data/images/82092117.png \
    --annotation_path data/FUNSD/testing_data/annotations/82092117.json \
    --output_dir predictions_layoutxlm
```

**Outputs:**
- `{image_name}_prediction.png`: Visualization với ground truth vs predictions
- `{image_name}_predictions.json`: Predictions chi tiết

## Kiến trúc và Implementation

### Model Architecture

```python
from transformers import LayoutLMv2ForTokenClassification, LayoutXLMProcessor

# Load processor
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")

# Load model
model = LayoutLMv2ForTokenClassification.from_pretrained(
    "microsoft/layoutxlm-base",
    num_labels=9  # BIO tagging: O, B-X, I-X for 4 entity types
)
```

### Preprocessing Pipeline

```python
# Encode document
encoding = processor(
    image,                    # PIL Image
    words,                    # List of words
    boxes=boxes,              # Normalized boxes [0-1000]
    word_labels=labels,       # BIO labels
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)

# encoding contains:
# - input_ids: Token IDs
# - attention_mask: Attention mask
# - bbox: Bounding boxes
# - labels: Aligned labels
# - image: Visual features (từ Detectron2)
```

### Training Loop

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    bbox=bbox,
    image=image,              # Note: 'image' parameter (not 'pixel_values')
    labels=labels
)

loss = outputs.loss
logits = outputs.logits
```

**Critical Notes:**
- LayoutXLM uses `image` parameter (not `pixel_values` như LayoutLMv3)
- Empty words must be filtered để tránh alignment issues
- Tokenizer sử dụng XLM-RoBERTa với subword tokenization

### Inference với Dummy Labels Approach

```python
# Use dummy labels để processor align đúng
dummy_labels = [0] * len(words)

encoding = processor(
    image,
    words,
    boxes=boxes,
    word_labels=dummy_labels,  # KEY: cần dummy labels
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# Extract predictions tại positions where label != -100
label_positions = encoding['labels'].squeeze(0)
predicted_labels = []
for i, label_val in enumerate(label_positions):
    if label_val != -100:  # First subword token
        pred_id = predictions[i].item()
        predicted_labels.append(id2label[pred_id])
```

## Ứng dụng cho ViFoodLabel

### Entity Types cho Food Labels

```python
LABEL2ID = {
    "O": 0,
    "B-THANH_PHAN": 1,   # Thành phần (ingredients)
    "I-THANH_PHAN": 2,
    "B-PHU_GIA": 3,       # Phụ gia (additives)
    "I-PHU_GIA": 4,
    "B-DINH_DUONG": 5,    # Dinh dưỡng (nutrition facts)
    "I-DINH_DUONG": 6,
    "B-CANH_BAO": 7,      # Cảnh báo (warnings)
    "I-CANH_BAO": 8
}
```

### Training Pipeline cho Vietnamese

1. **Prepare ViFoodLabel dataset** theo FUNSD format
2. **Train model**:
   ```bash
   python train_layoutxlm.py \
       --data_dir data/ViFoodLabel \
       --num_epochs 30 \
       --batch_size 2
   ```
3. **Evaluate và deploy**

### Ưu điểm cho tiếng Việt

- ✓ Tokenizer hỗ trợ sẵn tiếng Việt (không cần fine-tune)
- ✓ Pre-trained weights có kiến thức về Vietnamese
- ✓ Xử lý tốt dấu thanh và ký tự đặc biệt
- ✓ Vocabulary coverage cao (~250K tokens)

## Troubleshooting

### Issue 1: Detectron2 installation failed

**Problem**: `ModuleNotFoundError: No module named 'torch'` khi build

**Solution**: Sử dụng `--no-build-isolation` flag
```bash
pip install -e . --no-build-isolation
```

### Issue 2: Empty words causing misalignment

**Problem**: Số predictions khác số words trong inference

**Solution**: Filter empty words trong cả preprocessing và inference:
```python
word_text = word_info["text"]
if not word_text or not word_text.strip():
    continue
```

### Issue 3: 'image' vs 'pixel_values' parameter

**Problem**: AttributeError khi forward pass

**Solution**: LayoutXLM uses `image` parameter:
```python
# ✓ Correct
outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    image=image,  # Use 'image'
    labels=labels
)

# ✗ Wrong
outputs = model(
    pixel_values=pixel_values  # LayoutLMv3 style
)
```

### Issue 4: CUDA out of memory

**Solutions**:
- Giảm batch_size: `--batch_size 1`
- Tăng gradient_accumulation: `--gradient_accumulation_steps 4`
- Giảm max_length: `--max_length 384`

## References

- [LayoutXLM Paper](https://arxiv.org/abs/2104.08836) - Multimodal Pre-training for Multilingual Document Understanding
- [LayoutXLM HuggingFace](https://huggingface.co/microsoft/layoutxlm-base) - Model card và documentation
- [Microsoft UniLM GitHub](https://github.com/microsoft/unilm/tree/master/layoutxlm) - Official implementation
- [XFUN Dataset](https://github.com/doc-analysis/XFUN) - Multilingual form understanding benchmark
- [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/) - Form Understanding in Noisy Scanned Documents
- [Detectron2](https://github.com/facebookresearch/detectron2) - Visual backbone

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install ninja setuptools wheel
cd /tmp && git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install -e . --no-build-isolation

# 2. Download data
python funsd_download.py

# 3. Train model
python train_layoutxlm.py --num_epochs 20

# 4. Evaluate
python evaluate_layoutxlm.py \
    --model_path outputs_layoutxlm/run_*/best_model

# 5. Inference
python inference_layoutxlm.py \
    --model_path outputs_layoutxlm/run_*/best_model \
    --image_path data/FUNSD/testing_data/images/82092117.png \
    --annotation_path data/FUNSD/testing_data/annotations/82092117.json
```

## License

MIT License - See project root for details

## Contact

For issues và questions, please open an issue on GitHub.

