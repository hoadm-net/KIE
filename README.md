# KIE - Key Information Extraction với LayoutLMv3

Repository nghiên cứu và thử nghiệm **LayoutLMv3** (Microsoft) cho bài toán **Token Classification** trên documents.

## 🎯 Mục tiêu

### 1. Khảo sát Dataset FUNSD
- **FUNSD (Form Understanding in Noisy Scanned Documents)**
- Dataset chuẩn cho document understanding
- 199 scanned forms với annotations (149 train, 50 test)
- Entity types: QUESTION, ANSWER, HEADER, OTHER

### 2. Thử nghiệm LayoutLMv3 trên FUNSD
- Fine-tune LayoutLMv3 model trên FUNSD dataset
- Đánh giá hiệu suất trên các entity types
- Tối ưu hóa pipeline: preprocessing, training, evaluation, inference

### 3. Ứng dụng cho ViFoodLabel Dataset
- **ViFoodLabel**: Dataset tiếng Việt về nhãn mã thực phẩm
- Nhận diện các thành phần trên bao bì:
  - **Thành phần**: Nguyên liệu chính
  - **Phụ gia**: Chất bảo quản, chất tạo màu, ...
  - **Dinh dưỡng**: Calories, protein, carbs, ...
  - **Cảnh báo**: Allergens, hạn sử dụng, ...
- Thích ứng model cho tiếng Việt với dấu và ký tự đặc biệt

## 📁 Cấu trúc Project

```
KIE/
├── preprocess_funsd.py      # Data preprocessing và loading
├── train_layoutlmv3.py       # Training pipeline
├── evaluate_layoutlmv3.py    # Evaluation với metrics
├── inference_layoutlmv3.py   # Inference và visualization
├── funsd_download.py         # Download FUNSD dataset
├── requirements.txt          # Python dependencies
├── LayoutLMv3_README.md      # Tài liệu kỹ thuật chi tiết
└── data/
    └── FUNSD/                # FUNSD dataset
```

## 🚀 Quick Start

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python funsd_download.py
```

### 3. Training
```bash
python train_layoutlmv3.py \
  --data_dir data/FUNSD \
  --output_dir outputs/experiment \
  --num_epochs 20 \
  --batch_size 2
```

### 4. Evaluation
```bash
python evaluate_layoutlmv3.py \
  --model_path outputs/experiment/run_*/best_model \
  --data_dir data/FUNSD \
  --output_dir outputs/evaluation
```

### 5. Inference
```bash
python inference_layoutlmv3.py \
  --model_path outputs/experiment/run_*/best_model \
  --image_path data/FUNSD/testing_data/images/example.png \
  --annotation_path data/FUNSD/testing_data/annotations/example.json \
  --output_dir predictions
```

## 📚 Tài liệu

Xem [LayoutLMv3_README.md](LayoutLMv3_README.md) để biết:
- Kiến trúc LayoutLMv3 chi tiết
- Implementation details (Approach 2 - Dummy Labels)
- Giải pháp cho Dataset tiếng Việt
- Troubleshooting và best practices

## 🇻🇳 ViFoodLabel Dataset

### Challenges với Tiếng Việt
- **Tokenization**: RoBERTa tokenizer không tối ưu cho dấu tiếng Việt
- **Word boundaries**: Cần word segmentation rõ ràng
- **Entity types**: Đa dạng và phức tạp hơn FUNSD

### Solutions
1. **Fine-tune custom tokenizer** trên Vietnamese corpus
2. **Use LayoutXLM** (multilingual pre-trained)
3. **Pre-tokenize** với Vietnamese word segmentation tools

## 🔬 Nghiên cứu

### FUNSD Experiments
- [x] Setup pipeline hoàn chỉnh
- [x] Training với different hyperparameters
- [x] Evaluation metrics và analysis
- [x] Inference pipeline với visualization

### ViFoodLabel (Upcoming)
- [ ] Collect và annotate dataset
- [ ] Tokenizer adaptation cho tiếng Việt
- [ ] Fine-tune trên food label domain

## 📊 Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- CUDA (recommended)

Chi tiết trong `requirements.txt`

## 📖 References

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/)
- [Hugging Face LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)

## 📝 License

MIT License
