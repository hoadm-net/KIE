# Key Information Extraction (KIE)

Repository tìm hiểu và thực hành về bài toán **Key Information Extraction** - trích xuất thông tin quan trọng từ documents.

---

## 📋 Giới thiệu về KIE

**Key Information Extraction (KIE)** là bài toán trích xuất thông tin có cấu trúc từ các documents như hóa đơn, biểu mẫu, hợp đồng, chứng từ, v.v. KIE kết hợp nhiều kỹ thuật từ Computer Vision và Natural Language Processing để hiểu và trích xuất thông tin từ documents.

### Input
- **Document images**: Ảnh scan hoặc ảnh chụp của documents (PDF, PNG, JPG)
- **Text content**: Có thể được trích xuất bằng OCR hoặc từ digital documents
- **Layout information**: Thông tin về vị trí, bounding boxes của text

### Output
- **Structured data**: Thông tin được trích xuất dưới dạng key-value pairs
  - Ví dụ: `{"Invoice Number": "INV-2024-001", "Total": "1,500,000 VND", "Date": "10/02/2026"}`
- **Entity labels**: Phân loại các text entities (question, answer, header, v.v.)
- **Relationships**: Mối quan hệ giữa các entities (question-answer linking)

### Metrics đánh giá

**Entity Recognition:**
- **Precision**: Tỷ lệ entities được dự đoán đúng / tổng số entities dự đoán
- **Recall**: Tỷ lệ entities được dự đoán đúng / tổng số entities ground truth
- **F1-score**: Trung bình điều hòa của Precision và Recall

**Entity Linking:**
- **Precision/Recall/F1**: Đánh giá độ chính xác của việc link các cặp entities (question-answer)

**End-to-End:**
- **Exact Match**: Entity phải match cả label và text chính xác
- **Relaxed Match**: Cho phép partial match về text
- **IoU (Intersection over Union)**: Đánh giá overlap của bounding boxes

---

## 🎯 Bài toán con trong KIE

KIE thường được chia thành các bài toán con:

### 1. **Text Detection**
- Phát hiện vị trí của text trong document
- Output: Bounding boxes của text regions

### 2. **OCR (Optical Character Recognition)**
- Nhận dạng nội dung text từ detected regions
- Output: Text strings

### 3. **Layout Analysis**
- Phân tích cấu trúc không gian của document
- Nhóm các text elements có liên quan

### 4. **Entity Recognition**
- Phân loại các text entities
- Labels: question, answer, header, field name, field value, v.v.

### 5. **Entity Linking**
- Xác định mối quan hệ giữa các entities
- Ví dụ: Link question với answer tương ứng

---

## 🔧 Công nghệ & Approaches

### Traditional Approaches
- Rule-based methods
- Template matching
- Regex patterns
- Heuristic algorithms

### Deep Learning Approaches
- **CNN**: Feature extraction từ document images
- **RNN/LSTM/Transformer**: Sequence modeling cho text
- **Graph Neural Networks**: Modeling spatial relationships
- **Multimodal models**: Kết hợp visual và textual features

### State-of-the-art Models
- **LayoutLM family**: LayoutLM, LayoutLMv2, LayoutLMv3
- **DocFormer**: Multimodal transformer for document understanding
- **FormNet**: Structured form understanding
- **BROS**: BERT Relying On Spatiality

---

## 📚 Tài liệu tham khảo

- [FUNSD Paper](https://arxiv.org/abs/1905.13538) - Form Understanding in Noisy Scanned Documents
- [LayoutLM](https://arxiv.org/abs/1912.13318) - Pre-training of Text and Layout for Document Image Understanding
- [Document AI](https://cloud.google.com/document-ai) - Google Cloud Document AI

---