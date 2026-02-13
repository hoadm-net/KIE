"""
Inference và visualize predictions với LayoutLMv3
"""
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path
import numpy as np

from preprocess_funsd import FUNSDDataset


# Color mapping cho labels
LABEL_COLORS = {
    "O": "#95a5a6",              # Gray
    "B-QUESTION": "#3498db",      # Blue
    "I-QUESTION": "#5dade2",      # Light Blue
    "B-ANSWER": "#2ecc71",        # Green
    "I-ANSWER": "#58d68d",        # Light Green
    "B-HEADER": "#e74c3c",        # Red
    "I-HEADER": "#ec7063",        # Light Red
    "B-OTHER": "#f39c12",         # Orange
    "I-OTHER": "#f8c471"          # Light Orange
}


class LayoutLMv3Predictor:
    """Predictor class cho inference"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        processor: LayoutLMv3Processor = None
    ):
        self.device = device
        
        # Load model
        print(f"🤖 Loading model from {model_path}...")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Load processor
        if processor is None:
            print(f"📦 Loading processor...")
            self.processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base",
                apply_ocr=False
            )
        else:
            self.processor = processor
        
        self.id2label = FUNSDDataset.ID2LABEL
    
    def predict(
        self,
        image: Image.Image,
        words: list,
        boxes: list
    ):
        """
        Predict labels cho document
        
        Args:
            image: PIL Image
            words: List of words
            boxes: List of normalized boxes [x0, y0, x1, y1] on 0-1000 scale
        
        Returns:
            predictions: List of predicted labels for each word
            logits: Model output logits
        """
        # APPROACH 2: Dùng dummy labels để processor tự động align
        # Đảm bảo tokenization và alignment GIỐNG HỆT training
        dummy_labels = [0] * len(words)  # Giá trị không quan trọng
        
        # Encode inputs GIỐNG TRAINING (chỉ khác labels là dummy)
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=dummy_labels,  # KEY: thêm dummy labels
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,)
        
        # Extract predictions dựa vào label positions
        # Label != -100 → first token của một word (processor đã align)
        label_positions = encoding['labels'].squeeze(0).cpu()  # (seq_len,)
        predictions_cpu = predictions.cpu()
        
        predicted_labels = []
        label_indices = []  # Track vị trí các labels
        
        for i, label_val in enumerate(label_positions):
            if label_val != -100:
                # Đây là first token của một word
                label_indices.append(i)
                pred_id = predictions_cpu[i].item()
                label = self.id2label[pred_id]
                predicted_labels.append(label)
        
        # Debug: print first 20 predictions to verify
        print(f"\n🔍 First 20 word predictions:")
        for idx in range(min(20, len(predicted_labels))):
            word_text = words[idx] if idx < len(words) else "N/A"
            pred_label = predicted_labels[idx]
            token_pos = label_indices[idx]
            pred_id = predictions_cpu[token_pos].item()
            # Also show dummy label value from encoding
            dummy_label_val = label_positions[token_pos].item()
            print(f"  {idx:3d}: '{word_text:20s}' → pred={pred_label:15s} (pred_id={pred_id}, pos={token_pos}, dummy={dummy_label_val})")
        
        # Debug: in ra thông tin
        print(f"\n🔍 DEBUG INFO:")
        print(f"  - Số words input: {len(words)}")
        print(f"  - Số positions có label != -100: {len(predicted_labels)}")
        print(f"  - Label positions: {label_indices[:20]}...")  # In 20 đầu
        print(f"  - Max position with label: {max(label_indices) if label_indices else 0}")
        print(f"  - Sequence length: {len(label_positions)}")
        print(f"  - Input_ids shape: {encoding['input_ids'].shape}")
        
        # Kiểm tra xem có words nào không tạo ra token không
        print(f"\n  📝 Kiểm tra empty words:")
        empty_count = 0
        for i, word in enumerate(words[:10]):  # Check 10 words đầu
            tokens = self.processor.tokenizer.tokenize(word)
            if len(tokens) == 0:
                print(f"    - Word {i}: '{word}' → EMPTY (no tokens)")
                empty_count += 1
        if empty_count > 0:
            print(f"  ⚠️  Có {empty_count}/10 words không tạo token!")
        
        # Đảm bảo số lượng predictions = số words
        if len(predicted_labels) != len(words):
            print(f"  ⚠️  WARNING: Số predictions ({len(predicted_labels)}) != số words ({len(words)})")
            print(f"  - Có thể do truncation hoặc empty words")
            print(f"  - Đang padding/truncating để match...")
        
        # Nếu thiếu predictions (truncation), thêm "O"
        # Nếu thừa predictions (không nên xảy ra), cắt bớt
        
        # Đảm bảo số lượng predictions = số words
        # (Nếu truncation xảy ra, một số words cuối có thể bị thiếu)
        while len(predicted_labels) < len(words):
            predicted_labels.append("O")
        predicted_labels = predicted_labels[:len(words)]
        
        return predicted_labels, logits
    
    def predict_from_json(
        self,
        image_path: str,
        annotation_path: str
    ):
        """
        Predict from FUNSD-format JSON annotation
        
        Args:
            image_path: Path to image file
            annotation_path: Path to JSON annotation file
        
        Returns:
            image: PIL Image
            words: List of words
            boxes: List of boxes
            true_labels: List of true labels (if available)
            pred_labels: List of predicted labels
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Load annotation
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract words, boxes, labels
        words = []
        boxes = []
        true_labels = []
        
        from preprocess_funsd import FUNSDDataset
        dataset = FUNSDDataset(data_dir="", split="train")  # Just for helper methods
        
        for entity in data["form"]:
            entity_words = entity["words"]
            entity_label = entity["label"]
            
            # Get BIO labels
            bio_labels = dataset.convert_to_bio_labels(entity_words, entity_label)
            
            for word_info, label in zip(entity_words, bio_labels):
                # Skip empty or whitespace-only words (MUST match training!)
                word_text = word_info["text"]
                if not word_text or not word_text.strip():
                    continue
                    
                words.append(word_text)
                # Normalize box
                box = word_info["box"]
                normalized_box = [
                    int(1000 * box[0] / width),
                    int(1000 * box[1] / height),
                    int(1000 * box[2] / width),
                    int(1000 * box[3] / height),
                ]
                boxes.append(normalized_box)
                true_labels.append(label)
        
        # Get predictions
        pred_labels, _ = self.predict(image, words, boxes)
        
        return image, words, boxes, true_labels, pred_labels


def visualize_predictions(
    image: Image.Image,
    words: list,
    boxes: list,
    pred_labels: list,
    true_labels: list = None,
    save_path: str = None
):
    """
    Visualize predictions on document image
    
    Args:
        image: PIL Image
        words: List of words
        boxes: List of normalized boxes [x0, y0, x1, y1] on 0-1000 scale
        pred_labels: List of predicted labels
        true_labels: List of true labels (optional)
        save_path: Path to save visualization
    """
    width, height = image.size
    
    # Create figure
    if true_labels:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        axes = [ax1, ax2]
        titles = ["Ground Truth", "Predictions"]
        label_lists = [true_labels, pred_labels]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        axes = [ax]
        titles = ["Predictions"]
        label_lists = [pred_labels]
    
    for ax, title, labels in zip(axes, titles, label_lists):
        ax.imshow(image)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Draw boxes and labels
        for word, box, label in zip(words, boxes, labels):
            # Denormalize box
            x0 = box[0] * width / 1000
            y0 = box[1] * height / 1000
            x1 = box[2] * width / 1000
            y1 = box[3] * height / 1000
            
            w = x1 - x0
            h = y1 - y0
            
            # Get color for label
            color = LABEL_COLORS.get(label, "#95a5a6")
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x0, y0), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Draw label
            if label != "O":  # Don't show O labels
                ax.text(
                    x0, y0 - 5,
                    label,
                    fontsize=8,
                    color='white',
                    weight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=color,
                        alpha=0.8,
                        edgecolor='none'
                    )
                )
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor='black', label=label)
        for label, color in LABEL_COLORS.items()
        if label != "O"
    ]
    
    if true_labels:
        ax2.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9
        )
    else:
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Visualization saved to {save_path}")
    
    plt.show()


def calculate_accuracy(true_labels: list, pred_labels: list):
    """Calculate accuracy and per-label metrics"""
    # Print first 20 comparisons for debugging
    print(f"\n🔍 First 20 TRUE vs PRED comparisons:")
    for i in range(min(20, len(true_labels))):
        true_l = true_labels[i]
        pred_l = pred_labels[i] if i < len(pred_labels) else "N/A"
        match = "✓" if true_l == pred_l else "✗"
        print(f"  {i:3d}: TRUE={true_l:15s} | PRED={pred_l:15s} {match}")
    
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Per-label accuracy
    label_correct = {}
    label_total = {}
    
    for t, p in zip(true_labels, pred_labels):
        if t not in label_total:
            label_total[t] = 0
            label_correct[t] = 0
        label_total[t] += 1
        if t == p:
            label_correct[t] += 1
    
    print(f"\n🏷️  Per-Label Accuracy:")
    for label in sorted(label_total.keys()):
        acc = label_correct[label] / label_total[label]
        print(f"  {label:15s}: {acc:.4f} ({label_correct[label]}/{label_total[label]})")


def main():
    parser = argparse.ArgumentParser(description="Inference with LayoutLMv3")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Path to JSON annotation file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Initialize predictor
    predictor = LayoutLMv3Predictor(
        model_path=args.model_path,
        device=device
    )
    
    # Run prediction
    print(f"\n🔍 Running inference on {args.image_path}...")
    image, words, boxes, true_labels, pred_labels = predictor.predict_from_json(
        image_path=args.image_path,
        annotation_path=args.annotation_path
    )
    
    print(f"  ✅ Processed {len(words)} words")
    
    # Calculate accuracy
    calculate_accuracy(true_labels, pred_labels)
    
    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(args.image_path).stem
    save_path = output_dir / f"{image_name}_prediction.png"
    
    print(f"\n📊 Creating visualization...")
    visualize_predictions(
        image=image,
        words=words,
        boxes=boxes,
        pred_labels=pred_labels,
        true_labels=true_labels,
        save_path=save_path
    )
    
    # Save predictions as JSON
    predictions_file = output_dir / f"{image_name}_predictions.json"
    predictions_data = {
        "image": str(args.image_path),
        "words": words,
        "boxes": boxes,
        "true_labels": true_labels,
        "predicted_labels": pred_labels
    }
    
    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"💾 Predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
