"""
Preprocessing FUNSD dataset cho LayoutXLM
"""
import json
import os
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutXLMProcessor

class FUNSDDataset(Dataset):
    """
    Custom Dataset cho FUNSD với LayoutXLM
    
    FUNSD format:
    - images/: chứa ảnh documents (.png)
    - annotations/: chứa file JSON với structure:
        {
            "form": [
                {
                    "id": int,
                    "text": str,
                    "box": [x_left, y_top, x_right, y_bottom],
                    "label": "question" | "answer" | "header" | "other",
                    "words": [
                        {
                            "text": str,
                            "box": [x_left, y_top, x_right, y_bottom]
                        },
                        ...
                    ],
                    "linking": [[from_id, to_id], ...]
                },
                ...
            ]
        }
    """
    
    # Label mapping cho entity recognition
    LABEL2ID = {
        "O": 0,          # Outside
        "B-QUESTION": 1, # Beginning of Question
        "I-QUESTION": 2, # Inside Question  
        "B-ANSWER": 3,   # Beginning of Answer
        "I-ANSWER": 4,   # Inside Answer
        "B-HEADER": 5,   # Beginning of Header
        "I-HEADER": 6,   # Inside Header
        "B-OTHER": 7,    # Beginning of Other
        "I-OTHER": 8     # Inside Other
    }
    
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        processor: LayoutXLMProcessor = None,
        max_length: int = 512
    ):
        """
        Args:
            data_dir: Đường dẫn đến thư mục FUNSD dataset
            split: "train" hoặc "test"
            processor: LayoutXLMProcessor instance
            max_length: Max sequence length
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.max_length = max_length
        
        # Đường dẫn tới images và annotations
        # FUNSD structure: training_data/ và testing_data/
        split_dir = "training_data" if split == "train" else "testing_data"
        self.img_dir = self.data_dir / split_dir / "images"
        self.anno_dir = self.data_dir / split_dir / "annotations"
        
        # Lấy danh sách file annotations
        self.annotation_files = sorted(list(self.anno_dir.glob("*.json")))
        
        print(f"Loaded {len(self.annotation_files)} {split} examples from {self.data_dir}")
        
    def __len__(self):
        return len(self.annotation_files)
    
    def normalize_box(self, box: List[int], width: int, height: int) -> List[int]:
        """
        Normalize bounding box coordinates sang scale 0-1000
        
        Args:
            box: [x_left, y_top, x_right, y_bottom] trong pixel coordinates
            width: Image width
            height: Image height
            
        Returns:
            normalized_box: [x_left, y_top, x_right, y_bottom] trong scale 0-1000
        """
        return [
            int(1000 * box[0] / width),
            int(1000 * box[1] / height),
            int(1000 * box[2] / width),
            int(1000 * box[3] / height)
        ]
    
    def convert_to_bio_labels(self, entities: List[Dict]) -> Tuple[List[str], List[List[int]]]:
        """
        Convert FUNSD entities sang BIO tagging format
        
        Args:
            entities: List of entity dictionaries từ FUNSD annotation
            
        Returns:
            words: List of words
            boxes: List of bounding boxes (normalized)
            labels: List of BIO labels
        """
        words = []
        boxes = []
        labels = []
        
        for entity in entities:
            label_type = entity["label"].upper()
            entity_words = entity["words"]
            
            for idx, word_info in enumerate(entity_words):
                # Filter empty or whitespace-only words (MUST match training!)
                word_text = word_info["text"]
                if not word_text or not word_text.strip():
                    continue
                
                words.append(word_text)
                boxes.append(word_info["box"])
                
                # BIO tagging: first word gets B- prefix, rest get I- prefix
                if idx == 0:
                    labels.append(f"B-{label_type}")
                else:
                    labels.append(f"I-{label_type}")
        
        return words, boxes, labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Lấy một sample từ dataset
        
        Returns:
            Dict chứa input_ids, bbox, attention_mask, pixel_values, labels
        """
        # Load annotation JSON
        anno_path = self.annotation_files[idx]
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load corresponding image
        img_filename = anno_path.stem + ".png"
        img_path = self.img_dir / img_filename
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Extract words, boxes, labels từ annotations
        words, boxes, labels_str = self.convert_to_bio_labels(data["form"])
        
        # Normalize bounding boxes
        boxes = [self.normalize_box(box, width, height) for box in boxes]
        
        # Convert label strings sang IDs
        labels = [self.LABEL2ID[label] for label in labels_str]
        
        # Sử dụng processor để encode inputs
        # LayoutXLMProcessor yêu cầu word_labels parameter
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension (processor thêm batch dim)
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        return encoding


def create_dataloaders(
    data_dir: str,
    processor: LayoutXLMProcessor,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train và test sets
    
    Args:
        data_dir: Đường dẫn đến FUNSD dataset
        processor: LayoutXLMProcessor instance
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of workers cho DataLoader
        
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = FUNSDDataset(
        data_dir=data_dir,
        split="train",
        processor=processor,
        max_length=max_length
    )
    
    test_dataset = FUNSDDataset(
        data_dir=data_dir,
        split="test",
        processor=processor,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    """Test preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/FUNSD")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutxlm-base")
    args = parser.parse_args()
    
    # Initialize processor
    processor = LayoutXLMProcessor.from_pretrained(args.model_name)
    
    # Create dataset
    train_dataset = FUNSDDataset(
        data_dir=args.data_dir,
        split="train",
        processor=processor
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    
    # Test first sample
    sample = train_dataset[0]
    print("\nFirst sample keys:", sample.keys())
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Bbox shape:", sample["bbox"].shape)
    print("Labels shape:", sample["labels"].shape)
    print("Pixel values shape:", sample["pixel_values"].shape)
    print("\nFirst 10 tokens:")
    for i in range(10):
        token_id = sample["input_ids"][i].item()
        bbox = sample["bbox"][i].tolist()
        label_id = sample["labels"][i].item()
        label = train_dataset.ID2LABEL.get(label_id, f"ID:{label_id}")
        token = processor.tokenizer.decode([token_id])
        print(f"  {i}: token='{token}' bbox={bbox} label={label}")
