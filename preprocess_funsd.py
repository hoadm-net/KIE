"""
Preprocessing FUNSD dataset cho LayoutLMv3
"""
import json
import os
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor

class FUNSDDataset(Dataset):
    """
    Custom Dataset cho FUNSD
    
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
        processor: LayoutLMv3Processor = None,
        max_length: int = 512
    ):
        """
        Args:
            data_dir: Đường dẫn đến thư mục FUNSD (chứa training_data/ và testing_data/)
            split: 'train' hoặc 'test'
            processor: LayoutLMv3Processor instance
            max_length: Max sequence length
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        
        # Map split name
        if split == "train":
            self.split_dir = self.data_dir / "training_data"
        elif split == "test":
            self.split_dir = self.data_dir / "testing_data"
        else:
            raise ValueError(f"Split must be 'train' or 'test', got {split}")
        
        self.images_dir = self.split_dir / "images"
        self.annotations_dir = self.split_dir / "annotations"
        
        # Get all annotation files
        self.annotation_files = sorted(list(self.annotations_dir.glob("*.json")))
        
        # Initialize processor
        if processor is None:
            self.processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base",
                apply_ocr=False  # FUNSD đã có OCR sẵn
            )
        else:
            self.processor = processor
        
        print(f"📊 Loaded {len(self.annotation_files)} samples from {split} split")
    
    def __len__(self):
        return len(self.annotation_files)
    
    def normalize_box(self, box: List[int], width: int, height: int) -> List[int]:
        """
        Normalize bounding box to 0-1000 scale
        
        Args:
            box: [x_left, y_top, x_right, y_bottom]
            width: Image width
            height: Image height
        
        Returns:
            Normalized box [x0, y0, x1, y1] on 0-1000 scale
        """
        return [
            int(1000 * box[0] / width),
            int(1000 * box[1] / height),
            int(1000 * box[2] / width),
            int(1000 * box[3] / height),
        ]
    
    def convert_to_bio_labels(self, words: List[Dict], entity_label: str) -> List[str]:
        """
        Convert entity label sang BIO tagging scheme
        
        Args:
            words: List of word dicts
            entity_label: "question", "answer", "header", or "other"
        
        Returns:
            List of BIO labels cho mỗi word
        """
        labels = []
        for i, word in enumerate(words):
            if i == 0:
                labels.append(f"B-{entity_label.upper()}")
            else:
                labels.append(f"I-{entity_label.upper()}")
        return labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get một sample from dataset
        
        Returns:
            Dict with keys:
                - input_ids: token ids
                - attention_mask: attention mask
                - bbox: bounding boxes (normalized to 0-1000)
                - labels: entity labels (BIO scheme)
                - pixel_values: image tensor
                - original_image: PIL Image (for visualization)
        """
        # Load annotation file
        annotation_file = self.annotation_files[idx]
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load image
        image_file = self.images_dir / f"{annotation_file.stem}.png"
        image = Image.open(image_file).convert("RGB")
        width, height = image.size
        
        # Extract words, boxes, and labels
        words = []
        boxes = []
        word_labels = []
        
        for entity in data["form"]:
            entity_words = entity["words"]
            entity_label = entity["label"]
            
            # Get BIO labels for this entity
            bio_labels = self.convert_to_bio_labels(entity_words, entity_label)
            
            for word_info, label in zip(entity_words, bio_labels):
                # Skip empty or whitespace-only words
                word_text = word_info["text"]
                if not word_text or not word_text.strip():
                    continue
                    
                words.append(word_text)
                # Normalize box to 0-1000 scale
                normalized_box = self.normalize_box(
                    word_info["box"], width, height
                )
                boxes.append(normalized_box)
                word_labels.append(self.LABEL2ID[label])
        
        # Encode with processor
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension (processor adds it)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add original image for visualization
        encoding["original_image"] = image
        encoding["image_path"] = str(image_file)
        
        return encoding


def create_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 0,
    processor: LayoutLMv3Processor = None
):
    """
    Tạo train và test DataLoader
    
    Args:
        data_dir: Đường dẫn đến FUNSD dataset
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        processor: LayoutLMv3Processor instance
    
    Returns:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = FUNSDDataset(
        data_dir=data_dir,
        split="train",
        processor=processor
    )
    
    test_dataset = FUNSDDataset(
        data_dir=data_dir,
        split="test",
        processor=processor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, test_loader


def custom_collate_fn(batch):
    """
    Custom collate function để handle batching
    """
    # Separate original images (không thể stack)
    original_images = [item.pop("original_image") for item in batch]
    image_paths = [item.pop("image_path") for item in batch]
    
    # Stack tensors
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    
    # Add back original images as list
    batch_dict["original_images"] = original_images
    batch_dict["image_paths"] = image_paths
    
    return batch_dict


def test_dataset():
    """Test function để kiểm tra dataset"""
    print("="*60)
    print("TESTING FUNSD DATASET")
    print("="*60)
    
    # Initialize processor
    from transformers import LayoutLMv3Processor
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )
    
    # Create dataset
    dataset = FUNSDDataset(
        data_dir="data/FUNSD",
        split="train",
        processor=processor
    )
    
    # Get first sample
    sample = dataset[0]
    
    print(f"\n📄 Sample 0:")
    print(f"  - input_ids shape: {sample['input_ids'].shape}")
    print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  - bbox shape: {sample['bbox'].shape}")
    print(f"  - labels shape: {sample['labels'].shape}")
    print(f"  - pixel_values shape: {sample['pixel_values'].shape}")
    print(f"  - Image: {sample['image_path']}")
    
    # Decode some tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(
        sample['input_ids'][:50]
    )
    print(f"\n📝 First 50 tokens:")
    print(tokens)
    
    # Show label distribution
    labels = sample['labels']
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"\n🏷️  Label distribution:")
    for label_id, count in zip(unique_labels, counts):
        if label_id != -100:  # Ignore padding
            label_name = dataset.ID2LABEL.get(label_id.item(), "UNKNOWN")
            print(f"  {label_name}: {count.item()}")
    
    print("\n✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
