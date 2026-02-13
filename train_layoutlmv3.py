"""
Train LayoutLMv3 trên FUNSD dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

from preprocess_funsd import FUNSDDataset, create_dataloaders


class LayoutLMv3Trainer:
    """Trainer class cho LayoutLMv3"""
    
    def __init__(
        self,
        model: LayoutLMv3ForTokenClassification,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        output_dir: str = "outputs",
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # History
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_precision": [],
            "eval_recall": [],
            "eval_f1": []
        }
        
        print(f"📊 Trainer initialized:")
        print(f"  - Device: {device}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Num epochs: {num_epochs}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
    
    def train_epoch(self, epoch: int):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]"
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bbox = batch["bbox"].to(self.device)
            labels = batch["labels"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Only count non-padding tokens
            mask = labels != -100
            correct_preds += ((predictions == labels) & mask).sum().item()
            total_preds += mask.sum().item()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            current_acc = correct_preds / total_preds if total_preds > 0 else 0
            progress_bar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "acc": f"{current_acc:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct_preds / total_preds if total_preds > 0 else 0
        
        return avg_loss, avg_acc
    
    def evaluate(self):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                total_loss += loss.item()
                
                # Collect predictions and labels (excluding padding)
                mask = labels != -100
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Calculate per-class precision, recall, F1
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")
        
        best_f1 = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = self.evaluate()
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["eval_loss"].append(eval_loss)
            self.history["eval_accuracy"].append(eval_acc)
            self.history["eval_precision"].append(eval_precision)
            self.history["eval_recall"].append(eval_recall)
            self.history["eval_f1"].append(eval_f1)
            
            # Print metrics
            print(f"\n📊 Epoch {epoch+1}/{self.num_epochs} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {eval_acc:.4f}")
            print(f"  Precision: {eval_precision:.4f} | Recall: {eval_recall:.4f} | F1: {eval_f1:.4f}")
            
            # Save best model
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                self.save_model("best_model")
                print(f"  🎯 New best F1 score: {best_f1:.4f} - Model saved!")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}")
            
            print()
        
        # Save final model
        self.save_model("final_model")
        self.save_history()
        
        print("="*60)
        print("TRAINING COMPLETED")
        print(f"Best F1 Score: {best_f1:.4f}")
        print("="*60)
    
    def save_model(self, name: str):
        """Save model"""
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        print(f"  💾 Model saved to {save_dir}")
    
    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  📊 Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 on FUNSD")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/FUNSD",
        help="Path to FUNSD dataset directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/layoutlmv3-base",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Initialize processor
    print(f"\n📦 Loading processor from {args.model_name}...")
    processor = LayoutLMv3Processor.from_pretrained(
        args.model_name,
        apply_ocr=False
    )
    
    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        processor=processor
    )
    
    # Initialize model
    print(f"\n🤖 Loading model from {args.model_name}...")
    num_labels = 9  # 9 BIO labels for FUNSD
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    
    # Save config
    config_file = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  📝 Config saved to {config_file}")
    
    # Initialize trainer
    trainer = LayoutLMv3Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=output_dir,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
