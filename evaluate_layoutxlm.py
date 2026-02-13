"""
Evaluate LayoutXLM model trên FUNSD dataset
"""
import torch
from transformers import LayoutLMv2ForTokenClassification, LayoutXLMProcessor
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess_funsd_layoutxlm import FUNSDDataset, create_dataloaders


class LayoutXLMEvaluator:
    """Evaluator class cho LayoutXLM"""
    
    def __init__(
        self,
        model: LayoutLMv2ForTokenClassification,
        test_loader,
        device: str = "cuda",
        id2label: dict = None
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.id2label = id2label or FUNSDDataset.ID2LABEL
    
    def evaluate(self, verbose: bool = True):
        """
        Evaluate model and return detailed metrics
        
        Returns:
            dict with metrics: loss, accuracy, precision, recall, f1, 
                               per_class_metrics, confusion_matrix
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.test_loader, desc="Evaluating") if verbose else self.test_loader
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                labels = batch["labels"].to(self.device)
                image = batch["image"].to(self.device) if "image" in batch else None
                
                # Forward pass - LayoutXLM uses 'image' parameter
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    image=image,
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
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate overall metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                all_labels,
                all_predictions,
                average=None,
                zero_division=0
            )
        
        # Get unique labels that actually appear in data
        unique_labels = sorted(set(all_labels.tolist() + all_predictions.tolist()))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
        
        # Classification report
        target_names = [self.id2label[i] for i in unique_labels]
        report = classification_report(
            all_labels,
            all_predictions,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
        
        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_metrics": {
                self.id2label[i]: {
                    "precision": per_class_precision[i],
                    "recall": per_class_recall[i],
                    "f1": per_class_f1[i],
                    "support": int(per_class_support[i])
                }
                for i in range(len(per_class_precision))
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        return results
    
    def print_results(self, results: dict):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS - LayoutXLM")
        print("="*60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Loss:      {results['loss']:.4f}")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        
        print(f"\n🏷️  Per-Class Metrics:")
        for label, metrics in results['per_class_metrics'].items():
            print(f"\n  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1:        {metrics['f1']:.4f}")
            print(f"    Support:   {metrics['support']}")
        
        print(f"\n📋 Classification Report:")
        print(results['classification_report'])
        
        print("="*60)
    
    def plot_confusion_matrix(self, results: dict, save_path: str = None):
        """Plot confusion matrix"""
        cm = np.array(results['confusion_matrix'])
        labels = [self.id2label[i] for i in range(len(cm))]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix - LayoutXLM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Confusion matrix saved to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LayoutXLM on FUNSD")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/FUNSD",
        help="Path to FUNSD dataset directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results_layoutxlm",
        help="Output directory for results"
    )
    parser.add_argument(
        "--plot_cm",
        action="store_true",
        help="Plot confusion matrix"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Initialize processor
    print(f"\n📦 Loading processor...")
    processor = LayoutXLMProcessor.from_pretrained(
        "microsoft/layoutxlm-base",
        apply_ocr=False
    )
    
    # Create test dataloader
    print(f"\n📊 Creating test dataloader...")
    from preprocess_funsd_layoutxlm import FUNSDDataset
    from torch.utils.data import DataLoader
    
    test_dataset = FUNSDDataset(
        data_dir=args.data_dir,
        split="test",
        processor=processor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"\n🤖 Loading model from {args.model_path}...")
    model = LayoutLMv2ForTokenClassification.from_pretrained(args.model_path)
    
    # Create evaluator
    evaluator = LayoutXLMEvaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Evaluate
    print("\n🔍 Starting evaluation...")
    results = evaluator.evaluate(verbose=True)
    
    # Print results
    evaluator.print_results(results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to JSON
    results_file = output_dir / "evaluation_results.json"
    # Convert numpy types to Python types for JSON serialization
    json_results = {
        "loss": float(results["loss"]),
        "accuracy": float(results["accuracy"]),
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
        "f1": float(results["f1"]),
        "per_class_metrics": results["per_class_metrics"],
        "confusion_matrix": results["confusion_matrix"]
    }
    
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Save classification report
    report_file = output_dir / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(results["classification_report"])
    print(f"💾 Classification report saved to {report_file}")
    
    # Plot confusion matrix
    if args.plot_cm:
        cm_path = output_dir / "confusion_matrix.png"
        evaluator.plot_confusion_matrix(results, save_path=cm_path)


if __name__ == "__main__":
    main()
