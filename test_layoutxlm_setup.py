#!/usr/bin/env python
"""
Quick test script để kiểm tra LayoutXLM setup
"""
import sys

print("=" * 60)
print("Testing LayoutXLM Setup")
print("=" * 60)

# Test 1: Import transformers
print("\n1. Testing transformers import...")
try:
    import transformers
    print(f"   ✓ transformers version: {transformers.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import LayoutXLMProcessor
print("\n2. Testing LayoutXLMProcessor import...")
try:
    from transformers import LayoutXLMProcessor
    print("   ✓ LayoutXLMProcessor imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Load processor
print("\n3. Testing processor loading...")
try:
    processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
    print("   ✓ Processor loaded successfully")
    print(f"   - Tokenizer: {type(processor.tokenizer).__name__}")
    print(f"   - Image processor: {type(processor.image_processor).__name__}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Import LayoutLMv2ForTokenClassification
print("\n4. Testing LayoutLMv2ForTokenClassification import...")
try:
    from transformers import LayoutLMv2ForTokenClassification
    print("   ✓ LayoutLMv2ForTokenClassification imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Load model
print("\n5. Testing model loading...")
try:
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutxlm-base",
        num_labels=9
    )
    print("   ✓ Model loaded successfully")
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - Num labels: {model.config.num_labels}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 6: Check FUNSD data
print("\n6. Checking FUNSD data...")
import os
from pathlib import Path

data_dir = Path("data/FUNSD")
if not data_dir.exists():
    print(f"   ✗ FUNSD directory not found: {data_dir}")
    sys.exit(1)

train_dir = data_dir / "training_data"
test_dir = data_dir / "testing_data"

if not train_dir.exists():
    print(f"   ✗ Train directory not found: {train_dir}")
    sys.exit(1)

if not test_dir.exists():
    print(f"   ✗ Test directory not found: {test_dir}")
    sys.exit(1)

train_annos = list((train_dir / "annotations").glob("*.json"))
test_annos = list((test_dir / "annotations").glob("*.json"))

print(f"   ✓ FUNSD data found")
print(f"   - Train samples: {len(train_annos)}")
print(f"   - Test samples: {len(test_annos)}")

# Test 7: Test preprocessing
print("\n7. Testing preprocessing...")
try:
    from preprocess_funsd_layoutxlm import FUNSDDataset
    
    dataset = FUNSDDataset(
        data_dir="data/FUNSD",
        split="train",
        processor=processor
    )
    
    print(f"   ✓ Dataset created successfully")
    print(f"   - Size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"   ✓ Sample loaded successfully")
    print(f"   - Keys: {list(sample.keys())}")
    print(f"   - input_ids shape: {sample['input_ids'].shape}")
    print(f"   - bbox shape: {sample['bbox'].shape}")
    print(f"   - labels shape: {sample['labels'].shape}")
    # LayoutXLM uses 'image' key instead of 'pixel_values'
    if 'image' in sample:
        print(f"   - image shape: {sample['image'].shape}")
    elif 'pixel_values' in sample:
        print(f"   - pixel_values shape: {sample['pixel_values'].shape}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
