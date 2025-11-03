"""
Quick script to test RepE setup and data loading.
Run this before submitting to SLURM to catch any issues early.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    - CUDA version: {torch.version.cuda}")
            print(f"    - GPU count: {torch.cuda.device_count()}")
            print(f"    - GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_data():
    """Test that CLUTRR data is available and valid."""
    print("Testing data...")
    
    data_dir = Path("data/clutrr")
    
    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print("    Run: python download_data.py")
        return False
    
    import json
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f"{split}.json"
        
        if not file_path.exists():
            print(f"  ✗ Missing {split}.json")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"  ✓ {split:5s}: {len(data):5d} examples")
            
            # Check first example structure
            if len(data) > 0:
                example = data[0]
                required_keys = ['story', 'query', 'target']
                missing_keys = [k for k in required_keys if k not in example]
                if missing_keys:
                    print(f"    ⚠ Missing keys in example: {missing_keys}")
        
        except json.JSONDecodeError as e:
            print(f"  ✗ Invalid JSON in {split}.json: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Error reading {split}.json: {e}")
            return False
    
    print("\n✓ All data files present and valid!\n")
    return True


def test_model():
    """Test that RepE model can be initialized."""
    print("Testing model initialization...")
    
    try:
        from repe_clutrr import RepEConfig, RepEModel
        from transformers import AutoTokenizer
        
        config = RepEConfig(
            model_name='gpt2',
            batch_size=2,
            device='cpu'  # Use CPU for testing
        )
        
        print(f"  Creating config...")
        print(f"    - Model: {config.model_name}")
        print(f"    - Device: {config.device}")
        
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  Initializing model...")
        model = RepEModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"    - Total parameters: {total_params:,}")
        print(f"    - Trainable parameters: {trainable_params:,}")
        
        print("\n✓ Model initialized successfully!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test that dataloader works."""
    print("Testing dataloader...")
    
    try:
        from repe_clutrr import RepEConfig, create_dataloaders
        from transformers import AutoTokenizer
        
        config = RepEConfig(
            model_name='gpt2',
            batch_size=2,
            device='cpu'
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataloaders = create_dataloaders(
            config=config,
            tokenizer=tokenizer,
            data_dir=Path("data/clutrr")
        )
        
        # Try loading one batch
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"  ✓ Loaded batch:")
        print(f"    - input_ids shape: {batch['input_ids'].shape}")
        print(f"    - attention_mask shape: {batch['attention_mask'].shape}")
        print(f"    - Number of examples: {len(batch['target'])}")
        
        print("\n✓ Dataloader working correctly!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error with dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RepE CLUTRR Setup Test")
    print("="*60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data", test_data()))
    results.append(("Model", test_model()))
    results.append(("Dataloader", test_dataloader()))
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:15s}: {status}")
    
    print()
    
    if all(r[1] for r in results):
        print("✓ All tests passed! Ready to submit to SLURM.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
