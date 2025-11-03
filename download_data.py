"""
Download CLUTRR dataset from the lo-fit repository.
"""

import os
import json
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/fc2869/lo-fit/main/dataset/clutrr"
DATA_DIR = Path("data/clutrr")
SPLITS = ["train.json", "val.json", "test.json"]


def download_file(url: str, output_path: Path):
    """Download a file from URL to output path."""
    print(f"Downloading {url} to {output_path}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            
        with open(output_path, 'wb') as f:
            f.write(data)
            
        print(f"✓ Downloaded {output_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def validate_json(file_path: Path):
    """Validate that the downloaded file is valid JSON."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"  - Valid JSON with {len(data)} entries")
        return True
        
    except json.JSONDecodeError as e:
        print(f"  - Invalid JSON: {e}")
        return False


def main():
    """Download all CLUTRR dataset splits."""
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {DATA_DIR}\n")
    
    # Download each split
    success_count = 0
    for split in SPLITS:
        url = f"{BASE_URL}/{split}"
        output_path = DATA_DIR / split
        
        if download_file(url, output_path):
            if validate_json(output_path):
                success_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"Successfully downloaded {success_count}/{len(SPLITS)} files")
    
    if success_count == len(SPLITS):
        print("✓ All dataset files downloaded successfully!")
        print(f"\nDataset location: {DATA_DIR.absolute()}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        for split in SPLITS:
            file_path = DATA_DIR / split
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"  {split:12s}: {len(data):5d} examples")
    else:
        print("✗ Some files failed to download. Please check errors above.")


if __name__ == "__main__":
    main()
