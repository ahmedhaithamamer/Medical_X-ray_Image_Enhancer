"""
Helper script to download a small sample of NIH Chest X-ray images.

Since the full dataset is ~45GB, this script helps you download only
a manageable subset (e.g., 20-50 images) for your project.

Usage:
    python -m src.download_sample --num_images 30
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Please install required packages: pip install requests tqdm")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image from URL."""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def download_nih_sample(num_images: int = 30, 
                       data_dir: Optional[Path] = None) -> None:
    """
    Download a sample of NIH Chest X-ray images.
    
    Note: This assumes you have access to the NIH dataset URLs.
    You may need to:
    1. Download the dataset metadata CSV first
    2. Extract image URLs from the CSV
    3. Download a random subset
    
    For now, this is a template. You'll need to adapt it based on
    how the NIH dataset is actually distributed.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} sample images to {data_dir}")
    print("\nNOTE: The NIH Chest X-ray dataset distribution method may vary.")
    print("You may need to:")
    print("1. Download the dataset metadata CSV from the NIH website")
    print("2. Extract image URLs/IDs from the CSV")
    print("3. Download a random subset of images")
    print("\nAlternatively, you can manually download 20-50 images from:")
    print("https://www.innovatiana.com/en/datasets/nih-chest-x-rays")
    print("\nOr use a pre-existing subset if available.\n")
    
    # TODO: If you have a CSV with image URLs, you can implement:
    # 1. Read CSV
    # 2. Sample num_images rows
    # 3. Download each image
    
    print(f"Place {num_images} chest X-ray images (PNG/JPG) in: {data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a sample subset of NIH Chest X-ray images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=30,
        help="Number of images to download (default: 30)"
    )
    args = parser.parse_args()
    
    download_nih_sample(num_images=args.num_images)


if __name__ == "__main__":
    main()

