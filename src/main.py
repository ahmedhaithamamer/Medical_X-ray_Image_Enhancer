"""
Entry point for running the ECE354 image enhancement experiments.

Usage (from project root):
    python -m src.main
"""

import os
from pathlib import Path

from enhancement import enhancement_pipeline
from io_utils import list_images, load_grayscale, save_image
from metrics import mse, psnr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def process_dataset(use_clahe: bool = False) -> None:
    """
    Process all images in the data directory and save enhanced versions
    plus print simple MSE/PSNR metrics.
    """
    image_paths = list_images(str(DATA_DIR))
    if not image_paths:
        print(f"No images found in {DATA_DIR}. "
              f"Please add some PNG/JPG X-ray images.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for path in image_paths:
        img_name = Path(path).stem
        img = load_grayscale(path)

        outputs = enhancement_pipeline(img, use_clahe=use_clahe)

        # Compute metrics: original vs median, contrast, sharpen
        for key in ["median", "contrast", "sharpen"]:
            m = mse(outputs["original"], outputs[key])
            p = psnr(outputs["original"], outputs[key])
            print(f"{img_name} | {key:8s} | MSE: {m:10.2f} | PSNR: {p:6.2f} dB")

        # Save images to results folder
        for key, out_img in outputs.items():
            out_path = RESULTS_DIR / f"{img_name}_{key}.png"
            save_image(str(out_path), out_img)


if __name__ == "__main__":
    # Set use_clahe=True if you want to use CLAHE instead of global hist. eq.
    process_dataset(use_clahe=False)


