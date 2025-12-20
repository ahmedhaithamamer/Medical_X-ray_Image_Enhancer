import os
from pathlib import Path

from enhancement import enhancement_pipeline
from io_utils import list_images, load_grayscale, save_image
from metrics import mse, psnr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def process_dataset(use_clahe: bool = True,
                   use_bilateral: bool = True,
                   use_unsharp: bool = True,
                   include_best_pipeline: bool = True) -> None:
    image_paths = list_images(str(DATA_DIR))
    if not image_paths:
        print(f"No images found in {DATA_DIR}. "
              f"Please add some PNG/JPG X-ray images.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Define which outputs to compute metrics for
    metrics_keys = ["median", "contrast", "sharpen"]
    if use_bilateral:
        metrics_keys.append("bilateral")
    if use_clahe:
        metrics_keys.append("clahe")
    if use_unsharp:
        metrics_keys.append("unsharp")
    if include_best_pipeline:
        metrics_keys.append("best_pipeline")

    print("=" * 80)
    print("Medical X-ray Image Enhancement - Sequential Pipeline Processing")
    print("=" * 80)
    print(f"Processing {len(image_paths)} images...")
    print(f"Best Pipeline: Median → CLAHE → Unsharp")
    print("=" * 80)

    for path in image_paths:
        img_name = Path(path).stem
        img = load_grayscale(path)

        # Apply enhancement pipeline with sequential processing
        outputs = enhancement_pipeline(
            img,
            use_clahe=use_clahe,
            use_bilateral=use_bilateral,
            use_unsharp=use_unsharp,
            include_best_pipeline=include_best_pipeline
        )

        # Create a folder for this image's outputs
        image_output_dir = RESULTS_DIR / img_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics for all enhancement techniques
        print(f"\n{img_name}:")
        for key in metrics_keys:
            if key in outputs:
                m = mse(outputs["original"], outputs[key])
                p = psnr(outputs["original"], outputs[key])
                technique_name = key.replace("_", " ").title()
                print(f"  {technique_name:20s} | MSE: {m:10.2f} | PSNR: {p:6.2f} dB")

        # Save individual images to image-specific folder
        for key, out_img in outputs.items():
            out_path = image_output_dir / f"{key}.png"
            save_image(str(out_path), out_img)
    
    print("\n" + "=" * 80)
    print("Processing complete! Results saved to results/ directory")
    print("=" * 80)


if __name__ == "__main__":
    # Process dataset with optimal sequential pipeline
    # Best Pipeline: Median Filter → CLAHE → Unsharp Masking
    process_dataset(
        use_clahe=True,
        use_bilateral=True,
        use_unsharp=True,
        include_best_pipeline=True
    )


