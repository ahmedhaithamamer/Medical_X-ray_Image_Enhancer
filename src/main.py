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
                   use_unsharp: bool = True) -> None:
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

    for path in image_paths:
        img_name = Path(path).stem
        img = load_grayscale(path)

        outputs = enhancement_pipeline(
            img,
            use_clahe=use_clahe,
            use_bilateral=use_bilateral,
            use_unsharp=use_unsharp
        )

        # Create a folder for this image's outputs
        image_output_dir = RESULTS_DIR / img_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics for all enhancement techniques
        for key in metrics_keys:
            if key in outputs:
                m = mse(outputs["original"], outputs[key])
                p = psnr(outputs["original"], outputs[key])
                print(f"{img_name} | {key:12s} | MSE: {m:10.2f} | PSNR: {p:6.2f} dB")

        # Save individual images to image-specific folder
        for key, out_img in outputs.items():
            out_path = image_output_dir / f"{key}.png"
            save_image(str(out_path), out_img)


if __name__ == "__main__":
    process_dataset(
        use_clahe=True,
        use_bilateral=True,
        use_unsharp=True
    )


