## ECE354 Project – Medical X-ray Image Enhancement

This repository contains the code for the course project:
**"Medical X-ray Image Enhancement using Filtering & Contrast Methods"**
for **ECE354: Image and Video Processing**.

### Structure

- `data/` – Input images (e.g., NIH Chest X-ray subset). Not tracked in Git.
- `results/` – Processed images and figures.
- `src/` – Source code:
  - `main.py` – Entry point to run experiments.
  - `enhancement.py` – Image enhancement functions (median filter, hist. equalization, sharpening).
  - `metrics.py` – MSE and PSNR computation.
  - `io_utils.py` – Helper utilities for loading and saving images.
- `report/` – LaTeX/Word report, poster, and any supporting material.

### Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **Get a small subset of chest X-ray images** (you only need 20-50 images for this project!):
   
   **Option A: Use a pre-processed subset** (recommended)
   - Download a small subset from: https://github.com/MichaelNoya/nih-chest-xray-webdataset-subset
   - This is much smaller (~few GB) and already organized
   
   **Option B: Manual download from NIH dataset**
   - Visit: https://www.innovatiana.com/en/datasets/nih-chest-x-rays
   - Download only 20-50 sample images (not the full 45GB!)
   - Place them in the `data/` folder
   
   **Option C: Use any chest X-ray images**
   - You can use any chest X-ray images you have access to
   - Just make sure they're in PNG or JPG format
   - Place them in the `data/` folder
   
   **Note:** For a course project, 20-50 images is more than enough to:
   - Demonstrate all three techniques (median filter, histogram equalization, sharpening)
   - Show visual comparisons
   - Compute MSE/PSNR metrics
   - Create good report figures

### Running

From the project root:

```bash
python -m src.main
```

This will:
- Load images from `data/`
- Apply median filtering, histogram equalization, and sharpening
- Save processed images into `results/`


