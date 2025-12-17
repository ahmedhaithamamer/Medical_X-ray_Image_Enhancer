# Medical X-ray Image Enhancement

**ECE354 Project – Image and Video Processing**

This repository contains a comprehensive implementation of medical X-ray image enhancement techniques using filtering and contrast methods.

## Overview

This project implements and evaluates multiple image enhancement techniques for medical X-ray images, including noise reduction, contrast enhancement, and detail enhancement methods. The project includes both batch processing capabilities and an interactive web-based viewer for real-time comparison and analysis.

## Features

### Enhancement Techniques

**Noise Reduction:**
- **Median Filter**: Non-linear filtering effective for removing impulse noise while preserving edges
- **Bilateral Filter**: Edge-preserving noise reduction using spatial and intensity domain filtering

**Contrast Enhancement:**
- **Histogram Equalization**: Global contrast enhancement by redistributing pixel intensities
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Adaptive local contrast enhancement with clipping limit

**Detail Enhancement:**
- **Sharpening**: Convolution-based sharpening using a 3×3 kernel
- **Unsharp Masking**: Advanced sharpening technique using Gaussian blur and weighted combination

### Evaluation Metrics

- **MSE** (Mean Squared Error): Measures average squared difference between original and enhanced images
- **PSNR** (Peak Signal-to-Noise Ratio): Quantifies image quality in decibels

### Interactive Viewer

- Web-based interface using Streamlit
- Image upload capability (standalone operation)
- Before/after comparison with interactive slider
- Side-by-side comparison view
- Grid view of all enhancement techniques
- Real-time metrics display organized by category
- Download all enhanced images as ZIP file

## Project Structure

```
Medical_X-ray_Image_Enhancement/
├── data/                    # Input X-ray images (not tracked in Git)
├── results/                 # Processed images (organized by image name)
│   └── <image_name>/
│       ├── original.png
│       ├── median.png
│       ├── bilateral.png
│       ├── contrast.png
│       ├── clahe.png
│       ├── sharpen.png
│       └── unsharp.png
├── src/                     # Source code
│   ├── main.py             # Batch processing entry point
│   ├── enhancement.py      # Core enhancement functions
│   ├── metrics.py          # MSE and PSNR computation
│   ├── io_utils.py         # Image I/O utilities
│   └── streamlit_app.py    # Interactive web viewer
├── report/                  # LaTeX report and documentation
│   └── main.tex
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**

2. **Create and activate a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

- `opencv-python` - Image processing operations
- `numpy` - Numerical computations
- `matplotlib` - Visualization (for report generation)
- `scikit-image` - Additional image processing utilities
- `streamlit` - Interactive web application framework
- `pillow` - Image handling
- `pandas` - Data manipulation (for Streamlit)

## Dataset

### Dataset Description

> "We used a subset of 30 chest X-ray images from the NIH Chest X-ray dataset 
> (Innovatiana, 2024). The full dataset contains over 100,000 images (~45GB), 
> but for this project we selected a representative sample to demonstrate our 
> enhancement techniques. Images were resized to a standard resolution and 
> converted to grayscale for processing."

### Getting Images

**Option 1: Manual Download (Recommended)**
1. Visit: https://www.innovatiana.com/en/datasets/nih-chest-x-rays
2. Download 20-50 sample images (not the full 45GB dataset!)
3. Place PNG/JPG images in the `data/` folder

**Option 2: Use Your Own Images**
- Place any chest X-ray images (PNG, JPG, JPEG) in the `data/` folder
- Images will be automatically converted to grayscale during processing

## Usage

### Batch Processing

Process all images in the `data/` folder:

```bash
python -m src.main
```

This will:
- Load all images from `data/`
- Apply all enhancement techniques:
  - Median filtering
  - Bilateral filtering (if enabled)
  - Histogram equalization
  - CLAHE
  - Sharpening
  - Unsharp masking (if enabled)
- Compute MSE and PSNR metrics for each technique
- Save results to `results/<image_name>/` folders
- Print metrics to console

**Output Structure:**
Each processed image gets its own folder in `results/` containing:
- `original.png` - Original input image
- `median.png` - Median filtered result
- `bilateral.png` - Bilateral filtered result
- `contrast.png` - Histogram equalized result
- `clahe.png` - CLAHE result
- `sharpen.png` - Sharpened result
- `unsharp.png` - Unsharp masking result

### Interactive Viewer (Streamlit)

Launch the interactive web-based viewer:

```bash
streamlit run src/streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

#### Features

- **Image Upload**: Upload your own X-ray images directly in the browser (standalone operation)
- **Before/After Slider**: Interactive slider to blend between original and enhanced images
- **Side-by-Side Comparison**: Compare any two techniques side-by-side with metrics
- **Grid View**: View all enhancement techniques in a grid layout
- **Metrics Display**: See MSE and PSNR values organized by category (Noise Reduction, Contrast Enhancement, Detail Enhancement)
- **Live Processing**: Process images on-the-fly or load pre-computed results
- **Download All**: Download all enhanced images as a ZIP file (for uploaded images)

#### Usage Guide

1. **Select Image Source**:
   - Choose "Upload Image" to upload your own X-ray image (no pre-processing needed)
   - Or select "Select from Dataset" to use pre-processed images from `data/` folder

2. **Before/After Comparison**:
   - "Before" is always set to "Original"
   - Select "After" technique from dropdown
   - Use the slider to blend between the two images
   - Drag left to see more of "Before", right for "After"

3. **Side-by-Side View**:
   - Compare two techniques side-by-side
   - Metrics (MSE and PSNR) are displayed below each image

4. **Grid View**:
   - Toggle techniques on/off using checkboxes in the sidebar
   - View all selected techniques in a grid layout
   - Metrics are shown as captions below each image

5. **Processing Options**:
   - **Process Live**: Recompute enhancements on-the-fly (slower but always up-to-date)
   - **Load from Results**: Load pre-computed results from `results/` folder (faster, only for dataset images)

6. **Download Results**:
   - For uploaded images, download all enhanced versions as a ZIP file
   - Includes all individual enhanced images

#### Tips

- Use the sidebar controls to customize your view
- The slider is great for presentations and demos
- Metrics help quantify the enhancement quality
- Grid view is perfect for comparing all techniques at once
- Upload feature makes it a standalone tool - no need to pre-process images

#### Troubleshooting

**Problem**: "No images found"
- **Solution**: Make sure you have images in the `data/` folder and have run the enhancement pipeline, or use the "Upload Image" option

**Problem**: "Image not found"
- **Solution**: Run `python -m src.main` first to process images, or enable "Process Live", or use "Upload Image"

**Problem**: App won't start
- **Solution**: Make sure Streamlit is installed: `pip install streamlit` or `pip install -r requirements.txt`

## Enhancement Pipeline

The enhancement pipeline processes images in the following stages:

1. **Noise Reduction**: Apply median filter (and optionally bilateral filter) to reduce noise
2. **Contrast Enhancement**: Apply histogram equalization and CLAHE to the denoised image
3. **Detail Enhancement**: Apply sharpening and unsharp masking to the contrast-enhanced image

**Note**: Both histogram equalization and CLAHE are always computed for comparison, but the `use_clahe` parameter determines which one is used as the base for detail enhancement.

## Metrics Interpretation

- **MSE (Mean Squared Error)**: Lower is better (measures difference from original)
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (measures quality in dB)

**Expected Ranges:**
- **Noise Reduction**: MSE < 20, PSNR > 35 dB (minimal changes, high fidelity)
- **Contrast Enhancement**: MSE 200-2000, PSNR 15-25 dB (intentional significant changes)
- **Detail Enhancement**: MSE 300-500, PSNR 20-25 dB (moderate changes for detail enhancement)

## Code Structure

### `src/enhancement.py`
Contains all enhancement functions:
- `apply_median_filter()` - Median filtering
- `apply_bilateral_filter()` - Bilateral filtering
- `apply_hist_eq()` - Global histogram equalization
- `apply_clahe()` - CLAHE contrast enhancement
- `apply_sharpen()` - Convolution-based sharpening
- `apply_unsharp_masking()` - Unsharp masking
- `enhancement_pipeline()` - Main pipeline function

### `src/metrics.py`
Evaluation metrics:
- `mse()` - Mean Squared Error calculation
- `psnr()` - Peak Signal-to-Noise Ratio calculation

### `src/io_utils.py`
Image I/O utilities:
- `list_images()` - Find all images in a directory
- `load_grayscale()` - Load image as grayscale
- `save_image()` - Save image to disk

### `src/main.py`
Batch processing script:
- `process_dataset()` - Process all images in data/ folder

### `src/streamlit_app.py`
Interactive web viewer:
- Image upload and processing
- Interactive comparison tools
- Metrics visualization
- Download functionality

## Report

The LaTeX report is located in `report/main.tex`. It includes:
- Abstract and introduction
- Background and methodology
- Implementation details
- Results and discussion
- Quantitative metrics tables
- Visual results

## License

This project is for educational purposes as part of ECE354: Image and Video Processing course.

## Acknowledgments

- [NIH Chest X-ray Dataset](https://www.innovatiana.com/en/datasets/nih-chest-x-rays) (Innovatiana, 2024) - Free for academic research, contains 112,120 chest X-rays with annotations
- OpenCV community for image processing libraries
