# Medical X-ray Image Enhancement

**ECE354 Project – Image and Video Processing**

A comprehensive implementation of medical X-ray image enhancement techniques using filtering and contrast methods, with an interactive web-based viewer for real-time comparison.

## Overview

This project implements six image enhancement techniques for medical X-ray images:
- **Noise Reduction**: Median Filter, Bilateral Filter
- **Contrast Enhancement**: Histogram Equalization, CLAHE
- **Detail Enhancement**: Sharpening, Unsharp Masking

Evaluation is performed using MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio) metrics.

## Features

- ✅ Six enhancement techniques organized by category
- ✅ Batch processing pipeline for multiple images
- ✅ Interactive Streamlit web viewer with image upload
- ✅ Real-time metrics calculation and visualization
- ✅ Before/after comparison with interactive slider
- ✅ Download all enhanced images as ZIP file

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Medical_X-ray_Image_Enhancement

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Batch Processing:**
```bash
python -m src.main
```

**Interactive Viewer:**
```bash
streamlit run src/streamlit_app.py
```

## Project Structure

```
Medical_X-ray_Image_Enhancement/
├── data/              # Input X-ray images
├── results/           # Processed images (organized by image name)
├── src/               # Source code
│   ├── main.py        # Batch processing
│   ├── enhancement.py # Enhancement functions
│   ├── metrics.py     # MSE/PSNR calculation
│   ├── io_utils.py    # Image I/O utilities
│   ├── streamlit_app.py # Interactive viewer
│   └── calculate_stats.py # Statistics calculator
├── report/            # LaTeX report
└── requirements.txt   # Dependencies
```

## Dataset

The project uses a subset of 30-50 chest X-ray images from the [NIH Chest X-ray Dataset](https://www.innovatiana.com/en/datasets/nih-chest-x-rays) (Innovatiana, 2024). The full dataset contains over 100,000 images (~45GB), but a representative sample is sufficient for demonstration.

> "We used a subset of 30 chest X-ray images from the NIH Chest X-ray dataset 
> (Innovatiana, 2024). The full dataset contains over 100,000 images (~45GB), 
> but for this project we selected a representative sample to demonstrate our 
> enhancement techniques. Images were resized to a standard resolution and 
> converted to grayscale for processing."

## Enhancement Techniques

### Noise Reduction
- **Median Filter**: 3×3 kernel, effective for impulse noise
- **Bilateral Filter**: Edge-preserving with spatial and intensity domain filtering

### Contrast Enhancement
- **Histogram Equalization**: Global contrast enhancement
- **CLAHE**: Adaptive local contrast with clipping limit (clip=2.0, tiles=8×8)

### Detail Enhancement
- **Sharpening**: 3×3 convolution kernel
- **Unsharp Masking**: Gaussian blur (σ=1.0) with strength factor 1.5

## Evaluation Metrics

- **MSE** (Mean Squared Error): Lower is better
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (in dB)

**Expected Ranges:**
- Noise Reduction: MSE < 50, PSNR > 30 dB
- Contrast Enhancement: MSE 200-2000, PSNR 15-25 dB
- Detail Enhancement: MSE 300-800, PSNR 19-22 dB

## Interactive Viewer

The Streamlit app provides:
- Image upload (standalone operation)
- Before/after slider comparison
- Side-by-side comparison view
- Grid view of all techniques
- Real-time metrics display
- Download enhanced images as ZIP

Launch with: `streamlit run src/streamlit_app.py`

## Results

Average metrics across 50 images:

| Category | Technique | Avg MSE | Avg PSNR (dB) |
|----------|-----------|---------|---------------|
| Noise Reduction | Median Filter | 12.50 | 37.73 |
| Noise Reduction | Bilateral Filter | 35.34 | 32.79 |
| Contrast Enhancement | Histogram Equalization | 1,196.96 | 20.57 |
| Contrast Enhancement | CLAHE | 426.05 | 22.01 |
| Detail Enhancement | Sharpening | 716.98 | 19.65 |
| Detail Enhancement | Unsharp Masking | 532.18 | 20.99 |

## Dependencies

- `opencv-python` - Image processing
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `streamlit` - Web application
- `pillow` - Image handling
- `pandas` - Data manipulation

## Code Structure

- `src/enhancement.py` - Core enhancement functions
- `src/metrics.py` - MSE and PSNR calculation
- `src/io_utils.py` - Image I/O utilities
- `src/main.py` - Batch processing pipeline
- `src/streamlit_app.py` - Interactive web viewer
- `src/calculate_stats.py` - Statistics calculator

## Report

The LaTeX report is located in `report/main.tex`. To compile:
- Use [Overleaf](https://www.overleaf.com) (online, no installation)
- Or install LaTeX locally and run `pdflatex main.tex`

## License

This project is licensed under the MIT License with academic use restrictions. See the [LICENSE](LICENSE) file for details.

**Academic Use Only**: This code and documentation are provided for educational and research purposes. Commercial use is not permitted without 
explicit permission.

**Dataset License**: The NIH Chest X-ray Dataset is free for academic research under terms specified by the National Institutes of Health (NIH). See the [dataset page](https://www.innovatiana.com/en/datasets/nih-chest-x-rays) for full license details.

## Acknowledgments

- [NIH Chest X-ray Dataset](https://www.innovatiana.com/en/datasets/nih-chest-x-rays) (Innovatiana, 2024) - Free for academic research, contains 112,120 chest X-rays with annotations
- OpenCV community for image processing libraries
- Streamlit for the web application framework

## Contact

For questions or issues related to this course project, please contact the course instructor or refer to the course materials.

---

**Note**: This project is part of an academic course and is intended for educational purposes only.
