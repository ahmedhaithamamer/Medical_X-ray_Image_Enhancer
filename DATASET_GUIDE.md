# Working with the NIH Chest X-ray Dataset (45GB)

## Problem
The full NIH Chest X-ray dataset is **~45GB**, which is too large for a course project. You only need a **small subset (20-50 images)** to demonstrate your enhancement techniques.

## Solutions

### Option 1: Manual Download (Recommended)
1. Visit the NIH dataset page: https://www.innovatiana.com/en/datasets/nih-chest-x-rays
2. Download **20-50 sample images** manually (not the full dataset)
3. Place them in the `data/` folder
4. Run: `python -m src.main --max_images 30`

### Option 2: Use Dataset Metadata CSV
If the NIH dataset provides a CSV file with image URLs/IDs:
1. Download the metadata CSV
2. Randomly sample 20-50 rows
3. Download only those images
4. Place them in `data/` folder

### Option 3: Use Alternative Sources
You can also use chest X-ray images from:
- Other medical imaging datasets (if you have access)
- Public domain medical image repositories
- Sample images provided by your course instructor

## Recommended Workflow

1. **Start small**: Download 10-20 images first to test your code
2. **Test your pipeline**: Run `python -m src.main --max_images 10`
3. **Expand if needed**: Add more images (up to 50) for final results
4. **Document**: In your report, mention:
   - "We used a subset of 30 images from the NIH Chest X-ray dataset"
   - "The full dataset is 45GB; we selected a representative sample"

## File Structure
```
data/
  ├── image_001.png  (your sample images)
  ├── image_002.png
  └── ...
```

## Running with Limited Images

```bash
# Process only first 20 images
python -m src.main --max_images 20

# Process all images in data/ folder (if you have many)
python -m src.main
```

## For Your Report

In the **Data Preparation** section, you can write:

> "We used a subset of 30 chest X-ray images from the NIH Chest X-ray dataset 
> (Innovatiana, 2024). The full dataset contains over 100,000 images (~45GB), 
> but for this project we selected a representative sample to demonstrate our 
> enhancement techniques. Images were resized to a standard resolution and 
> converted to grayscale for processing."

This is perfectly acceptable for a course project!

