import cv2
import numpy as np


def apply_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def apply_hist_eq(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def apply_bilateral_filter(img: np.ndarray, d: int = 9,
                           sigma_color: float = 75.0,
                           sigma_space: float = 75.0) -> np.ndarray:
    """
    Apply bilateral filtering to reduce noise while preserving edges.
    
    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    d : int
        Diameter of pixel neighborhood.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_unsharp_masking(img: np.ndarray,
                          sigma: float = 1.0,
                          strength: float = 1.5,
                          threshold: int = 0) -> np.ndarray:
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # sharpened = original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened


def apply_sharpen(img: np.ndarray) -> np.ndarray:

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp


def best_pipeline(img: np.ndarray, 
                  denoise_method: str = "median",
                  contrast_method: str = "clahe",
                  sharpen_method: str = "unsharp") -> np.ndarray:
    # Step 1: Denoise (Clean)
    if denoise_method == "bilateral":
        denoised = apply_bilateral_filter(img, d=9, sigma_color=75.0, sigma_space=75.0)
    else:  # median
        denoised = apply_median_filter(img, ksize=3)
    
    # Step 2: Contrast (Enhance)
    if contrast_method == "clahe":
        contrasted = apply_clahe(denoised, clip_limit=2.0, tile_grid_size=(8, 8))
    else:  # hist_eq
        contrasted = apply_hist_eq(denoised)
    
    # Step 3: Sharpen (Detail)
    if sharpen_method == "unsharp":
        sharpened = apply_unsharp_masking(contrasted, sigma=1.0, strength=1.5)
    else:  # sharpen
        sharpened = apply_sharpen(contrasted)
    
    return sharpened


def enhancement_pipeline(img: np.ndarray,
                         use_clahe: bool = False,
                         use_bilateral: bool = True,
                         use_unsharp: bool = True,
                         include_best_pipeline: bool = True) -> dict[str, np.ndarray]:
    outputs = {
        "original": img,
    }
    
    # ========================================================================
    # Step 1: Denoise (Clean) - Remove noise while preserving edges
    # ========================================================================
    outputs["median"] = apply_median_filter(img, ksize=3)
    
    if use_bilateral:
        outputs["bilateral"] = apply_bilateral_filter(img, d=9, sigma_color=75.0, sigma_space=75.0)
    
    # Use median-filtered image as base for contrast enhancement
    # This ensures noise is removed before enhancing contrast
    denoised = outputs["median"]
    
    # ========================================================================
    # Step 2: Contrast (Enhance) - Improve visibility of structures
    # ========================================================================
    # Always compute both contrast methods for comparison
    outputs["contrast"] = apply_hist_eq(denoised)
    outputs["clahe"] = apply_clahe(denoised)
    
    # Select contrast-enhanced image as base for detail enhancement
    # This ensures good contrast before sharpening
    contrast_base = outputs["clahe"] if use_clahe else outputs["contrast"]
    
    # ========================================================================
    # Step 3: Sharpen (Detail) - Enhance fine details and edges
    # ========================================================================
    outputs["sharpen"] = apply_sharpen(contrast_base)
    
    if use_unsharp:
        outputs["unsharp"] = apply_unsharp_masking(contrast_base, sigma=1.0, strength=1.5)
    
    # ========================================================================
    # Best Pipeline: Optimal sequential combination
    # ========================================================================
    # Applies all three steps sequentially: Denoise → Contrast → Sharpen
    # Best combination: Median Filter → CLAHE → Unsharp Masking
    # This produces superior results compared to independent application
    if include_best_pipeline:
        best_denoise = "median"  # Always use median for best pipeline
        best_contrast = "clahe" if use_clahe else "hist_eq"
        best_sharpen = "unsharp" if use_unsharp else "sharpen"
        
        outputs["best_pipeline"] = best_pipeline(
            img,
            denoise_method=best_denoise,
            contrast_method=best_contrast,
            sharpen_method=best_sharpen
        )
    
    return outputs


