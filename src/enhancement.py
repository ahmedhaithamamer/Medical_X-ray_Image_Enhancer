import cv2
import numpy as np


def apply_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply median filtering to reduce impulse noise while preserving edges.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    ksize : int
        Kernel size (must be odd and > 1).
    """
    return cv2.medianBlur(img, ksize)


def apply_hist_eq(img: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization to enhance contrast.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    """
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
    """
    Apply unsharp masking for detail enhancement.
    
    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    sigma : float
        Standard deviation for Gaussian blur.
    strength : float
        Strength of the sharpening effect.
    threshold : int
        Threshold for edge detection (0 = no threshold).
    
    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Create sharpened version
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        mask = cv2.absdiff(img, blurred) > threshold
        result = np.where(mask, sharpened, img)
        return result.astype(np.uint8)
    
    return sharpened


def apply_sharpen(img: np.ndarray) -> np.ndarray:
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp


def enhancement_pipeline(img: np.ndarray,
                         use_clahe: bool = False,
                         use_bilateral: bool = True,
                         use_unsharp: bool = True) -> dict[str, np.ndarray]:
    """
    Run a comprehensive enhancement pipeline on a single image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    use_clahe : bool
        Use CLAHE instead of global histogram equalization.
    use_bilateral : bool
        Include bilateral filtering results.
    use_unsharp : bool
        Include unsharp masking results.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with all enhancement results.
    """
    outputs = {
        "original": img,
    }
    
    # Noise reduction techniques
    outputs["median"] = apply_median_filter(img, ksize=3)
    
    if use_bilateral:
        outputs["bilateral"] = apply_bilateral_filter(img, d=9, sigma_color=75.0, sigma_space=75.0)
    
    # Use median-filtered image for contrast enhancement (best balance)
    denoised = outputs["median"]
    
    # Contrast enhancement - always compute both for comparison
    outputs["contrast"] = apply_hist_eq(denoised)
    outputs["clahe"] = apply_clahe(denoised)
    
    # Use the selected contrast method for detail enhancement
    contrast_base = outputs["clahe"] if use_clahe else outputs["contrast"]
    
    # Detail enhancement
    outputs["sharpen"] = apply_sharpen(contrast_base)
    
    if use_unsharp:
        outputs["unsharp"] = apply_unsharp_masking(contrast_base, sigma=1.0, strength=1.5)
    
    return outputs


