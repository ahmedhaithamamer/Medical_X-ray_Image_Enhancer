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
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Useful for medical images where global HE may over-amplify noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=tile_grid_size)
    return clahe.apply(img)


def apply_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Apply a simple sharpening filter using a convolution kernel.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp


def enhancement_pipeline(img: np.ndarray,
                         use_clahe: bool = False) -> dict[str, np.ndarray]:
    """
    Run a simple enhancement pipeline on a single image.

    Returns a dictionary with intermediate results that can be
    saved or visualized in the report.
    """
    med = apply_median_filter(img, ksize=3)
    if use_clahe:
        contrast = apply_clahe(med)
    else:
        contrast = apply_hist_eq(med)
    sharp = apply_sharpen(contrast)

    return {
        "original": img,
        "median": med,
        "contrast": contrast,
        "sharpen": sharp,
    }


