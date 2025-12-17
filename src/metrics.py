import numpy as np


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Mean Squared Error between two images.
    Images are expected to have the same shape.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return float(np.mean((img1 - img2) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) between two images.
    If the MSE is zero, returns +inf.
    """
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((max_pixel ** 2) / m))


