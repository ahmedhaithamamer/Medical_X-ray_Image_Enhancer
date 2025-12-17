import glob
import os
from typing import List

import cv2
import numpy as np


def list_images(input_dir: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    
    #List image files in a directory with given extensions
    paths: list[str] = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        paths.extend(glob.glob(pattern))
    paths.sort()
    return paths


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    #Save an image to disk, creating directories if needed

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


