import cv2
import numpy as np

from config import MIN_CONTOUR_AREA


def binarize_mask(mask):
    """Convert mask to binary, handling both binary and grayscale inputs."""
    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()).issubset({0, 255}):
        return mask.copy()
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def extract_contours(binary_mask, min_area=MIN_CONTOUR_AREA):
    """Find external contours from a binary mask, filtering out small noise."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]
