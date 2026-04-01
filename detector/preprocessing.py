import cv2
import numpy as np


def binarize_mask(mask):
    """Convert mask to binary, handling both binary and grayscale inputs."""
    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()).issubset({0, 255}):
        return mask.copy()
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def extract_contours(binary_mask):
    """Find external contours from a binary mask."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
