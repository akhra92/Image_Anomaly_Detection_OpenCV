import cv2
import numpy as np

from config import DARK_PIXEL_THRESHOLD, DARK_PIXEL_RATIO_THRESHOLD


def classify_contour(contour, gray_image, mask_shape):
    """Classify a single contour as 'Good' or 'Defective' based on dark pixel ratio.

    Returns:
        dict with keys: status, color, bbox (x, y, w, h)
    """
    x, y, w, h = cv2.boundingRect(contour)

    # Create a filled mask for the object region
    object_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(object_mask, [contour], -1, 255, -1)

    # Analyze pixel intensities within the contour
    object_pixels = gray_image[object_mask == 255]
    dark_pixels = np.sum(object_pixels < DARK_PIXEL_THRESHOLD)
    total_pixels = object_pixels.size
    dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

    if dark_ratio > DARK_PIXEL_RATIO_THRESHOLD:
        status = "Defective"
        color = (0, 0, 255)  # Red
    else:
        status = "Good"
        color = (0, 255, 0)  # Green

    return {"status": status, "color": color, "bbox": (x, y, w, h)}
