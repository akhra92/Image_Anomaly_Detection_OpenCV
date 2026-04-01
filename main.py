import cv2

from config import IMAGE_DIR, MASK_DIR, OUTPUT_DIR
from utils.io import get_image_filenames, load_image_and_mask, save_result
from detector.preprocessing import binarize_mask, extract_contours
from detector.classifier import classify_contour
from detector.visualizer import draw_detections


def process_image(image, mask):
    """Run defect detection on a single image-mask pair."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_mask = binarize_mask(mask)
    contours = extract_contours(binary_mask)

    detections = [
        classify_contour(contour, gray, mask.shape)
        for contour in contours
    ]

    return draw_detections(image, detections)


def main():
    filenames = get_image_filenames(IMAGE_DIR)

    for filename in filenames:
        image, mask = load_image_and_mask(IMAGE_DIR, MASK_DIR, filename)
        result = process_image(image, mask)
        save_result(OUTPUT_DIR, filename, result)


if __name__ == '__main__':
    main()
