import os
import cv2


def get_image_filenames(directory, extension='.png'):
    """Return sorted list of filenames with the given extension."""
    return sorted(
        f for f in os.listdir(directory) if f.endswith(extension)
    )


def load_image_and_mask(image_dir, mask_dir, filename):
    """Load a color image and its corresponding grayscale mask."""
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    return image, mask


def save_result(output_dir, filename, image):
    """Save result image to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    print(f"Saved result to {output_path}")
