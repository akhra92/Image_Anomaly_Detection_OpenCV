import cv2
import numpy as np
import streamlit as st

from config import IMAGE_DIR, MASK_DIR, DARK_PIXEL_THRESHOLD, DARK_PIXEL_RATIO_THRESHOLD
from main import process_image
from utils.io import get_image_filenames, load_image_and_mask


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Defect Detection", layout="wide")
st.title("Defect Detection using Traditional CV")

# Sidebar — interactive threshold sliders
st.sidebar.header("Detection Parameters")
dark_thresh = st.sidebar.slider(
    "Dark Pixel Threshold",
    min_value=0, max_value=255, value=DARK_PIXEL_THRESHOLD,
    help="Pixels with intensity below this value are considered dark."
)
ratio_thresh = st.sidebar.slider(
    "Dark Pixel Ratio Threshold",
    min_value=0.0, max_value=1.0, value=DARK_PIXEL_RATIO_THRESHOLD,
    step=0.01,
    help="Minimum ratio of dark pixels to classify an object as defective."
)

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader("Upload the corresponding mask", type=["png", "jpg", "jpeg"])

if uploaded_image is not None and uploaded_mask is not None:
    image_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    mask_bytes = np.frombuffer(uploaded_mask.read(), dtype=np.uint8)
    mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)

    result = process_image(image, mask, dark_thresh, ratio_thresh)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input Image")
        st.image(bgr_to_rgb(image))
    with col2:
        st.subheader("Predicted Output")
        st.image(bgr_to_rgb(result))
    with col3:
        st.subheader("Segmentation Mask")
        st.image(mask)

else:
    st.info("No files uploaded — showing default samples.")

    filenames = get_image_filenames(IMAGE_DIR)

    for fname in filenames:
        image, mask = load_image_and_mask(IMAGE_DIR, MASK_DIR, fname)
        result = process_image(image, mask, dark_thresh, ratio_thresh)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input Image")
            st.image(bgr_to_rgb(image))
        with col2:
            st.subheader("Predicted Output")
            st.image(bgr_to_rgb(result))
        with col3:
            st.subheader("Segmentation Mask")
            st.image(mask)
