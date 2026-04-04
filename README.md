# Defect Detection using Traditional Computer Vision

A lightweight defect detection pipeline built with OpenCV. It analyzes objects in images using pre-generated segmentation masks and classifies each object as **Good** or **Defective** based on dark pixel intensity analysis.

**[Live Demo on Streamlit Cloud](https://imageanomalydetectionopencv.streamlit.app/)** <!-- TODO: replace # with your Streamlit Cloud URL -->

## How It Works

1. **Load** a color image and its corresponding binary segmentation mask.
2. **Binarize** the mask and extract object contours.
3. **Classify** each object by measuring the ratio of dark pixels within its contour region. If the ratio exceeds a configurable threshold, the object is marked as **Defective**; otherwise, it is **Good**.
4. **Annotate** the image with color-coded bounding boxes and labels (green for Good, red for Defective).
5. **Save** the annotated result.

## Sample Results

| Input Image | Predicted Output | Segmentation Mask |
|:-----------:|:----------------:|:-----------------:|
| ![Input](samples/images/0.png) | ![Output](samples/outputs/0.png) | ![Mask](samples/masks/0.png) |
| ![Input](samples/images/10.png) | ![Output](samples/outputs/10.png) | ![Mask](samples/masks/10.png) |
| ![Input](samples/images/60.png) | ![Output](samples/outputs/60.png) | ![Mask](samples/masks/60.png) |

## Project Structure

```
Traditional_CV/
├── app.py                   # Streamlit web app
├── main.py                  # CLI entry point — orchestrates the pipeline
├── config.py                # Thresholds and directory paths
├── requirements.txt         # Python dependencies
├── detector/
│   ├── preprocessing.py     # Mask binarization & contour extraction
│   ├── classifier.py        # Dark-pixel defect classification
│   └── visualizer.py        # Bounding box & label drawing
├── utils/
│   └── io.py                # Image/mask loading & result saving
└── samples/
    ├── images/              # Sample input images
    ├── masks/               # Sample segmentation masks
    └── outputs/             # Sample predicted outputs
```

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App

Run the interactive web app:

```bash
streamlit run app.py
```

Upload your own image and mask, or view the default sample results. Use the sidebar sliders to tune detection thresholds in real time.

### CLI Pipeline

1. Place your input images in the `samples/images/` directory.
2. Place the corresponding segmentation masks (same filenames) in the `samples/masks/` directory.
3. Run the pipeline:

```bash
python main.py
```

4. Annotated results will be saved to the `samples/outputs/` directory.

## Configuration

All parameters are defined in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_DIR` | `samples/images` | Directory containing input images |
| `MASK_DIR` | `samples/masks` | Directory containing segmentation masks |
| `OUTPUT_DIR` | `samples/outputs` | Directory for saving annotated results |
| `DARK_PIXEL_THRESHOLD` | `70` | Pixel intensity below which a pixel is considered dark |
| `DARK_PIXEL_RATIO_THRESHOLD` | `0.03` | Minimum dark pixel ratio to classify an object as defective |

Both thresholds can also be adjusted interactively via the sidebar sliders in the Streamlit app.
