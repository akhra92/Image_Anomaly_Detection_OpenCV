import cv2


def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image.

    Args:
        image: BGR image to annotate.
        detections: list of dicts with 'status', 'color', 'bbox' keys.

    Returns:
        Annotated copy of the image.
    """
    result = image.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = det["color"]
        status = det["status"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, status, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result
