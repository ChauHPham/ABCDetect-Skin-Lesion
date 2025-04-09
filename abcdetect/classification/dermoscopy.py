# abcdetect/classification/dermoscopy.py
import cv2
import numpy as np


def detect_dermoscopic_structures(image_path):
    """
    Detects basic dermoscopic structures from a lesion image.
    Currently implemented: dots and globules using blob detection and thresholded by area.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to isolate darker regions (potential dots/globules)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = 0
    globules = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 <= area < 50:
            dots += 1
        elif 50 <= area < 200:
            globules += 1

    has_dots = dots > 0
    has_globules = globules > 0

    # Each feature contributes 0.5 if present
    d_score = 0.5 * int(has_dots) + 0.5 * int(has_globules)

    return {
        "dots": has_dots,
        "globules": has_globules,
        "D_score": d_score,
        "pigment_network": None,
        "streaks": None,
        "structureless_areas": None
    }


if __name__ == "__main__":
    import sys
    result = detect_dermoscopic_structures(sys.argv[1])
    print("Detected dermoscopic structures:", result)
