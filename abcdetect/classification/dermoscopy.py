# abcdetect/classification/dermoscopy.py
import cv2
import numpy as np


def detect_dermoscopic_structures(image_path):
    """
    Detects 4 basic dermoscopic structures from a lesion image.
    Currently implemented: dots/globules using blob detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blob detection (for dots/globules)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.7

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    has_dots_globules = len(keypoints) > 0

    return {
        "pigment_network": None,  # Not yet implemented
        "dots_globules": has_dots_globules,
        "streaks": None,          # Not yet implemented
        "structureless_areas": None  # Not yet implemented
    }


if __name__ == "__main__":
    import sys
    result = detect_dermoscopic_structures(sys.argv[1])
    print("Detected dermoscopic structures:", result)
