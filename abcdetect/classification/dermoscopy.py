# abcdetect/classification/dermoscopy.py
import cv2
import numpy as np
import os
from skimage.util import view_as_windows


def compute_dermoscopic_score(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None) -> dict:
    """
    Compute dermoscopic structure score (Part D of ABCD rule).
    Visualizes detected dots, globules, and structureless areas if save_vis_path is provided.
    """
    lesion = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    dots = 0
    globules = 0
    vis_img = image.copy()
    structureless_overlay = np.zeros_like(image)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Contour area: {area:.2f}")
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if 10 <= area < 50:
            dots += 1
            if save_vis_path:
                cv2.circle(vis_img, center, radius, (0, 255, 0), 2)  # Green for dots
        elif 50 <= area < 200:
            globules += 1
            if save_vis_path:
                cv2.circle(vis_img, center, radius, (0, 0, 255), 2)  # Red for globules

    has_dots = dots > 0
    has_globules = globules > 0

    # Structureless detection via local variance
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    window_size = 9
    windows = view_as_windows(masked_gray, (window_size, window_size))
    variances = np.var(windows, axis=(2, 3))

    low_var_map = variances < 15
    percent_low_var = np.sum(low_var_map) / low_var_map.size
    has_structureless = bool(percent_low_var > 0.20)

    print(f"Structureless area %: {percent_low_var:.2f}")

    # Smooth and visualize structureless regions as semi-transparent yellow overlay
    if save_vis_path:
        stride = 1
        for i in range(low_var_map.shape[0]):
            for j in range(low_var_map.shape[1]):
                if low_var_map[i, j]:
                    top_left = (j, i)
                    bottom_right = (j + window_size, i + window_size)
                    cv2.rectangle(structureless_overlay, top_left, bottom_right, (0, 255, 255), -1)  # filled yellow
        vis_img = cv2.addWeighted(vis_img, 1.0, structureless_overlay, 0.3, 0)

    has_pigment_network = None
    has_streaks = None

    present_features = sum([
        int(has_dots),
        int(has_globules),
        0,  # pigment network
        0,  # streaks
        int(has_structureless)
    ])
    D_score = present_features * 0.5

    if save_vis_path:
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        cv2.imwrite(save_vis_path, vis_img)
        print(f"Saved blob + structureless visualization to {save_vis_path}")

    return {
        "dots": has_dots,
        "globules": has_globules,
        "D_score": D_score,
        "pigment_network": has_pigment_network,
        "streaks": has_streaks,
        "structureless_areas": has_structureless
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dermoscopy.py <image_path> [<mask_path>]")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2] if len(sys.argv) > 2 else None

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
    else:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        print("No mask provided. Using full image as lesion mask.")

    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = f"output/visuals/{image_filename}_detections.jpg"

    result = compute_dermoscopic_score(image, mask, save_vis_path=vis_path)
    print("Detected dermoscopic structures:", result)