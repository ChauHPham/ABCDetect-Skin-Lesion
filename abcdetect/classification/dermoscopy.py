import os

import cv2
import numpy as np
from skimage.util import view_as_windows


def calculate_dermoscopic_structure_score(image: np.ndarray, mask: np.ndarray) -> int:
    """Calculate dermoscopic structure score for the lesion according to ABCD rule.
    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.

    Returns:
        Dermoscopic structure score (0 to 5) based on the presence of dots, globules, and structureless areas.
    """
    return compute_dermoscopic_score(image, mask)["D_score"]


# --- Area conversion utils (assuming ~0.1 mm/pixel) ---
def pixels_to_mm2(area_px):
    return area_px * 0.01  # ≈ 0.1mm/pixel → 0.01 mm²/pixel

def mm2_to_pixels(area_mm2):
    return area_mm2 / 0.01


def compute_dermoscopic_score(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None) -> dict:
    """Compute dermoscopic structure score (Part D of ABCD rule).

    Visualizes detected dots, globules, and structureless areas if save_vis_path is provided.

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        save_vis_path: Path to save visualization image (if provided).

    Returns:
        Dictionary containing:
            - dots: Boolean indicating presence of dots.
            - globules: Boolean indicating presence of globules.
            - D_score: Dermoscopic structure score.
            - pigment_network: Boolean indicating presence of pigment network (not implemented).
            - streaks: Boolean indicating presence of streaks (not implemented).
            - structureless_areas: Boolean indicating presence of structureless areas.
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

        # Dot: 10–50 px² ≈ 0.1–0.5 mm²
        # Globule: 50–200 px² ≈ 0.5–2.0 mm²
        if mm2_to_pixels(0.1) <= area < mm2_to_pixels(0.5):
            dots += 1
            if save_vis_path:
                cv2.circle(vis_img, center, radius, (0, 255, 0), 2)  # Green for dots
        elif mm2_to_pixels(0.5) <= area < mm2_to_pixels(2.0):
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

    if save_vis_path:
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        cv2.imwrite(save_vis_path, vis_img)
        print(f"Saved blob + structureless visualization to {save_vis_path}")

    return {
        "dots": has_dots,
        "globules": has_globules,
        "D_score": present_features,
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