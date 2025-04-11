import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from typing import Tuple, Dict

def calculate_colour_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate the colour score for the lesion according to ABCD rule using Lab ΔE distance.

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization graph for detected colors.

    Returns:
        Number of distinct colors detected in the lesion (1-6).
    """
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    lesion_mask = (mask > 0).astype(np.uint8)
    lesion_area = np.sum(lesion_mask)
    if lesion_area == 0:
        return 1

    # Compute skin reference via dilation
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(lesion_mask, kernel, iterations=1)
    skin_mask = dilated_mask - lesion_mask

    image_lab = color.rgb2lab(image / 255.0)

    if np.sum(skin_mask) > 0:
        skin_lab = image_lab[skin_mask > 0]
        skin_mean = np.mean(skin_lab, axis=0)
    else:
        skin_mean = np.array([75, 0, 0])

    # Reference Lab values (estimated from color schematic)
    ref_colors = {
        "white":       np.array([95, 0, 0]),
        "red":         np.array([53, 75, 50]),
        "light brown": np.array([70, 15, 25]),
        "dark brown":  np.array([40, 10, 15]),
        "blue-gray":   np.array([40, 0, -35]),
        "black":       np.array([10, 0, 0]),
    }

    # ΔE calculation
    lesion_lab = image_lab[lesion_mask > 0]
    min_percentage = 0.5
    delta_thresh = 25
    detected_colors = {}

    for color_name, ref_lab in ref_colors.items():
        delta_e = np.linalg.norm(lesion_lab - ref_lab, axis=1)

        if color_name == "white":
            # Only count as white if L* significantly higher than skin
            is_lighter = lesion_lab[:, 0] > (skin_mean[0] + 10)
            match_mask = (delta_e < delta_thresh) & is_lighter
        else:
            match_mask = delta_e < delta_thresh

        percent = (np.sum(match_mask) / lesion_area) * 100
        if percent >= min_percentage:
            detected_colors[color_name] = percent

    if not detected_colors:
        dominant_color = min(ref_colors, key=lambda c: np.mean(np.linalg.norm(lesion_lab - ref_colors[c], axis=1)))
        detected_colors[dominant_color] = 100.0

    color_score = len(detected_colors)

    print(f"\nDetected {color_score} colors in the lesion:")
    for cname, pct in detected_colors.items():
        print(f"  - {cname}: {pct:.2f}%")

    if show_graph:
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, len(detected_colors) + 1, 1)
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")

        for i, cname in enumerate(detected_colors, 1):
            patch_lab = np.ones((50, 50, 3)) * ref_colors[cname]
            rgb_patch = color.lab2rgb(patch_lab)
            ax = fig.add_subplot(1, len(detected_colors) + 1, i + 1)
            ax.imshow(rgb_patch)
            ax.set_title(cname)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return color_score
