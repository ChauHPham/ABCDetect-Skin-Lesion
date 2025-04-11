import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from typing import Tuple, Dict

def calculate_colour_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate the colour score for the lesion according to ABCD rule using relative Lab comparison.

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

    # Compute border skin reference using dilation
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(lesion_mask, kernel, iterations=1)
    skin_mask = dilated_mask - lesion_mask

    # Convert to Lab space
    image_lab = color.rgb2lab(image / 255.0)

    # Get mean Lab values for skin
    if np.sum(skin_mask) > 0:
        skin_lab = image_lab[skin_mask > 0]
        skin_mean = np.mean(skin_lab, axis=0)
    else:
        skin_mean = np.array([75, 0, 0])  # Fallback neutral light skin

    # Compute Lab difference image (ΔE from skin tone)
    delta_e = np.linalg.norm(image_lab - skin_mean, axis=2)

    # Create masks for lesion only
    lesion_lab = image_lab[lesion_mask > 0]
    delta_e_lesion = delta_e[lesion_mask > 0]

    # Define reference Lab values for ABCD colors (approximate)
    ref_colors = {
        "white": np.array([95, 0, 0]),
        "red": np.array([53, 80, 67]),
        "light brown": np.array([65, 15, 25]),
        "dark brown": np.array([40, 15, 20]),
        "blue-gray": np.array([35, 0, -30]),
        "black": np.array([15, 0, 0]),
    }

    min_percentage = 3.0
    detected_colors = {}

    for color_name, ref_lab in ref_colors.items():
        delta = np.linalg.norm(lesion_lab - ref_lab, axis=1)
        match_mask = (delta < 20)  # ΔE threshold for perceptual similarity
        percent = (np.sum(match_mask) / lesion_area) * 100
        if percent >= min_percentage:
            detected_colors[color_name] = percent

    if not detected_colors:
        dominant_color = min(ref_colors, key=lambda c: np.mean(np.linalg.norm(lesion_lab - ref_colors[c], axis=1)))
        detected_colors[dominant_color] = 100.0

    color_score = len(detected_colors)

    print(f"\nDetected {color_score} colors in the lesion:")
    for color_name, percentage in detected_colors.items():
        print(f"  - {color_name}: {percentage:.2f}%")

    if show_graph:
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, len(detected_colors) + 1, 1)
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")

        for i, (color_name, _) in enumerate(detected_colors.items(), 1):
            ref_lab = ref_colors[color_name]
            lab_patch = np.ones((50, 50, 3)) * ref_lab
            rgb_patch = color.lab2rgb(lab_patch)
            ax = fig.add_subplot(1, len(detected_colors) + 1, i + 1)
            ax.imshow(rgb_patch)
            ax.set_title(color_name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return color_score