"""
color.py

Color classification module for ABCDetect.
This module defines a function to compute the color score according to the ABCD rule:
Each of the following six colors is considered:
  - white
  - red
  - light brown
  - dark brown
  - blue-gray
  - black

Each color that is present (above a minimum fraction threshold)
adds 1 point to the raw score; then the raw score is multiplied by 0.5 for the TDS contribution.
The raw color score is expected to be in the range 1 to 6.
Note: For white, the clinical definition requires the lesion area to be lighter than the surrounding skin.
Without a normal-skin reference here, we use a high-brightness threshold.
"""

from typing import Any
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_color_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> float:
    """
    Calculate color score for the lesion according to ABCD rule.

    The lesion is expected to contain one or more of these colors:
      - white
      - red
      - light brown
      - dark brown
      - blue-gray
      - black
    Each distinct color (if present above a small threshold area) counts as 1 point.
    The TDS contribution is computed as: color_score = (number of present colors) * 0.5

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization plots for detected color regions.

    Returns:
        TDS contribution score for color classification.
    """
    # Ensure that there is lesion area; if not, return 0.0.
    lesion_area = np.sum(mask)
    if lesion_area == 0:
        return 0.0

    # Convert the input RGB image to HSV (makes it easier to set threshold ranges).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define HSV threshold ranges for each color.
    # Note: OpenCV’s HSV uses H:0-180, S:0-255, V:0-255.
    thresholds = {
        "white": {"lower": np.array([0, 0, 200]), "upper": np.array([180, 40, 255])},
        # For red: due to hue wraparound, we use two ranges and later combine them.
        "red1": {"lower": np.array([0, 100, 50]), "upper": np.array([10, 255, 255])},
        "red2": {"lower": np.array([170, 100, 50]), "upper": np.array([180, 255, 255])},
        "light brown": {"lower": np.array([10, 50, 150]), "upper": np.array([30, 200, 230])},
        "dark brown": {"lower": np.array([10, 100, 50]), "upper": np.array([30, 255, 150])},
        "blue-gray": {"lower": np.array([100, 20, 50]), "upper": np.array([130, 100, 200])},
        "black": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},
    }

    # Minimum fraction of the lesion area for a color to be considered "present"
    presence_threshold = 0.01  # e.g. 1% of lesion pixels

    detected_colors = {}

    # Process red separately (combine two ranges)
    red_mask1 = cv2.inRange(hsv, thresholds["red1"]["lower"], thresholds["red1"]["upper"])
    red_mask2 = cv2.inRange(hsv, thresholds["red2"]["lower"], thresholds["red2"]["upper"])
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # Constrain to the lesion area:
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask.astype(np.uint8))
    red_fraction = np.sum(red_mask > 0) / lesion_area
    if red_fraction > presence_threshold:
        detected_colors["red"] = red_fraction

    # Process the remaining colors
    for color in ["white", "light brown", "dark brown", "blue-gray", "black"]:
        color_mask = cv2.inRange(hsv, thresholds[color]["lower"], thresholds[color]["upper"])
        # Apply lesion mask to focus on the lesion area
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask.astype(np.uint8))
        fraction = np.sum(color_mask > 0) / lesion_area
        if fraction > presence_threshold:
            detected_colors[color] = fraction

    # Count the total number of different colors detected.
    color_count = len(detected_colors)
    # If no color is detected (which is unlikely), assume at least one exists.
    if color_count == 0:
        color_count = 1

    # Apply the factor for TDS contribution.
    factor = 0.5
    tds_contribution = color_count * factor

    # Optional visualization
    if show_graph:
        # Define display colors (in RGB) for each key color.
        display_colors = {
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "light brown": (210, 180, 140),  # a tan-like color
            "dark brown": (101, 67, 33),
            "blue-gray": (102, 153, 204),
            "black": (0, 0, 0),
        }
        # Create a figure with one panel per detected color plus one for the original lesion.
        num_plots = len(detected_colors) + 1
        fig, ax = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        # Ensure ax is always an array
        if num_plots == 1:
            ax = [ax]

        i = 0
        for col, frac in detected_colors.items():
            if col == "red":
                color_mask_disp = red_mask
            else:
                color_mask_disp = cv2.inRange(hsv, thresholds[col]["lower"], thresholds[col]["upper"])
                color_mask_disp = cv2.bitwise_and(color_mask_disp, color_mask_disp, mask=mask.astype(np.uint8))
            # Create an overlay image showing detected color regions
            overlay = image.copy()
            overlay[color_mask_disp > 0] = display_colors[col]
            ax[i].imshow(overlay)
            ax[i].set_title(f"{col}\nFraction: {frac:.2f}")
            ax[i].axis("off")
            i += 1

        # Show original lesion overlay (with lesion mask)
        lesion_overlay = image.copy()
        # Create a simple mask overlay in grayscale (scaled to 255)
        lesion_mask_rgb = np.stack([mask * 255, mask * 255, mask * 255], axis=-1)
        lesion_overlay = cv2.addWeighted(lesion_overlay, 0.7, lesion_mask_rgb, 0.3, 0)
        ax[i].imshow(lesion_overlay)
        ax[i].set_title("Original Lesion")
        ax[i].axis("off")
        plt.tight_layout()
        plt.show()

    return tds_contribution


# If run as a script, execute a simple demo (this demo expects a test image and mask)
if __name__ == '__main__':
    # For demonstration, create dummy data.
    # Replace these with actual image and mask paths (or load your sample lesion image and mask)
    import sys

    # Create a dummy RGB image (e.g. 256x256 with random colors) and a circular mask.
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 80, 1, -1)  # binary mask with a circle

    # Calculate color score.
    score = calculate_color_score(img, mask, show_graph=True)
    print(f"Calculated TDS color contribution score: {score:.2f}")

    # Exit gracefully.
    sys.exit(0)
