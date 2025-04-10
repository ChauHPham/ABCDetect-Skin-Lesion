import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_colour_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate colour score for the lesion according to ABCD rule.

    The lesion is expected to contain one or more of these colours:
      - white
      - red
      - light brown
      - dark brown
      - blue-gray
      - black
    Each distinct colour (if present above a small threshold area) counts as 1 point.

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization plots for detected colour regions.

    Returns:
        Number of distinct colours detected in the lesion.
    """
    # Ensure that there is lesion area; if not, return 0.0.
    lesion_area = np.sum(mask)
    if lesion_area == 0:
        return 0

    # Convert the input RGB image to HSV (makes it easier to set threshold ranges).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define HSV threshold ranges for each colour.
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

    # Minimum fraction of the lesion area for a colour to be considered "present"
    presence_threshold = 0.01  # e.g. 1% of lesion pixels

    detected_colours = {}

    # Process red separately (combine two ranges)
    red_mask1 = cv2.inRange(hsv, thresholds["red1"]["lower"], thresholds["red1"]["upper"])
    red_mask2 = cv2.inRange(hsv, thresholds["red2"]["lower"], thresholds["red2"]["upper"])
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # Constrain to the lesion area:
    red_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask.astype(np.uint8))
    red_fraction = np.sum(red_mask > 0) / lesion_area
    if red_fraction > presence_threshold:
        detected_colours["red"] = red_fraction

    # Process the remaining colours
    for colour in ["white", "light brown", "dark brown", "blue-gray", "black"]:
        colour_mask = cv2.inRange(hsv, thresholds[colour]["lower"], thresholds[colour]["upper"])
        # Apply lesion mask to focus on the lesion area
        colour_mask = cv2.bitwise_and(colour_mask, colour_mask, mask=mask.astype(np.uint8))
        fraction = np.sum(colour_mask > 0) / lesion_area
        if fraction > presence_threshold:
            detected_colours[colour] = fraction

    # Count the total number of different colours detected.
    colour_count = len(detected_colours)
    # If no colour is detected (which is unlikely), assume at least one exists.
    if colour_count == 0:
        colour_count = 1

    print("Detected colours: ", end="")
    for col, frac in detected_colours.items():
        print(f"{col} ({frac*100:.2f}%) ", end="")
    print()

    # Optional visualization
    if show_graph:
        # Define display colours (in RGB) for each key colour.
        display_colours = {
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "light brown": (210, 180, 140),  # a tan-like colour
            "dark brown": (101, 67, 33),
            "blue-gray": (102, 153, 204),
            "black": (0, 0, 0),
        }
        # Create a figure with one panel per detected colour plus one for the original lesion.
        num_plots = len(detected_colours) + 1
        fig, ax = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        # Ensure ax is always an array
        if num_plots == 1:
            ax = [ax]

        i = 0
        for col, frac in detected_colours.items():
            if col == "red":
                colour_mask_disp = red_mask
            else:
                colour_mask_disp = cv2.inRange(hsv, thresholds[col]["lower"], thresholds[col]["upper"])
                colour_mask_disp = cv2.bitwise_and(colour_mask_disp, colour_mask_disp, mask=mask.astype(np.uint8))
            # Create an overlay image showing detected colour regions
            overlay = image.copy()
            overlay[colour_mask_disp > 0] = display_colours[col]
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

    return colour_count
