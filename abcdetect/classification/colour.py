import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from typing import Tuple, Dict, Optional
import csv
import os

# --- Reference LAB values for ABCD color categories --------------------------
REF_COLOURS: Dict[str, np.ndarray] = {
    "white":       np.array([95,  0,   0]),
    "red":         np.array([53, 80,  67]),
    "light brown": np.array([65, 15,  25]),
    "dark brown":  np.array([40, 15,  20]),
    "blue-gray":   np.array([35,  0, -30]),
    "black":       np.array([15,  0,   0]),
}

# --- RGB colors used only for histogram bar display --------------------------
VISUAL_RGB = {
    "white":       "#ffffff",
    "red":         "#ff4d4d",
    "light brown": "#c68642",
    "dark brown":  "#5c4033",
    "blue-gray":   "#6e7f80",
    "black":       "#000000",
}

COLOUR_NAMES = list(REF_COLOURS.keys())                   # Ordered list of color labels
PROTOTYPES   = np.stack(list(REF_COLOURS.values()))       # Shape (6, 3)

# ──────────────────────────────────────────────────────────────────────────────
def calculate_colour_score(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    show_graph: bool = False,
    csv_path: Optional[str] = None,
    min_percentage: float = 3.0,
) -> int:
    """
    Count distinct ABCD colours inside a *dilated* lesion mask using **relative RGB**.
    Shows a color-coded bar chart and returns integer 1‑6 (ABCD colour score).
    """
    # --- 1. Resize image if needed to match mask ----------------------------------
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    # --- 2. Create lesion and skin masks -----------------------------------------
    lesion_mask = (mask > 0).astype(np.uint8)
    if lesion_mask.sum() == 0:
        return 1  # return early if no lesion present

    kernel        = np.ones((15, 15), np.uint8)
    dilated_mask  = cv2.dilate(lesion_mask, kernel, iterations=1).astype(bool)
    skin_mask     = dilated_mask & (lesion_mask == 0)

    # --- 3. Extract RGB pixels from lesion and skin areas -------------------------
    lesion_pixels = image[lesion_mask.astype(bool)].astype(np.float32)
    skin_pixels   = image[skin_mask].astype(np.float32)
    if lesion_pixels.size == 0 or skin_pixels.size == 0:
        return 1  # return early if missing pixels

    # --- 4. Normalize lesion pixels relative to skin tone -------------------------
    skin_mean = skin_pixels.mean(axis=0)
    prototype_shifted = PROTOTYPES + (skin_mean - 128)

    # --- 5. Classify lesion pixels by closest prototype ---------------------------
    dists  = np.linalg.norm(lesion_pixels[:, None, :] - prototype_shifted[None, :, :], axis=2)
    labels = dists.argmin(axis=1)
    counts = np.bincount(labels, minlength=len(COLOUR_NAMES))

    # --- 6. Compute pixel percentages per class -----------------------------------
    total_pixels = counts.sum()
    percentages = (counts / total_pixels) * 100
    detected_colors = {
        COLOUR_NAMES[i]: percentages[i]
        for i in range(len(COLOUR_NAMES)) if percentages[i] >= min_percentage
    }

    # --- 7. Console output --------------------------------------------------------
    print("\nDetected ABCD Colors (≥ {:.1f}%):".format(min_percentage))
    for name, pct in detected_colors.items():
        print(f"  - {name:12}: {pct:.2f}%")

    # --- 8. Optional CSV export ---------------------------------------------------
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Color", "Percentage"])
            for name, pct in detected_colors.items():
                writer.writerow([name, f"{pct:.2f}"])
        print(f"Color breakdown saved to: {csv_path}")

    # --- 9. Plot color histogram --------------------------------------------------
    if show_graph:
        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        fig.suptitle("Colour Detection Pipeline (Relative RGB)", fontsize=16)

        # Original input image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Color-coded bar chart with labels
        bars = axes[1].bar(
            COLOUR_NAMES,
            percentages,
            color=[VISUAL_RGB[name] for name in COLOUR_NAMES]
        )

        for bar, pct in zip(bars, percentages):
            if pct >= min_percentage:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2, height + 1,
                             f"{pct:.1f}%", ha='center', va='bottom', fontsize=10)

        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("ABCD Colour Distribution (Dilated Mask)")
        axes[1].set_xticks(range(len(COLOUR_NAMES)))
        axes[1].set_xticklabels(COLOUR_NAMES, rotation=45, ha="right")
        axes[1].set_ylim(0, max(100, np.max(percentages) + 10))

        plt.tight_layout()
        plt.show()

    # --- 10. Return number of distinct detected colors (1–6) ----------------------
    distinct = len(detected_colors)
    return max(1, distinct)
