import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from typing import Tuple, Dict

REF_COLOURS: Dict[str, np.ndarray] = {
    "white":       np.array([95,  0,   0]),
    "red":         np.array([53, 80,  67]),
    "light brown": np.array([65, 15,  25]),
    "dark brown":  np.array([40, 15,  20]),
    "blue-gray":   np.array([35,  0, -30]),
    "black":       np.array([15,  0,   0]),
}

COLOUR_NAMES = list(REF_COLOURS.keys())
PROTOTYPES   = np.stack(list(REF_COLOURS.values()))        # shape (6, 3)

# ──────────────────────────────────────────────────────────────────────────────
def calculate_colour_score(
    image: np.ndarray,
    mask:  np.ndarray,
    *,
    show_graph: bool = False,
) -> int:
    """
    Count distinct ABCD colours inside a *dilated* lesion mask using **RGB**.
    Returns an integer 1‑6 (ABCD colour score).
    """
    # --- 1.  Make sure image & mask sizes match ------------------------------
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    lesion_mask = (mask > 0).astype(np.uint8)
    if lesion_mask.sum() == 0:
        return 1

    # --- 2.  Dilate the mask to include immediate border skin ----------------
    kernel        = np.ones((15, 15), np.uint8)
    dilated_mask  = cv2.dilate(lesion_mask, kernel, iterations=1).astype(bool)
    skin_mask     = dilated_mask & (lesion_mask == 0)

    # --- 3.  Extract RGB pixels ---------------------------------------------
    lesion_pixels = image[lesion_mask.astype(bool)].astype(np.float32)
    skin_pixels   = image[skin_mask].astype(np.float32)

    if lesion_pixels.size == 0 or skin_pixels.size == 0:
        return 1

    # --- 4.  Normalize RGB pixels relative to average skin tone --------------
    skin_mean     = skin_pixels.mean(axis=0)               # (3,)
    lesion_rel    = lesion_pixels - skin_mean              # (N, 3)

    # --- 5.  Shift prototypes relative to actual skin tone -------------------
    prototype_shifted = PROTOTYPES + (skin_mean - 128)     # shape (6, 3)

    # --- 6.  Vectorised nearest‑prototype classification ---------------------
    dists  = np.linalg.norm(lesion_pixels[:, None, :] - prototype_shifted[None, :, :], axis=2)
    labels = dists.argmin(axis=1)

    # --- 7.  Count occurrences -----------------------------------------------
    counts = np.bincount(labels, minlength=len(COLOUR_NAMES))

    # --- 8.  Optional histogram ----------------------------------------------
    if show_graph:
        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        fig.suptitle("Colour Detection Pipeline (Relative RGB, Skin-Aware)", fontsize=16)

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].bar(COLOUR_NAMES, counts)
        axes[1].set_ylabel("Pixel count")
        axes[1].set_title("ABCD colour distribution inside dilated mask")
        axes[1].set_xticks(range(len(COLOUR_NAMES)))
        axes[1].set_xticklabels(COLOUR_NAMES, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    # --- 9.  Return distinct‑colour count (≥1) -------------------------------
    distinct = np.count_nonzero(counts)
    return max(1, distinct)
