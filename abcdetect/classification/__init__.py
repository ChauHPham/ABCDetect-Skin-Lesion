from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .asymmetry import calculate_asymmetry_score
from .border import calculate_border_score
from .colour import calculate_colour_score
from .dermoscopy import calculate_dermoscopic_structure_score

__all__ = ["classify_lesion", "analyze_abcd_features", "visualize_abcd_analysis"]


def classify_lesion(tds: float) -> str:
    """Classify the lesion based on the Total Dermascopic Score (TDS).

    Args:
        tds: Total Dermoscopic Score (TDS) calculated from ABCD features.

    Returns:
        Classification of the lesion as "Benign", "Suspicious", or "Malignant".
    """
    if tds < 4.75:
        return "Benign"
    elif tds <= 5.45:
        return "Suspicious"
    else:
        return "Malignant"


def analyze_abcd_features(
    image_path: Path, mask_path: Path, *, show_graph: bool = False
) -> dict[str, float | int]:
    """Analyze the ABCD features of a given lesion image and its mask.

    Args:
        image_path: Path to the lesion image.
        mask_path: Path to the binary mask of the lesion.
        show_graph: Whether to display the analysis graphically.

    Returns:
        A dictionary containing the scores for asymmetry, border, colour, dermoscopic structure,
        and the Total Dermoscopy Score (TDS).
    """

    # Load image and mask
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Ensure binary mask
    mask = (mask > 0).astype(np.float32)

    # Calculate ABCD features
    asymmetry_score = calculate_asymmetry_score(image, mask, show_graph=show_graph)
    border_score = calculate_border_score(image, mask, show_graph=show_graph)
    colour_score = calculate_colour_score(image, mask, show_graph=show_graph)
    dermoscopic_structure_score = calculate_dermoscopic_structure_score(image, mask)

    # Calculate TDS (Total Dermascopic Score)
    # TDS = (A × 1.3) + (B × 0.1) + (C × 0.5) + (D × 0.5)
    tds = (
        (asymmetry_score * 1.3) + (border_score * 0.1) + (colour_score * 0.5) + (dermoscopic_structure_score * 0.5)
    )

    result = {
        "asymmetry": asymmetry_score,
        "border": border_score,
        "colour": colour_score,
        "dermoscopic_structure": dermoscopic_structure_score,
        "tds": tds,
    }

    if show_graph:
        # Visualize the results
        visualize_abcd_analysis(image_path, mask_path, result)

    return result


def visualize_abcd_analysis(image_path: Path, mask_path: Path, results: dict[str, float | int]) -> None:
    """Visualize the ABCD analysis results.

    Args:
        image_path: Path to the lesion image.
        mask_path: Path to the binary mask of the lesion.
        results: Dictionary containing the ABCD scores.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 0).astype(np.float32)

    # Create image overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image with overlay
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title("Lesion Mask Overlay")
    ax[1].axis("off")

    # Add ABCD scores as text
    plt.figtext(
        0.5,
        0.02,
        f"A: {results['asymmetry']:.2f} (Asymmetry) | "
        f"B: {results['border']:.2f} (Border) | "
        f"C: {results['colour']:.2f} (Colour) | "
        f"D: {results['dermoscopic_structure']:.2f} (Dermoscopic Structure) | "
        f"TDS: {results['tds']:.2f}",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
