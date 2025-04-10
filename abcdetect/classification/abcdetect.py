import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from abcdetect.classification.asymmetry import calculate_asymmetry as compute_asymmetry_score
from abcdetect.classification.border import calculate_border as compute_border_score
from abcdetect.classification.color import  calculate_color_score as compute_color_score
from abcdetect.classification.dermoscopy import compute_dermoscopic_score


def classify_tds(tds):
    if tds < 4.75:
        return "Benign"
    elif 4.75 <= tds <= 5.45:
        return "Suspicious"
    else:
        return "Malignant"


def visualize(image, mask, overlay_path):
    overlay = cv2.imread(overlay_path)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title("Detected Structures")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def main(args):
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Could not load image or mask")

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    vis_path = f"output/visuals/{base_name}_detections.jpg"

    # Run ABCD rule analysis
    A = compute_asymmetry_score(image, mask)
    B = compute_border_score(image, mask)
    C = compute_color_score(image, mask)
    D_result = compute_dermoscopic_score(image, mask, save_vis_path=vis_path)
    D = D_result["D_score"]

    # Total Dermoscopy Score
    TDS = (A * 1.3) + (B * 0.1) + (C * 0.5) + (D * 0.5)
    classification = classify_tds(TDS)

    # Print results
    print("\n=== ABCD Rule Results ===")
    print(f"Asymmetry Score (A): {A:.2f}")
    print(f"Border Score (B): {B:.2f}")
    print(f"Color Score (C): {C:.2f}")
    print(f"Dermoscopy Score (D): {D:.2f}")
    print(f"---------------------------")
    print(f"Total Dermoscopy Score: {TDS:.2f} → {classification}")

    # Plot results
    visualize(image, mask, vis_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a skin lesion using the ABCD rule")
    parser.add_argument("--image", required=True, help="Path to the lesion image")
    parser.add_argument("--mask", required=True, help="Path to the segmentation mask")
    args = parser.parse_args()

    main(args)
