import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import (erosion, dilation, closing, opening,
                                remove_small_objects, disk)
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_otsu



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

def detect_dots_and_globules(gray_img, mask, image, save_vis_path=None):
    """Detect dots and globules based on contour area."""
    lesion = cv2.bitwise_and(image, image, mask=mask)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    dots, globules = 0, 0
    vis_img = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if mm2_to_pixels(0.1) <= area < mm2_to_pixels(0.5):
            dots += 1
            if save_vis_path:
                cv2.circle(vis_img, center, radius, (0, 255, 0), 2)  # Green for dots
        elif mm2_to_pixels(0.5) <= area < mm2_to_pixels(2.0):
            globules += 1
            if save_vis_path:
                cv2.circle(vis_img, center, radius, (0, 0, 255), 2)  # Red for globules

    return dots > 0, globules > 0, vis_img

def detect_structureless_areas(gray_img, mask, vis_img=None, save_vis_path=None, window_size=9, var_threshold=15, area_thresh=0.20):
    masked_image = gray_img * mask

    # Structural element for the upcoming morphological operations
    selem = disk(3)

    # Smooth the image by filling in small gaps, such as globules and networks
    closed_image = closing(masked_image, selem)
    
    # Subtract the complement of the masked_image from the closed image to emphasize differences
    # Change from dark to light will go to 0, light areas with little variation will remain the same
    diff_image = np.clip(closed_image.astype(int) - (255 - masked_image.astype(int)), 0, 255)

    # Apply Otsu thresholding on the difference image (only within lesion mask)
    otsu_thresh = threshold_otsu(diff_image[mask])
    structureless_mask = (diff_image > otsu_thresh) & mask

    # First, use morphological opening to remove noise, then remove small objects
    structureless_mask_improved = remove_small_objects(opening(structureless_mask, selem), min_size=20)

    # Due to mask leakage, part of the skin interferes with the results of the structureless area test
    # Erode the lesion mask to create an inner region. The erosion radius can be tuned.
    border_margin = 15  # tuning parameter: number of pixels to erode from the border
    inner_lesion_mask = erosion(mask, disk(border_margin))

    # Retain only those structureless regions that are fully contained within the inner lesion region.
    structureless_mask_final = structureless_mask_improved & inner_lesion_mask

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Gray Image")
    axes[0].axis('off')

    axes[1].imshow(masked_image, cmap='gray')
    axes[1].set_title("Closed Image")
    axes[1].axis('off')

    axes[2].imshow(closed_image, cmap='gray')
    axes[2].set_title("Difference Image")
    axes[2].axis('off')

    axes[3].imshow(diff_image, cmap='gray')
    axes[3].set_title("Otsu Threshold Image")
    axes[3].axis('off')

    axes[4].imshow(structureless_mask_improved, cmap='gray')
    axes[4].set_title("Post Noise Removal")
    axes[4].axis('off')

    axes[5].imshow(structureless_mask_final, cmap='gray')
    axes[5].set_title("Final: Inner Region Only")
    axes[5].axis('off')
    plt.tight_layout()
    plt.show()

    return False, gray_img

def compute_dermoscopic_score(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None) -> dict:
    """Compute dermoscopic structure score (Part D of ABCD rule)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure that the mask and the image have the same size
    target_shape = (mask.shape[1], mask.shape[0])  # width, height
    image = cv2.resize(image, target_shape)
    gray = cv2.resize(gray, target_shape)

    # --- Feature Detection ---
    has_dots, has_globules, vis_img = detect_dots_and_globules(gray, mask, image, save_vis_path)
    has_structureless, vis_img = detect_structureless_areas(gray, mask, vis_img, save_vis_path)

    # Not implemented yet
    has_pigment_network = None
    has_streaks = None

    # Final scoring (0.5 points per feature present)
    present_features = sum([
        int(has_dots),
        int(has_globules),
        0,  # pigment network
        0,  # streaks
        int(has_structureless)
    ])

    # Save visualization if requested
    if save_vis_path:
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        cv2.imwrite(save_vis_path, vis_img)
        print(f"Saved visualization to {save_vis_path}")

    return {
        "dots": has_dots,
        "globules": has_globules,
        "structureless_areas": has_structureless,
        "pigment_network": has_pigment_network,
        "streaks": has_streaks,
        "D_score": present_features
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