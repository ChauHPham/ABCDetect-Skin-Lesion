import os
import cv2
import numpy as np

from skimage.morphology import (erosion, closing, remove_small_objects, disk, skeletonize)
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.filters.rank import entropy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx



def calculate_dermoscopic_structure_score(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None, show_graph: bool = False) -> int:
    """Calculate dermoscopic structure score for the lesion according to ABCD rule.
    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.

    Returns:
        Dermoscopic structure score (0 to 5) based on the presence of dots, globules, and structureless areas.
    """

    return compute_dermoscopic_score(image, mask, show_graph=show_graph)["D_score"]


# --- Area conversion utils (assuming ~0.1 mm/pixel) ---
def pixels_to_mm2(area_px):
    return area_px * 0.01  # ≈ 0.1mm/pixel → 0.01 mm²/pixel

def mm2_to_pixels(area_mm2):
    return area_mm2 / 0.01

def normalize(img, min_val=None, max_val=None):
    """
    Normalize image to the range [0, 1].

    Parameters:
        img (ndarray): Input image (e.g. float32 or uint8).
        min_val (float, optional): Minimum value for normalization. Defaults to img.min().
        max_val (float, optional): Maximum value for normalization. Defaults to img.max().

    Returns:
        Normalized image in [0, 1].
    """
    min_val = img.min() if min_val is None else min_val
    max_val = img.max() if max_val is None else max_val
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - min_val) / (max_val - min_val)).astype(np.float32)

def detect_dots_and_globules(gray_img, mask, image, *, save_vis_path=None, show_graph=False):
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
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Combine shape filters
        is_circular = circularity > 0.7
        is_squareish = 0.75 < aspect_ratio < 1.25

        if is_circular and is_squareish:
        # Only now check for size and classify
            if mm2_to_pixels(0.1) <= area < mm2_to_pixels(0.5):
                dots += 1
                cv2.circle(vis_img, (int(x + w / 2), int(y + h / 2)), int(max(w, h) / 2), (0, 255, 0), 2)  # Green
            elif mm2_to_pixels(0.5) <= area < mm2_to_pixels(2.5):
                globules += 1
                cv2.circle(vis_img, (int(x + w / 2), int(y + h / 2)), int(max(w, h) / 2), (0, 0, 255), 2)  # Red

    if show_graph:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title("Grayscale Lesion")
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')

        axes[2].imshow(thresh, cmap='gray')
        axes[2].set_title("Thresholded (Binary)")
        axes[2].axis('off')

        # Show overlay with blobs (after drawing)
        axes[3].imshow(vis_img)
        axes[3].set_title("Detected Dots/Globules")
        axes[3].axis('off')

        # ✅ Add legend inside the last subplot
        red_patch = mpatches.Patch(color='red', label='Globules')
        green_patch = mpatches.Patch(color='green', label='Dots')
        axes[3].legend(handles=[green_patch, red_patch], loc='lower right', fontsize='small', frameon=True)

        plt.tight_layout()
        plt.show()


    return dots > 0, globules > 0, vis_img

def detect_structureless_areas(gray_img, mask, vis_img=None,  *, save_vis_path=None, window_size=9, var_threshold=15, area_thresh=0.20,  show_graph=False):
    """
    Criteria:
        - Occupies at least 10% of the total lesion area
        - Can be hypo-, hyper-, or normally pigmented
        - Lacks other discernible structures (e.g., dots, globules, networks, streaks)

    Detection Algorithm:
        1. Convert image to grayscale to reduce color distractions.
        2. Apply morphological closing (dilation followed by erosion) to fill in small texture gaps.
           This suppresses small structures like dots, globules, and network patterns.
        3. Compute the difference between the closed image and the inverted original image.
           This emphasizes regions that remain uniform after closing — likely to be structureless.
        4. Apply Otsu thresholding within the lesion mask to binarize the difference image:
               - Otsu's method automatically finds the intensity threshold that best separates two classes
                 (in this case, uniform vs non-uniform regions, IE minimum intra class variance).
               - This results in a binary mask of likely structureless regions.
        5. Remove small noisy detections using morphological opening and `remove_small_objects`.
        6. Erode the lesion mask to create an "inner region" and exclude border-adjacent areas that might include skin or edge artifacts.
        7. Keep only structureless regions that are fully within the eroded lesion area.
        8. Return the binary mask or indicator of whether structureless regions are present.
    """

    binary_mask = mask > 0.5
    masked_image = gray_img * binary_mask

    # Structural element for the upcoming morphological operations
    selem = disk(3)

    # Smooth the image by filling in small gaps, such as globules and networks
    closed_image = closing(masked_image, selem)
    
    # Subtract the complement of the masked_image from the closed image to emphasize differences
    # Change from dark to light will go to 0, light areas with little variation will remain the same
    diff_image = np.clip(closed_image.astype(int) - (255 - masked_image.astype(int)), 0, 255)

    # Apply Otsu thresholding on the difference image (only within lesion mask)
    otsu_thresh = threshold_otsu(diff_image[binary_mask])
    structureless_mask = (diff_image > otsu_thresh) & binary_mask

    # First, use morphological opening to remove noise, then remove small objects
    # structureless_mask_improved = opening(structureless_mask, selem)

    # Due to mask leakage, part of the skin interferes with the results of the structureless area test
    # Erode the lesion mask to create an inner region. The erosion radius can be tuned.
    border_margin = 15  # tuning parameter: number of pixels to erode from the border
    inner_lesion_mask = erosion(mask, disk(border_margin))

    border_removed_mask = closing(structureless_mask & inner_lesion_mask, selem)

    cleaned_mask = remove_small_objects(border_removed_mask.astype(bool))

    # Retain only those structureless regions that are fully contained within the inner lesion region.
    # structureless_mask_final = remove_small_objects(remove_small_objects(structureless_mask_improved & inner_lesion_mask, min_size=60))
    if show_graph:
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        fig.suptitle("Structureless Area Detection Pipeline", fontsize=16)

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

        axes[4].imshow(border_removed_mask, cmap='gray')
        axes[4].set_title("Post Noise Removal")
        axes[4].axis('off')

        axes[5].imshow(cleaned_mask, cmap='gray')
        axes[5].set_title("Final: Inner Region Only")
        axes[5].axis('off')

        plt.tight_layout()
        plt.show()

    # Check if the structureless area takes up 10% of the mask
    total_lesion_pixels = np.sum(binary_mask.astype(bool))
    total_unstructured_area = np.sum(cleaned_mask.astype(bool))

    return bool(total_unstructured_area / total_lesion_pixels > 0.10), cleaned_mask.astype(np.uint8)

def build_directional_dog_kernel(theta_deg, size=21, sigma1=(2,2), sigma2=(2,1)):
    """
    Create a Difference-of-Gaussians (DoG) kernel rotated by a given angle.
    
    Parameters:
        theta_deg (float): Angle in degrees
        size (int): Kernel size (must be odd)
        sigma1 (tuple): (σx, σy) the standard deviation (amount of blur) of the gaussian
        sigma2 (tuple): (σx, σy) std of sharper gaussian
    
    Returns:
        2D DoG kernel rotated to angle theta
    """

    theta = np.deg2rad(theta_deg)
    ax = np.linspace(-size // 2, size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # Rotate coordinates (x', y')
    x_prime = xx * np.cos(theta) + yy * np.sin(theta)
    y_prime = -xx * np.sin(theta) + yy * np.cos(theta)

    # Build Gaussians
    G1 = np.exp(-(x_prime**2 / (2 * sigma1[0]**2) + y_prime**2 / (2 * sigma1[1]**2)))
    G2 = np.exp(-(x_prime**2 / (2 * sigma2[0]**2) + y_prime**2 / (2 * sigma2[1]**2)))

    G1 /= G1.sum()
    G2 /= G2.sum()

    return G1 - G2

def apply_filter_bank(gray_image, angles=np.linspace(-90, 90, 12, endpoint=False), kernel_size=21):
    """
    Apply a bank of directional DoG filters and compute max response per pixel.

    Parameters:
        gray_image (ndarray): Input grayscale image
        angles (list): Angles to apply filters at
        kernel_size (int): Size of the DoG kernel

    Returns:
        max_response (ndarray): Max filter response across orientations
    """
    responses = []

    for angle in angles:
        kernel = build_directional_dog_kernel(angle, size=kernel_size)
        response = convolve(gray_image.astype(np.float32), kernel)
        responses.append(response)

    # Combine all directional responses
    max_response = np.max(responses, axis=0)
    return max_response

def prune_by_shape(binary_mask, ecc_thresh=0.95, solidity_thresh=0.5):
    """
    Remove elongated, wrinkle-like segments based on eccentricity and solidity.
    
    Parameters:
        binary_mask: binary image of detected structures
        ecc_thresh: max allowed eccentricity (close to 1 = long and thin)
        solidity_thresh: min allowed solidity (area / convex area)

    Returns:
        Refined binary mask with more blob-like regions.
    """

    labeled = label(binary_mask)
    out = np.zeros_like(binary_mask)
    for region in regionprops(labeled):
        if region.eccentricity < ecc_thresh and region.solidity > solidity_thresh:
            out[labeled == region.label] = 1
    return out

def compute_network(masked_image, binary_mask):
    # Apply directional DoG filters and threshold the response
    network_response = apply_filter_bank(masked_image) * binary_mask
    thresh_val = threshold_otsu(network_response[binary_mask > 0])
    network_mask = (network_response > thresh_val) & binary_mask

    # Thin structures to 1-pixel width
    network_mask = skeletonize(network_mask)

    # Remove long, thin structures based on shape
    network_mask = prune_by_shape(network_mask, ecc_thresh=0.8, solidity_thresh=0.1)

    # Keep only areas with high local entropy (textural complexity)
    entropy_img = entropy(masked_image.astype(np.uint8), disk(9))
    entropy_mask = entropy_img > threshold_otsu(entropy_img[binary_mask > 0])
    network_mask &= entropy_mask

    return network_mask

def skeleton_to_graph(skeleton_mask):
    # Convert skeleton into a graph where each pixel is a node
    G = networkx.Graph()
    coords = np.column_stack(np.nonzero(skeleton_mask))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton_mask.shape[0] and 0 <= nx_ < skeleton_mask.shape[1] and skeleton_mask[ny, nx_]):
                    G.add_edge((y, x), (ny, nx_))
    return G

def count_graph_edges_and_nodes(skeleton_mask):
    # Count nodes and edges in the graph
    G = skeleton_to_graph(skeleton_mask)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    return num_nodes, num_edges

def detect_pigment_networks(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None, show_graph=False):
    binary_mask = mask > 0.5
    green_channel = image[:, :, 1]  # Use the green channel for better contrast
    masked_green_channel = green_channel * binary_mask  # Apply lesion mask

    # Build directional filter responses for detecting hair
    responses = [apply_filter_bank(masked_green_channel, kernel_size=size) for size in [2, 5, 10, 15, 21]]
    stacked = np.stack(responses, axis=-1)

    # Detect hair by thresholding max response
    max_response = np.max(stacked, axis=-1)
    max_response = normalize(max_response, np.min(max_response), np.max(max_response))
    thresh_val = threshold_otsu(max_response[binary_mask])
    hair_mask = (max_response > thresh_val).astype(np.uint8)

    # Inpaint the green channel to remove detected hair
    inpainted_image = cv2.inpaint(masked_green_channel.astype(np.uint8), hair_mask, 5, cv2.INPAINT_TELEA)

    # Compute pigment network masks for both original and hair-inpainted images
    network_mask = compute_network(masked_green_channel, binary_mask)
    network_mask_hairless = compute_network(inpainted_image, binary_mask)

    # Use the better network (larger detected structure) as the final mask
    optimal_mask = network_mask if np.sum(network_mask) > np.sum(network_mask_hairless) else network_mask_hairless

    # Analyze the resulting graph
    V, E = count_graph_edges_and_nodes(optimal_mask)
    structure_ratio = E / (V + 1)  # Prevent division by zero
    length_ratio = np.sum(optimal_mask) / np.sum(binary_mask)
    composite_score = structure_ratio * length_ratio

    # Optional visualization for debugging and analysis
    if show_graph:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.suptitle("Pigmented Network Area Detection Pipeline", fontsize=16)

        axes[0].imshow(green_channel, cmap='gray'); axes[0].set_title("Green Channel"); axes[0].axis('off')
        axes[1].imshow(masked_green_channel, cmap='gray'); axes[1].set_title("Masked Green"); axes[1].axis('off')
        axes[2].imshow(network_mask_hairless, cmap='gray'); axes[2].set_title("Hairless Network"); axes[2].axis('off')
        axes[3].imshow(network_mask, cmap='gray'); axes[3].set_title("Original Network"); axes[3].axis('off')
        axes[4].imshow(optimal_mask, cmap='gray'); axes[4].set_title("Selected Network"); axes[4].axis('off')
        plt.tight_layout()
        plt.show()

        print(f"Vertices: {V}, Edges: {E}, Skeleton length: {np.sum(optimal_mask)}, Lesion area: {np.sum(binary_mask)}")
        print(f"Structure ratio: {structure_ratio:.4f}, Length ratio: {length_ratio:.4f}, Composite score: {composite_score:.6f}")

    # Decide if a pigment network is present
    return composite_score >= 0.1, optimal_mask

def compute_dermoscopic_score(image: np.ndarray, mask: np.ndarray, save_vis_path: str = None, show_graph: bool = False) -> dict:
    """Compute dermoscopic structure score (Part D of ABCD rule)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure that the mask and the image have the same size
    target_shape = (mask.shape[1], mask.shape[0])  # width, height
    image = cv2.resize(image, target_shape)
    gray = cv2.resize(gray, target_shape)

    # --- Feature Detection ---
    has_dots, has_globules, vis_img = detect_dots_and_globules(gray, mask, image, save_vis_path=save_vis_path, show_graph=show_graph)
    has_structureless, vis_img = detect_structureless_areas(gray, mask, vis_img, save_vis_path=save_vis_path , show_graph=show_graph)

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
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_vis_path, vis_img_rgb)
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
    show_graph = True  
    result = compute_dermoscopic_score(image, mask, save_vis_path=vis_path,show_graph=show_graph)
    print("Detected dermoscopic structures:", result)