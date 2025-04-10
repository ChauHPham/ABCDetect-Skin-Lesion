from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature


def calculate_asymmetry_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> float:
    """Calculate asymmetry score for the lesion according to ABCD rule.

    Args:
        image: RGB image of the lesion.
        mask: Binary mask of the lesion.
        show_graph: If True, visualize calculation steps.

    Returns:
        Asymmetry score (0-2).
    """
    # Find contours of the lesion mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)

    # Find centroid of the lesion
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0.0

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    centroid = (cx, cy)

    # Calculate the principal axes
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, np.empty(0))
    axis1 = eigenvectors[0]
    axis2 = eigenvectors[1]

    # visualize_pca(image, mask, centroid, eigenvectors, eigenvalues)

    # Calculate asymmetry along each axis
    axis1_asymmetry = is_axis_asymmetric(image, mask, centroid, axis1, show_graph=show_graph)
    axis2_asymmetry = is_axis_asymmetric(image, mask, centroid, axis2, show_graph=show_graph)

    asymmetry_score = int(axis1_asymmetry) + int(axis2_asymmetry)
    return asymmetry_score


def is_axis_asymmetric(
    image: np.ndarray,
    mask: np.ndarray,
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    *,
    show_graph: bool = False,
    shape_weight: float = 0.40,
    colour_weight: float = 0.35,
    texture_weight: float = 0.25,
    asymmetry_threshold: float = 0.20,
    color_bins: int = 16,
    lbp_bins: int = 10,
) -> bool:
    """Determine if the lesion shows asymmetry along a specific axis.

    Asymmetry is calculated based on shape, colour, and texture features.

    Args:
        image: RGB image of the lesion.
        mask: Binary mask of the lesion.
        centroid: (x, y) coordinates of the lesion centroid.
        axis_vector: Vector defining the axis direction.
        show_graph: If True, visualize calculation steps.
        shape_weight: Weight for shape asymmetry in the final score.
        colour_weight: Weight for colour asymmetry in the final score.
        texture_weight: Weight for texture asymmetry in the final score.
        asymmetry_threshold: Threshold for determining asymmetry.
        color_bins: Number of bins for color histograms.
        lbp_bins: Number of bins for LBP histograms.

    Returns:
        True if asymmetric along this axis, False otherwise.
    """
    half1_mask, half2_mask = split_mask_by_axis(mask, centroid, axis_vector)

    # visualize_mask_split(image, mask, half1_mask, half2_mask, centroid, axis_vector)

    # Calculate shape asymmetry
    shape_asymmetry = calculate_shape_asymmetry(
        half1_mask, half2_mask, centroid, axis_vector, show_graph=show_graph
    )

    # Calculate colour distribution asymmetry
    colour_asymmetry = calculate_colour_asymmetry(
        image, half1_mask, half2_mask, centroid, axis_vector, color_bins=color_bins, show_graph=show_graph
    )

    # Calculate texture/structure asymmetry
    texture_asymmetry = calculate_texture_asymmetry(
        image, half1_mask, half2_mask, centroid, axis_vector, lbp_bins=lbp_bins, show_graph=show_graph
    )

    # Combined decision - asymmetry is present if any feature is asymmetric
    # Using a weighted threshold for decision
    asymmetry_score = (
        shape_weight * shape_asymmetry + colour_weight * colour_asymmetry + texture_weight * texture_asymmetry
    )
    print(
        f"Asymmetry score: {asymmetry_score:.3f} (shape: {shape_weight * shape_asymmetry:.3f}, "
        f"colour: {colour_weight * colour_asymmetry:.3f}, "
        f"texture: {texture_weight * texture_asymmetry:.3f}). "
        f"Threshold: {asymmetry_threshold:.3f}"
    )
    return asymmetry_score > asymmetry_threshold


def split_mask_by_axis(
    mask: np.ndarray, centroid: tuple[int, int], axis_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split a mask into two halves divided by an axis through the centroid.

    Args:
        mask: Binary mask image.
        centroid: (x, y) coordinates of the center point.
        axis_vector: Vector defining the axis direction.

    Returns:
        Tuple of (half1_mask, half2_mask).
    """
    cx, cy = centroid
    height, width = mask.shape

    # Create a line through the centroid along the axis
    # The line equation: ax + by + c = 0
    a, b = axis_vector
    c = -(a * cx + b * cy)

    # Create masks for each half
    half1_mask = np.zeros_like(mask)
    half2_mask = np.zeros_like(mask)

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0:
                continue

            # Check on which side of the line the pixel lies
            val = a * x + b * y + c
            if val > 0:
                half1_mask[y, x] = 1
            else:
                half2_mask[y, x] = 1

    return half1_mask, half2_mask


def reflect_across_axis(half: np.ndarray, centroid: tuple[int, int], axis_vector: np.ndarray) -> np.ndarray:
    """Reflect a binary mask across an axis through the centroid.

    Uses direct point-by-point mapping to ensure exact correspondence.

    Args:
        half: Binary mask of one half.
        centroid: (x, y) coordinates of the centroid.
        axis_vector: Vector defining the axis direction.

    Returns:
        Reflected binary mask.
    """
    height, width = half.shape
    cx, cy = centroid
    reflected = np.zeros_like(half)

    # Normalize the axis vector
    norm = np.sqrt(axis_vector[0] ** 2 + axis_vector[1] ** 2)
    axis_vector = axis_vector / norm if norm > 0 else axis_vector
    a, b = axis_vector

    # For each point in the output, check if its reflection is in the input
    for y in range(height):
        for x in range(width):
            # Vector from centroid to pixel
            dx, dy = x - cx, y - cy

            # Calculate reflected position
            # Reflection formula: r = p - 2 * (p·n) * n
            # where p is point vector, n is normal unit vector
            dot_product = dx * a + dy * b
            rx = int(round(x - 2 * dot_product * a))
            ry = int(round(y - 2 * dot_product * b))

            # Check if reflected point is within bounds and set value
            if 0 <= rx < width and 0 <= ry < height and half[ry, rx] > 0:
                reflected[y, x] = 1

    return reflected


def calculate_shape_asymmetry(
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    *,
    show_graph: bool = False,
) -> float:
    """Calculate the shape asymmetry between two halves of a lesion.

    Args:
        half1_mask: Binary mask of the first half.
        half2_mask: Binary mask of the second half.
        centroid: (x, y) coordinates of the lesion centroid.
        axis_vector: Vector defining the axis direction.
        show_graph: If True, visualize calculation steps.

    Returns:
        Shape asymmetry score (0.0 - 1.0, higher means more asymmetric).
    """
    # Calculate areas of each half
    area1 = np.sum(half1_mask)
    area2 = np.sum(half2_mask)
    total_area = area1 + area2

    if total_area == 0:
        return 0.0

    # Area ratio asymmetry - perfect symmetry would have ratio = 1
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 1.0
    area_asymmetry = 1.0 - area_ratio

    # Apply morphological operations to reduce noise before comparing
    kernel = np.ones((3, 3), np.uint8)
    half1_smooth = cv2.morphologyEx(half1_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    half2_smooth = cv2.morphologyEx(half2_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Calculate overlap
    reflected_half = reflect_across_axis(half1_smooth, centroid, axis_vector)

    intersection = np.logical_and(reflected_half, half2_smooth).sum()
    union = np.logical_or(reflected_half, half2_smooth).sum()

    # IoU: 1.0 means perfect overlap, 0.0 means no overlap
    iou = intersection / union if union > 0 else 0.0
    shape_overlap_asymmetry = 1.0 - iou

    # Reduce sensitivity for small lesions
    lesion_size_factor = min(1.0, total_area / 1000)  # Normalize by expected size
    adjusted_weight = 0.8 + 0.1 * lesion_size_factor  # Between 0.8-0.9 based on size

    # Combine area asymmetry and shape overlap asymmetry with adjusted weights
    shape_asymmetry = (1.0 - adjusted_weight) * area_asymmetry + adjusted_weight * shape_overlap_asymmetry
    if show_graph:
        visualize_shape_asymmetry(
            half1_mask,
            half2_mask,
            half1_smooth,
            half2_smooth,
            reflected_half,
            centroid,
            axis_vector,
            area_ratio,
            iou,
            shape_asymmetry,
        )
    return shape_asymmetry


def calculate_colour_asymmetry(
    image: np.ndarray,
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    *,
    show_graph: bool = False,
    color_bins: int = 16,
) -> float:
    """Calculate the colour asymmetry between two halves of a lesion.

    Args:
        image: RGB image of the lesion
        half1_mask: Binary mask of the first half.
        half2_mask: Binary mask of the second half.
        centroid: (x, y) coordinates of the lesion centroid.
        axis_vector: Vector defining the axis direction.
        show_graph: If True, visualize calculation steps.
        color_bins: Number of bins for color histograms.

    Returns:
        colour asymmetry score (0.0 - 1.0, higher means more asymmetric).
    """
    # Convert to LAB colour space for better colour analysis
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Extract colour features from each half
    half1_features = extract_colour_features(lab_image, half1_mask, color_bins=color_bins)
    half2_features = extract_colour_features(lab_image, half2_mask, color_bins=color_bins)

    if half1_features is None or half2_features is None:
        print("Colour feature extraction failed, returning 0.0.")
        return 0.0

    # Calculate colour histogram distance and mean colour difference
    hist_distance = calculate_colour_histogram_distance(half1_features["hist"], half2_features["hist"])
    mean_diff = calculate_mean_colour_difference(half1_features["mean"], half2_features["mean"])

    # Combine metrics (weighted average)
    colour_asymmetry = 0.6 * hist_distance + 0.4 * mean_diff

    if show_graph:
        visualize_colour_asymmetry(
            image,
            half1_mask,
            half2_mask,
            half1_features,
            half2_features,
            centroid,
            axis_vector,
            colour_asymmetry,
            hist_distance,
            mean_diff,
            color_bins,
        )
    return colour_asymmetry


def extract_colour_features(
    lab_image: np.ndarray, mask: np.ndarray, *, color_bins: int = 16
) -> dict[str, Any] | None:
    """Extract colour features from a masked region of an image.

    Args:
        lab_image: Image in LAB colour space.
        mask: Binary mask indicating the region of interest.
        color_bins: Number of bins for color histograms.

    Returns:
        Dictionary containing colour features (histogram, mean colour), or None if mask is empty.
    """
    if np.sum(mask) == 0:
        return None

    # Extract masked region
    masked_lab = lab_image.copy()
    masked_lab[mask == 0] = [0, 0, 0]

    # Compute mean colour
    l_vals = masked_lab[:, :, 0][mask > 0]
    a_vals = masked_lab[:, :, 1][mask > 0]
    b_vals = masked_lab[:, :, 2][mask > 0]

    mean_colour = [np.mean(l_vals), np.mean(a_vals), np.mean(b_vals)]

    # Compute colour histogram with variable bin count
    l_hist = np.histogram(l_vals, bins=color_bins, range=(0, 255))[0]
    a_hist = np.histogram(a_vals, bins=color_bins, range=(0, 255))[0]
    b_hist = np.histogram(b_vals, bins=color_bins, range=(0, 255))[0]

    # Normalize histograms
    l_hist = l_hist / np.sum(l_hist) if np.sum(l_hist) > 0 else l_hist
    a_hist = a_hist / np.sum(a_hist) if np.sum(a_hist) > 0 else a_hist
    b_hist = b_hist / np.sum(b_hist) if np.sum(b_hist) > 0 else b_hist

    return {"mean": mean_colour, "hist": [l_hist, a_hist, b_hist]}


def calculate_colour_histogram_distance(hist1: list[np.ndarray], hist2: list[np.ndarray]) -> float:
    """Calculate the distance between two colour histograms using Chi-Square distance.

    Args:
        hist1: First histogram (list of channel histograms).
        hist2: Second histogram (list of channel histograms).

    Returns:
        Distance value (0.0 - 1.0, higher means more different).
    """
    total_distance = 0

    for h1, h2 in zip(hist1, hist2):
        # Calculate Chi-Square distance
        chi_square = np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-10))
        total_distance += chi_square

    # Normalize to 0-1 range
    normalized_distance = min(1.0, total_distance / 1.5)
    return normalized_distance


def calculate_mean_colour_difference(mean1: list[float], mean2: list[float]) -> float:
    """Calculate the difference between mean colours using CIEDE2000 colour difference.

    Args:
        mean1: First mean colour [L, a, b].
        mean2: Second mean colour [L, a, b].

    Returns:
        Normalized colour difference (0.0 - 1.0).
    """
    delta_l = mean1[0] - mean2[0]
    delta_a = mean1[1] - mean2[1]
    delta_b = mean1[2] - mean2[2]

    # Euclidean distance in LAB space
    distance = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

    # Normalize to 0-1 range (max possible distance in LAB space is approx 255*sqrt(3))
    normalized_distance = min(1.0, distance / 75.0)
    return normalized_distance


def calculate_texture_asymmetry(
    image: np.ndarray,
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    *,
    show_graph: bool = False,
    lbp_bins: int = 10,
) -> float:
    """Calculate the texture/structure asymmetry between two halves of a lesion.

    Args:
        image: RGB image of the lesion.
        half1_mask: Binary mask of the first half.
        half2_mask: Binary mask of the second half.
        centroid: (x, y) coordinates of the lesion centroid.
        axis_vector: Vector defining the axis direction.
        show_graph: If True, visualize calculation steps.
        lbp_bins: Number of bins for LBP histograms.

    Returns:
        Texture asymmetry score (0.0 - 1.0, higher means more asymmetric).
    """
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract texture features from each half
    half1_features = extract_texture_features(gray, half1_mask, lbp_bins=lbp_bins)
    half2_features = extract_texture_features(gray, half2_mask, lbp_bins=lbp_bins)

    if half1_features is None or half2_features is None:
        print("Texture feature extraction failed, returning 0.0.")
        return 0.0

    diff_gabor = np.mean(np.abs(half1_features["gabor"] - half2_features["gabor"]))
    diff_lbp = np.mean(np.abs(half1_features["lbp"] - half2_features["lbp"]))

    # Apply balanced sensitivity multipliers
    gabor_multiplier = 30.0
    lbp_multiplier = 60.0
    diff_gabor = min(1.0, diff_gabor * gabor_multiplier)
    diff_lbp = min(1.0, diff_lbp * lbp_multiplier)

    # Combine metrics with adjusted weighting
    texture_asymmetry = min(1.0, (0.6 * diff_gabor + 0.4 * diff_lbp))

    if show_graph:
        visualize_texture_asymmetry(
            image,
            half1_mask,
            half2_mask,
            half1_features,
            half2_features,
            centroid,
            axis_vector,
            texture_asymmetry,
            diff_gabor,
            diff_lbp,
            lbp_bins,
        )

    return texture_asymmetry


def extract_texture_features(
    gray_img: np.ndarray, mask: np.ndarray, *, lbp_bins: int = 10
) -> dict[str, np.ndarray] | None:
    """Extract texture features from a masked region of a grayscale image.

    Args:
        gray_img: Grayscale image.
        mask: Binary mask indicating the region of interest.
        lbp_bins: Number of bins for LBP histograms.

    Returns:
        Dictionary containing texture features, or None if mask is empty.
    """
    if np.sum(mask) == 0:
        return None

    # Apply mask
    masked_img = gray_img.copy()
    masked_img[mask == 0] = 0

    # Apply moderate contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    temp_img = np.zeros_like(masked_img)
    temp_img[mask > 0] = masked_img[mask > 0]
    enhanced = clahe.apply(temp_img)
    masked_img_enhanced = np.zeros_like(masked_img)
    masked_img_enhanced[mask > 0] = enhanced[mask > 0]

    # Reduced set of Gabor features
    features = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:  # 4 orientations
        for sigma in [1.0, 3.0]:  # 2 scales
            kernel = cv2.getGaborKernel((9, 9), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            # Normalize kernel for consistent response magnitudes
            kernel = kernel / np.sum(np.abs(kernel))
            filtered = cv2.filter2D(masked_img_enhanced, cv2.CV_32F, kernel)

            valid_pixels = filtered[mask > 0]
            if len(valid_pixels) > 0:
                features.append(np.mean(valid_pixels))
                features.append(np.std(valid_pixels))
            else:
                features.append(0)
                features.append(0)

    gabor_features = np.array(features)

    # Ensure proper normalization
    gabor_max = np.max(np.abs(gabor_features))
    if gabor_max > 0:
        gabor_features = gabor_features / gabor_max

    # LBP with 2 scales instead of 3
    lbp_features = []
    for radius in [1, 2]:
        try:
            lbp = feature.local_binary_pattern(masked_img_enhanced, 8, radius, method="uniform")
            hist, _ = np.histogram(lbp[mask > 0], bins=lbp_bins, range=(0, lbp_bins), density=True)
            lbp_features.extend(hist)
        except Exception:
            lbp_features.extend([1.0 / lbp_bins] * lbp_bins)

    return {"gabor": gabor_features, "lbp": np.array(lbp_features)}


def visualize_pca(
    image: np.ndarray, mask: np.ndarray, centroid: tuple[int, int], eigvecs: np.ndarray, eigvals: np.ndarray
) -> None:
    """Visualize the centroid and principal axes on the image."""
    vis_img = image.copy()

    # Draw the lesion boundary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

    # Draw the centroid
    cx, cy = centroid
    cv2.circle(vis_img, (cx, cy), 5, (255, 0, 0), -1)

    # Scale factor for better visualization
    scale = np.sqrt(np.max(eigvals)) * 2

    # Draw the principal axes
    for i, eigvec in enumerate(eigvecs):
        # Calculate end points
        length = scale * np.sqrt(eigvals[i])
        end1_x, end1_y = int(cx + length * eigvec[0]), int(cy + length * eigvec[1])
        end2_x, end2_y = int(cx - length * eigvec[0]), int(cy - length * eigvec[1])

        # Draw lines in different colours
        colour = (0, 0, 255) if i == 0 else (255, 255, 0)
        cv2.line(vis_img, (cx, cy), (end1_x, end1_y), colour, 2)
        cv2.line(vis_img, (cx, cy), (end2_x, end2_y), colour, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_img)
    plt.title("Lesion with Centroid and Principal Axes")
    plt.axis("off")
    plt.show()


def visualize_mask_split(
    image: np.ndarray,
    orig_mask: np.ndarray,
    h1_mask: np.ndarray,
    h2_mask: np.ndarray,
    centroid: tuple[int, int],
    ax_vec: np.ndarray,
) -> None:
    """Visualize the split mask and the axis line."""
    # Create visualization image
    vis_img = image.copy()

    # Create coloured overlay for each half
    overlay = np.zeros_like(image)
    overlay[h1_mask > 0] = [255, 0, 0]  # First half in red
    overlay[h2_mask > 0] = [0, 0, 255]  # Second half in blue

    # Blend with original image
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    # Draw the axis line
    cx, cy = centroid
    height, width = orig_mask.shape[:2]

    # Calculate line endpoints that extend across the image
    scale = max(width, height)
    end1_x = int(cx + scale * ax_vec[0])
    end1_y = int(cy + scale * ax_vec[1])
    end2_x = int(cx - scale * ax_vec[0])
    end2_y = int(cy - scale * ax_vec[1])

    cv2.line(vis_img, (cx, cy), (end1_x, end1_y), (0, 255, 0), 2)  # Green line
    cv2.line(vis_img, (cx, cy), (end2_x, end2_y), (0, 255, 0), 2)
    cv2.circle(vis_img, (cx, cy), 5, (255, 255, 0), -1)  # Yellow dot for centroid

    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.title("Split Mask Visualization (Red/Blue)")
    plt.axis("off")
    plt.show()


def visualize_shape_asymmetry(
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    half1_smooth: np.ndarray,
    half2_smooth: np.ndarray,
    reflected_half: np.ndarray,
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    area_ratio: float,
    iou: float,
    final_score: float,
) -> None:
    """Visualize the shape asymmetry calculation process."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Original halves
    combined = np.zeros((*half1_mask.shape, 3), dtype=np.uint8)
    combined[half1_mask > 0] = [255, 0, 0]  # Red for first half
    combined[half2_mask > 0] = [0, 0, 255]  # Blue for second half
    ax[0, 0].imshow(combined)
    ax[0, 0].set_title("Original Halves")
    ax[0, 0].axis("off")

    # Draw the axis
    height, width = half1_mask.shape
    cx, cy = centroid
    scale = max(width, height)
    end1_x = int(cx + scale * axis_vector[0])
    end1_y = int(cy + scale * axis_vector[1])
    end2_x = int(cx - scale * axis_vector[0])
    end2_y = int(cy - scale * axis_vector[1])
    ax[0, 0].plot([end2_x, end1_x], [end2_y, end1_y], "g-", linewidth=2)
    ax[0, 0].plot(cx, cy, "yo", markersize=6)

    # Smoothed halves
    smoothed = np.zeros((*half1_smooth.shape, 3), dtype=np.uint8)
    smoothed[half1_smooth > 0] = [255, 0, 0]
    smoothed[half2_smooth > 0] = [0, 0, 255]
    ax[0, 1].imshow(smoothed)
    ax[0, 1].set_title("Smoothed Halves")
    ax[0, 1].axis("off")

    # Reflected half vs second half
    comparison = np.zeros((*half1_mask.shape, 3), dtype=np.uint8)
    comparison[reflected_half > 0] = [255, 0, 0]  # Red for reflected first half
    comparison[half2_smooth > 0] = [0, 0, 255]  # Blue for second half
    # Green for overlap
    overlap = np.logical_and(reflected_half, half2_smooth)
    comparison[overlap] = [0, 255, 0]
    ax[1, 0].imshow(comparison)
    ax[1, 0].set_title(f"Reflection Comparison (Overlap in Green)")
    ax[1, 0].axis("off")

    # Difference visualization
    difference = np.zeros((*half1_mask.shape, 3), dtype=np.uint8)
    only_reflected = np.logical_and(reflected_half, np.logical_not(half2_smooth))
    only_second = np.logical_and(half2_smooth, np.logical_not(reflected_half))
    difference[only_reflected] = [255, 0, 0]  # Only in reflected half
    difference[only_second] = [0, 0, 255]  # Only in second half
    ax[1, 1].imshow(difference)
    ax[1, 1].set_title("Difference Areas")
    ax[1, 1].axis("off")

    # Add metrics as text
    fig.suptitle(f"Shape Asymmetry Analysis", fontsize=16)
    fig.text(
        0.5,
        0.02,
        f"Area Ratio: {area_ratio:.3f} | IoU: {iou:.3f} | " f"Shape Asymmetry Score: {final_score:.3f}",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.08)
    plt.show()


def visualize_colour_asymmetry(
    image: np.ndarray,
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    half1_features: dict[str, Any],
    half2_features: dict[str, Any],
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    colour_asymmetry: float,
    hist_distance: float,
    mean_diff: float,
    color_bins: int,
) -> None:
    """Visualize the colour asymmetry calculation process."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Original image with halves overlay
    combined = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[half1_mask > 0] = [255, 0, 0]  # Red for first half
    overlay[half2_mask > 0] = [0, 0, 255]  # Blue for second half
    combined = cv2.addWeighted(combined, 0.7, overlay, 0.3, 0)

    # Draw the axis
    cx, cy = centroid
    height, width = image.shape[:2]
    scale = max(width, height)
    end1_x = int(cx + scale * axis_vector[0])
    end1_y = int(cy + scale * axis_vector[1])
    end2_x = int(cx - scale * axis_vector[0])
    end2_y = int(cy - scale * axis_vector[1])
    cv2.line(combined, (cx, cy), (end1_x, end1_y), (0, 255, 0), 2)
    cv2.line(combined, (cx, cy), (end2_x, end2_y), (0, 255, 0), 2)
    cv2.circle(combined, (cx, cy), 5, (255, 255, 0), -1)

    ax[0, 0].imshow(combined)
    ax[0, 0].set_title("Image with Halves Overlay")
    ax[0, 0].axis("off")

    # Extract masked regions
    half1_img = image.copy()
    half1_img[half1_mask == 0] = [0, 0, 0]
    half2_img = image.copy()
    half2_img[half2_mask == 0] = [0, 0, 0]

    ax[0, 1].imshow(half1_img)
    ax[0, 1].set_title("First Half")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(half2_img)
    ax[0, 2].set_title("Second Half")
    ax[0, 2].axis("off")

    # colour histograms
    channels = ["L", "a", "b"]
    x = np.arange(color_bins)  # Use variable bin count

    for i, channel in enumerate(channels):
        ax[1, i].bar(x - 0.2, half1_features["hist"][i], width=0.4, color="red", alpha=0.7, label="Half 1")
        ax[1, i].bar(x + 0.2, half2_features["hist"][i], width=0.4, color="blue", alpha=0.7, label="Half 2")
        ax[1, i].set_title(f"{channel} Channel Histogram")
        ax[1, i].set_xlim(-0.5, color_bins - 0.5)
        ax[1, i].legend()

    # Add metrics as text
    fig.suptitle(f"Colour Asymmetry Analysis", fontsize=16)
    fig.text(
        0.5,
        0.05,
        f"Histogram Distance: {hist_distance:.3f} | Mean Colour Difference: {mean_diff:.3f} | "
        f"Colour Asymmetry Score: {colour_asymmetry:.3f}",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.15)
    plt.show()


def visualize_texture_asymmetry(
    image: np.ndarray,
    half1_mask: np.ndarray,
    half2_mask: np.ndarray,
    half1_features: dict[str, np.ndarray],
    half2_features: dict[str, np.ndarray],
    centroid: tuple[int, int],
    axis_vector: np.ndarray,
    texture_asymmetry: float,
    diff_gabor: float,
    diff_lbp: float,
    lbp_bins: int,
) -> None:
    """Visualize the texture asymmetry calculation process."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Original image with halves overlay
    combined = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[half1_mask > 0] = [255, 0, 0]  # Red for first half
    overlay[half2_mask > 0] = [0, 0, 255]  # Blue for second half
    combined = cv2.addWeighted(combined, 0.7, overlay, 0.3, 0)

    # Draw the axis
    cx, cy = centroid
    height, width = image.shape[:2]
    scale = max(width, height)
    end1_x = int(cx + scale * axis_vector[0])
    end1_y = int(cy + scale * axis_vector[1])
    end2_x = int(cx - scale * axis_vector[0])
    end2_y = int(cy - scale * axis_vector[1])
    cv2.line(combined, (cx, cy), (end1_x, end1_y), (0, 255, 0), 2)
    cv2.line(combined, (cx, cy), (end2_x, end2_y), (0, 255, 0), 2)
    cv2.circle(combined, (cx, cy), 5, (255, 255, 0), -1)

    ax[0, 0].imshow(combined)
    ax[0, 0].set_title("Image with Halves Overlay")
    ax[0, 0].axis("off")

    # Show grayscale versions for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    half1_gray = gray.copy()
    half1_gray[half1_mask == 0] = 0
    half2_gray = gray.copy()
    half2_gray[half2_mask == 0] = 0

    ax[0, 1].imshow(half1_gray, cmap="gray")
    ax[0, 1].set_title("First Half (Grayscale)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(half2_gray, cmap="gray")
    ax[0, 2].set_title("Second Half (Grayscale)")
    ax[0, 2].axis("off")

    # Show Gabor features
    x = np.arange(len(half1_features["gabor"]))
    ax[1, 0].bar(x - 0.2, half1_features["gabor"], width=0.4, color="red", alpha=0.7, label="Half 1")
    ax[1, 0].bar(x + 0.2, half2_features["gabor"], width=0.4, color="blue", alpha=0.7, label="Half 2")
    ax[1, 0].set_title("Gabor Filter Responses")
    ax[1, 0].set_xlabel("Filter Orientation")
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels([f"{int(i*45)}°" for i in range(len(half1_features["gabor"]))])
    ax[1, 0].legend()

    # Show LBP features
    x_lbp = np.arange(len(half1_features["lbp"]))
    ax[1, 1].bar(x_lbp - 0.2, half1_features["lbp"], width=0.4, color="red", alpha=0.7, label="Half 1")
    ax[1, 1].bar(x_lbp + 0.2, half2_features["lbp"], width=0.4, color="blue", alpha=0.7, label="Half 2")
    ax[1, 1].set_title("LBP Histogram")
    ax[1, 1].set_xlabel("LBP Bin")
    ax[1, 1].set_xlim(-0.5, lbp_bins - 0.5)
    ax[1, 1].legend()

    # Feature difference visualization
    ax[1, 2].plot(x, np.abs(half1_features["gabor"] - half2_features["gabor"]), "r-o", label="Gabor Difference")
    ax[1, 2].plot(x_lbp, np.abs(half1_features["lbp"] - half2_features["lbp"]), "b-o", label="LBP Difference")
    ax[1, 2].set_title("Feature Differences")
    ax[1, 2].legend()

    # Add metrics as text
    fig.suptitle("Texture Asymmetry Analysis", fontsize=16)
    fig.text(
        0.5,
        0.05,
        f"Gabor Difference: {diff_gabor:.3f} | LBP Difference: {diff_lbp:.3f} | "
        f"Texture Asymmetry Score: {texture_asymmetry:.3f}",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.15)
    plt.show()
