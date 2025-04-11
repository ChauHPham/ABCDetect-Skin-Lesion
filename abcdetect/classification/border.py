from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def calculate_border_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate the border score for the lesion according to ABCD rule.

    The border score is evaluated through the presence of sharp distinct edges and irregular borders.
    Distinct edges and irregular borders are given a score of 1. The maximum border
    score is 8 and minimum border score is 0.

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization graph for detected border (weak and strong edges).

    Returns:
        Border score (0 to 8) based on the presence of distinct edges.
    """
    original_image = image.copy()

    image, mask = preprocess_image(image, mask)

    # Find the center of the lesion using moments
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center = (center_x, center_y)

    border_region = create_border_region(mask)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    main_contour = max(contours, key=cv2.contourArea) if contours else None

    # Create 8 segments and evaluate each
    border_score = 0
    angles = np.linspace(0, 2 * np.pi, 9)  # 9 points to create 8 segments

    # Segment results for visualization
    segment_results = []

    # Calculate global shape metrics for the entire lesion
    if main_contour is not None:
        irregularity_threshold = get_irregularity_threshold(mask, main_contour)
    else:
        irregularity_threshold = 1.5

    # Evaluate each segment
    for i in range(8):
        # Create a pizza-shaped segment mask
        segment_mask = create_segment_mask(border_region, i, angles, center)
        segment_border = segment_mask * border_region

        if np.sum(segment_border) == 0:  # Skip empty segments
            segment_results.append(
                {
                    "segment_id": i,
                    "start_angle": angles[i],
                    "end_angle": angles[i + 1],
                    "is_counted": False,
                    "colour_contrast": 0,
                    "irregularity": 0,
                }
            )
            continue

        segment_contrast = calculate_segment_contrast(image, mask, segment_border)

        # Calculate border irregularity for this segment using the new simplified method
        segment_irregularity = 0
        if main_contour is not None:
            segment_irregularity = calculate_segment_irregularity(
                main_contour, float(angles[i]), float(angles[i + 1]), center
            )

        sharp_edge = segment_contrast > 30
        irregular_border = segment_irregularity > irregularity_threshold

        is_counted = sharp_edge or irregular_border
        if is_counted:
            border_score += 1

        segment_results.append(
            {
                "segment_id": i,
                "start_angle": angles[i],
                "end_angle": angles[i + 1],
                "is_counted": is_counted,
                "colour_contrast": segment_contrast,
                "irregularity": segment_irregularity,
                "irregularity_threshold": irregularity_threshold,
                "sharp_edge": sharp_edge,
                "irregular_border": irregular_border,
            }
        )

    sharp_edges = sum(1 for r in segment_results if r.get("sharp_edge", False))
    irregular_borders = sum(1 for r in segment_results if r.get("irregular_border", False))
    print(f"Border score: {border_score}/8 (Sharp edges: {sharp_edges}, Irregular borders: {irregular_borders})")

    if show_graph:
        visualize_border_analysis(original_image, image, mask, border_region, center, angles, segment_results)
    return border_score


def get_irregularity_threshold(mask: np.ndarray, contour: np.ndarray) -> float:
    """Calculate an appropriate irregularity threshold based on the lesion shape.

    Args:
        mask: Binary mask of the lesion.
        contour: OpenCV contour of the lesion.

    Returns:
        Irregularity threshold value for segment analysis.
    """
    # Use scikit-image's regionprops for shape analysis
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)[0] if measure.regionprops(label_mask) else None

    if props is None:  # No valid region found
        return 1.5

    # Calculate basic shape metrics
    perimeter = cv2.arcLength(contour, True)
    area = props.area

    # Circularity = 4π*area/perimeter^2 (1.0 for perfect circle, less for irregular shapes)
    # But we are using the inverse for better interpretability (higher is more irregular)
    if area > 0:
        circularity = (perimeter**2) / (4 * np.pi * area)
    else:
        circularity = 1.0

    # Set irregularity threshold based on overall shape
    irregularity_threshold = 1.5  # Default value
    if circularity > 1.5:  # Already quite irregular
        irregularity_threshold = 1.3
    elif circularity < 1.2:  # Very regular overall
        irregularity_threshold = 1.7
    return irregularity_threshold


def calculate_segment_irregularity(
    contour: np.ndarray, start_angle: float, end_angle: float, center: tuple[int, int]
) -> float:
    """Calculate a simplified border irregularity score for a specific segment.

    Args:
        contour: OpenCV contour of the lesion.
        start_angle: Starting angle of the segment (in radians).
        end_angle: Ending angle of the segment (in radians).
        center: (x, y) tuple of the lesion center.

    Returns:
        Irregularity score for the segment.
    """
    center_x, center_y = center

    # Extract points in this segment
    segment_points = []
    for point in contour:
        x, y = point[0]
        dx = x - center_x
        dy = y - center_y
        angle = np.arctan2(dy, dx) % (2 * np.pi)

        if start_angle <= angle < end_angle:
            segment_points.append(point)

    if len(segment_points) < 3:  # Not enough points to calculate irregularity
        return 0.0

    # Calculate distance from center to each point
    distances = []
    for point in segment_points:
        x, y = point[0]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distances.append(distance)

    # Apply outlier rejection to distances using IQR method
    distances = np.array(distances)
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    # Filter out distance outliers
    filtered_distances = distances[(distances >= lower_bound) & (distances <= upper_bound)]
    if len(filtered_distances) < 3:
        filtered_distances = distances  # Fallback to original distances if too few points

    distance_std = np.std(filtered_distances)
    distance_mean = np.mean(filtered_distances)

    distance_variation = 0
    if distance_mean > 1e-6:  # Prevent division by zero
        distance_variation = min(2.0, distance_std / distance_mean)  # Cap at 2.0

    # Calculate the chord length (straight line between endpoints)
    perimeter_ratio = 1.0
    if len(segment_points) >= 2:
        x1, y1 = segment_points[0][0]
        x2, y2 = segment_points[-1][0]
        chord_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the actual perimeter length along the segment
        perimeter = 0
        for i in range(1, len(segment_points)):
            x1, y1 = segment_points[i - 1][0]
            x2, y2 = segment_points[i][0]
            perimeter += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if chord_length > 1e-6:  # Prevent division by zero
            perimeter_ratio = min(perimeter / chord_length, 5.0)  # Cap at 5.0

    irregularity_score = distance_variation * 5.0 + (perimeter_ratio - 1.0) * 1.5
    return min(irregularity_score, 10.0)  # Cap at 10.0


def calculate_segment_contrast(image: np.ndarray, mask: np.ndarray, segment_border: np.ndarray) -> float:
    """Calculate the colour contrast across the border for a specific segment.

    Args:
        image: RGB image of the lesion.
        mask: Binary mask of the lesion.
        segment_border: Binary mask of the segment border region.

    Returns:
        Contrast value representing the colour difference inside vs outside.
    """
    # Convert to LAB for better perceptual colour difference
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Create dilated and eroded versions of the segment border
    kernel = np.ones((3, 3), np.uint8)
    outer_border = cv2.dilate(segment_border, kernel, iterations=1) & ~mask
    inner_border = cv2.dilate(segment_border, kernel, iterations=1) & mask

    if np.sum(outer_border) == 0 or np.sum(inner_border) == 0:
        return 0

    # Get average LAB colours on both sides of the border
    inner_colour = np.mean(lab_image[inner_border > 0], axis=0)
    outer_colour = np.mean(lab_image[outer_border > 0], axis=0)

    # Calculate Euclidean distance in LAB space (Delta E)
    colour_distance = np.sqrt(np.sum((inner_colour - outer_colour) ** 2))
    return colour_distance


def preprocess_image(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the image to reduce noise like hair and bubbles.

    Args:
        image: Original RGB image.
        mask: Binary mask of the lesion.

    Returns:
        Tuple of (preprocessed_image, mask).
    """

    # 1. Apply bilateral filter to reduce noise while preserving edges
    image = cv2.bilateralFilter(image, 5, 50, 50)

    # 2. Hair removal using morphological operations
    # Convert to grayscale for hair detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a kernel for hair detection (line-shaped)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply blackhat morphological operation to detect dark hair
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, kernel, iterations=1)

    # Inpaint areas with detected hair
    image = cv2.inpaint(image, hair_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return image, mask


def create_border_region(mask: np.ndarray, width_factor: float = 0.2) -> np.ndarray:
    """Create a precise border region around the mask.

    Args:
        mask: Binary mask of the lesion.
        width_factor: Border width as a fraction of the mask's equivalent diameter.

    Returns:
        Binary mask of the border region.
    """
    # Calculate the approximate lesion diameter
    area = np.sum(mask)
    if area == 0:
        return np.zeros_like(mask)

    equiv_diameter = np.sqrt(4 * area / np.pi)
    total_width = max(6, int(equiv_diameter * width_factor))  # Minimum width of 6 pixels

    inner_width = max(6, int(total_width * 0.8))
    outer_width = max(1, int(total_width * 0.2))

    # Create kernels for inward and outward expansion
    inner_kernel = np.ones((inner_width, inner_width), np.uint8)
    outer_kernel = np.ones((outer_width, outer_width), np.uint8)

    # Create dilated mask for outer boundary
    dilated_mask = cv2.dilate(mask, outer_kernel, iterations=1)

    # Create eroded mask for inner boundary
    eroded_mask = cv2.erode(mask, inner_kernel, iterations=1)

    # Border region = dilated_mask - eroded_mask
    border_region = dilated_mask.copy()
    border_region[eroded_mask > 0] = 0
    return border_region


def create_segment_mask(
    border_region: np.ndarray, segment_idx: int, angles: np.ndarray, center: tuple[int, int]
) -> np.ndarray:
    """Create a mask for a specific segment of the border.

    Args:
        border_region: Binary mask of the border region.
        segment_idx: Index of the segment (0-7).
        angles: Array of angle values defining segment boundaries.
        center: (x, y) coordinates of the lesion center.

    Returns:
        Binary mask for the specified segment.
    """
    center_x, center_y = center
    segment_mask = np.zeros_like(border_region)
    start_angle = angles[segment_idx]
    end_angle = angles[segment_idx + 1]

    y_indices, x_indices = np.where(border_region > 0)
    for y, x in zip(y_indices, x_indices):
        dx = x - center_x
        dy = y - center_y
        angle = np.arctan2(dy, dx) % (2 * np.pi)

        # Include points within the segment's angle range
        if start_angle <= angle < end_angle:
            segment_mask[y, x] = 1

    return segment_mask


def visualize_border_analysis(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    mask: np.ndarray,
    border_region: np.ndarray,
    center: tuple[int, int],
    angles: np.ndarray,
    segment_results: list[dict[str, Any]],
) -> None:
    """Visualize the border analysis process.

    Args:
        original_image: Original unprocessed RGB image.
        processed_image: Preprocessed RGB image used for analysis.
        mask: Binary mask of the lesion.
        border_region: Binary mask of the border region.
        center: (x, y) coordinates of the lesion center.
        angles: Array of angles used to segment the border.
        segment_results: Results for each segment.
    """
    center_x, center_y = center
    height, width = mask.shape[:2]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Border Analysis", fontsize=14)

    # Original and preprocessed images
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(processed_image)
    axes[0, 1].set_title("Preprocessed Image")
    axes[0, 1].axis("off")

    # Create a coloured overlay for each segment with transparency
    segment_vis = processed_image.copy()
    segment_overlay = np.zeros_like(processed_image)

    for result in segment_results:
        segment_id = result["segment_id"]
        start_angle = result["start_angle"]
        end_angle = result["end_angle"]
        is_counted = result["is_counted"]

        # Create segment mask
        segment_mask = np.zeros_like(mask)
        y_indices, x_indices = np.where(border_region > 0)
        for y, x in zip(y_indices, x_indices):
            dx = x - center_x
            dy = y - center_y
            angle = np.arctan2(dy, dx) % (2 * np.pi)
            if start_angle <= angle < end_angle:
                segment_mask[y, x] = 1

        if is_counted:
            colour = [0, 255, 0]  # Green (counted in score)
        else:
            colour = [255, 0, 0]  # Red (not counted in score)

        segment_overlay[segment_mask > 0] = colour

        # Add segment number label at mid-angle
        mid_angle = (start_angle + end_angle) / 2
        label_radius = max(width, height) * 0.35  # Position label at 35% of radius
        label_x = int(center_x + label_radius * np.cos(mid_angle))
        label_y = int(center_y + label_radius * np.sin(mid_angle))
        cv2.putText(
            segment_vis,
            str(segment_id + 1),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Blend the segment overlay with transparency
    segment_vis = cv2.addWeighted(segment_vis, 1.0, segment_overlay, 0.4, 0)

    # Mark center and draw partition lines
    cv2.circle(segment_vis, (center_x, center_y), 5, (255, 255, 0), -1)
    max_radius = max(width, height)
    for angle in angles:
        end_x = int(center_x + max_radius * np.cos(angle))
        end_y = int(center_y + max_radius * np.sin(angle))
        cv2.line(segment_vis, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)

    axes[1, 0].imshow(segment_vis)
    axes[1, 0].set_title("Border Segments (Green: Counted, Red: Not Counted)")
    axes[1, 0].axis("off")

    # Feature comparison bar chart
    segment_ids = []
    contrast_values = []
    irregularity_values = []

    for result in segment_results:
        segment_ids.append(f"Seg {result['segment_id']+1}")
        contrast_values.append(result.get("colour_contrast", 0))
        irregularity_values.append(result.get("irregularity", 0))

    # Bar chart for feature comparison
    x = np.arange(len(segment_ids))
    width = 0.4

    axes[1, 1].bar(x - width / 2, contrast_values, width, color="lightgreen", label="Colour Contrast")
    axes[1, 1].bar(x + width / 2, irregularity_values, width, color="plum", label="Border Irregularity")

    # Add threshold lines
    if segment_results:
        irreg_threshold = segment_results[0].get("irregularity_threshold", 0)
        axes[1, 1].axhline(y=30, color="green", linestyle="--", label="Contrast Threshold: 30")
        axes[1, 1].axhline(
            y=irreg_threshold,
            color="purple",
            linestyle="--",
            label=f"Irregularity Threshold: {irreg_threshold:.2f}",
        )

    axes[1, 1].set_title("Border Features by Segment")
    axes[1, 1].set_xlabel("Segment")
    axes[1, 1].set_ylabel("Feature Value")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(segment_ids)
    axes[1, 1].legend(loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
