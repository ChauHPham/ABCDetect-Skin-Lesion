from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np

def calculate_color_classification(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    show_graph: bool = False,
    bins: int = 8,
) -> float:
    """
    Calculate the color classification score for the lesion based on its color heterogeneity.
    
    The function extracts the pixels within the lesion (as defined by the binary mask),
    converts them to the LAB color space, computes a 3D histogram over the three channels,
    and then computes the entropy of the color distribution. The entropy is normalized by the
    maximum possible entropy given the number of bins. This normalized entropy (0.0 to 1.0) is
    returned as the TDS contribution score for the color category.
    
    Args:
        image: An RGB image of the lesion (numpy array).
        mask: A binary numpy array mask indicating the lesion area.
        show_graph: If True, display a 2D visualization of the color distribution.
        bins: Number of bins per channel for the 3D histogram (default is 8).
    
    Returns:
        A normalized entropy score (0.0 to 1.0) representing the color variegation.
        Higher values indicate greater color diversity.
    """
    # Ensure that there are valid lesion pixels
    if np.sum(mask) == 0:
        return 0.0

    # Convert the image to LAB color space for a more perceptually uniform color representation.
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Extract only the pixels that are part of the lesion.
    lesion_pixels = lab_image[mask > 0]  # shape (num_pixels, 3)

    # Compute a 3D histogram over the LAB channels.
    # Here we use an equal number of bins in each channel.
    hist, edges = np.histogramdd(lesion_pixels, bins=(bins, bins, bins), range=[(0, 255), (0, 255), (0, 255)])
    
    # Normalize the histogram to turn counts into probabilities.
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0
    hist_prob = hist / total_pixels

    # Compute the entropy of the color distribution.
    # Only include bins with non-zero probability to avoid log(0).
    entropy = -np.sum([p * np.log2(p) for p in hist_prob.flatten() if p > 0])
    
    # Determine maximum possible entropy: if the distribution were perfectly uniform
    max_entropy = np.log2(bins ** 3)  # since we have bins^3 total bins
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    if show_graph:
        visualize_color_distribution(lesion_pixels, hist_prob, edges, entropy, normalized_entropy)

    return normalized_entropy

def visualize_color_distribution(
    lesion_pixels: np.ndarray,
    hist_prob: np.ndarray,
    edges: list,
    entropy: float,
    normalized_entropy: float,
) -> None:
    """
    Visualize the color distribution (projected into two dimensions) of the lesion.
    
    In this example we show a 2D histogram of the L vs. a channels extracted from the LAB pixels,
    along with the entropy metrics in the title.
    
    Args:
        lesion_pixels: Array of LAB pixels from the lesion.
        hist_prob: The normalized 3D histogram probabilities.
        edges: Bin edges from the histogramdd function.
        entropy: Computed raw entropy.
        normalized_entropy: Entropy normalized by the maximum possible entropy.
    """
    # We will visualize a projection: for instance, the L (lightness) and a channels.
    L_values = lesion_pixels[:, 0]
    a_values = lesion_pixels[:, 1]
    plt.figure(figsize=(8, 6))
    plt.hist2d(L_values, a_values, bins=32, cmap='inferno')
    plt.colorbar(label="Count")
    plt.title(
        f"Lesion Color Distribution (L vs. a)\n"
        f"Entropy: {entropy:.3f}, Normalized: {normalized_entropy:.3f}"
    )
    plt.xlabel("L channel")
    plt.ylabel("a channel")
    plt.show()

# Example test code; can be removed in production.
if __name__ == "__main__":
    # Create a dummy RGB image and a circular lesion mask for demonstration.
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Draw a filled circle to simulate a lesion region.
    cv2.circle(mask, (128, 128), 60, 1, -1)

    # Calculate the color classification (variegation) score.
    score = calculate_color_classification(image, mask, show_graph=True)
    print("Color classification score:", score)
