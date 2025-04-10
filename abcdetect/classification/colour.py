import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

def calculate_colour_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate the colour score for the lesion according to ABCD rule.
    
    The ABCD rule evaluates the presence of six colors:
    - white
    - red
    - light brown
    - dark brown
    - blue-gray
    - black
    
    Each color present adds 1 to the score, with a maximum of 6 and minimum of 1.
    
    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization graph for detected colors.
    
    Returns:
        Number of distinct colors detected in the lesion (1-6).
    """
    # Ensure image and mask are the same size
    if image.shape[:2] != mask.shape[:2]:
        target_shape = (mask.shape[1], mask.shape[0])
        image = cv2.resize(image, target_shape)
    
    # Convert binary mask to boolean and uint8
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Calculate lesion area for percentage calculations
    lesion_area = np.sum(binary_mask)
    if lesion_area == 0:
        return 1  # Default to 1 if no lesion detected
    
    # Apply mask to image to isolate the lesion
    masked_image = image.copy()
    masked_image[binary_mask == 0] = 0
    
    # Convert to HSV for better color segmentation
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    
    # Get surrounding skin color for white detection reference
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    surrounding_skin_mask = dilated_mask - binary_mask
    
    # Get average skin brightness if surrounding skin exists
    if np.sum(surrounding_skin_mask) > 0:
        surrounding_skin = image.copy()
        surrounding_skin[surrounding_skin_mask == 0] = 0
        surrounding_skin_hsv = cv2.cvtColor(surrounding_skin, cv2.COLOR_RGB2HSV)
        surrounding_skin_v = surrounding_skin_hsv[:,:,2][surrounding_skin_mask > 0]
        skin_brightness_threshold = np.mean(surrounding_skin_v) + 25  # Threshold above average skin
    else:
        skin_brightness_threshold = 200  # Default if no surrounding skin
    
    # Define color ranges in HSV space
    color_ranges = {
        # White: high value (brightness), low saturation
        "white": {
            "lower": np.array([0, 0, max(200, int(skin_brightness_threshold))]),
            "upper": np.array([180, 30, 255]),
            "display_color": [255, 255, 255]  # RGB for visualization
        },
        # Red: two ranges due to hue wrapping around 0/180
        "red1": {
            "lower": np.array([0, 100, 80]),
            "upper": np.array([10, 255, 255]),
            "display_color": [255, 0, 0]
        },
        "red2": {
            "lower": np.array([160, 100, 80]),
            "upper": np.array([180, 255, 255]),
            "display_color": [255, 0, 0]
        },
        # Light brown: orangish-brown hues with high brightness
        "light brown": {
            "lower": np.array([10, 30, 120]),
            "upper": np.array([25, 255, 255]),
            "display_color": [210, 180, 140]
        },
        # Dark brown: orangish-brown hues with lower brightness
        "dark brown": {
            "lower": np.array([10, 30, 40]),
            "upper": np.array([25, 255, 120]),
            "display_color": [101, 67, 33]
        },
        # Blue-gray: blue to cyan hues
        "blue-gray": {
            "lower": np.array([90, 20, 60]),
            "upper": np.array([130, 150, 180]),
            "display_color": [102, 153, 204]
        },
        # Black: very low brightness regardless of hue/saturation
        "black": {
            "lower": np.array([0, 0, 0]),
            "upper": np.array([180, 255, 40]),
            "display_color": [0, 0, 0]
        }
    }
    
    # Minimum percentage of lesion area to consider a color present
    min_percentage = 3.0  # 3% of lesion area
    
    # Detect each color
    detected_colors = {}
    color_masks = {}
    
    # Process each color
    for color_name, ranges in color_ranges.items():
        # Special case for red (combines two ranges)
        if color_name == "red1":
            mask1 = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
            mask2 = cv2.inRange(hsv_image, color_ranges["red2"]["lower"], color_ranges["red2"]["upper"])
            color_mask = cv2.bitwise_or(mask1, mask2)
            color_name = "red"  # Rename for output
        elif color_name == "red2":
            continue  # Skip, already processed with red1
        else:
            color_mask = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate percentage of lesion with this color
        color_pixels = np.sum(color_mask > 0)
        percentage = (color_pixels / lesion_area) * 100
        
        # Store if above threshold
        if percentage >= min_percentage:
            detected_colors[color_name] = percentage
            color_masks[color_name] = color_mask
            
    # Ensure at least one color is detected (minimum score is 1)
    if not detected_colors:
        # Default to the most dominant color if none meet the threshold
        max_pixels = 0
        dominant_color = "light brown"  # Default
        
        for color_name, ranges in color_ranges.items():
            if color_name == "red2":
                continue  # Skip, already processed with red1
                
            if color_name == "red1":
                mask1 = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
                mask2 = cv2.inRange(hsv_image, color_ranges["red2"]["lower"], color_ranges["red2"]["upper"])
                color_mask = cv2.bitwise_or(mask1, mask2)
                color_name = "red"  # Rename for output
            else:
                color_mask = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
                
            color_pixels = np.sum(color_mask > 0)
            if color_pixels > max_pixels:
                max_pixels = color_pixels
                dominant_color = color_name
                dominant_mask = color_mask
                
        detected_colors[dominant_color] = (max_pixels / lesion_area) * 100
        color_masks[dominant_color] = dominant_mask
    
    # Count number of colors (score)
    color_score = len(detected_colors)
    
    # Print detected colors and percentages
    print(f"\nDetected {color_score} colors in the lesion:")
    for color, percentage in detected_colors.items():
        print(f"  - {color}: {percentage:.2f}% of lesion area")
    
    # Visualization
    if show_graph:
        # Create visualization of detected colors
        fig = plt.figure(figsize=(15, 8))
        
        # Original image
        ax1 = fig.add_subplot(1, len(detected_colors) + 1, 1)
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # Display each detected color
        for i, (color_name, percentage) in enumerate(detected_colors.items(), 1):
            # Create color overlay
            color_overlay = np.zeros_like(image)
            
            # Get display color
            if color_name == "red":
                display_color = color_ranges["red1"]["display_color"]
            else:
                display_color = color_ranges[color_name]["display_color"]
                
            # Apply color to mask area
            color_mask = color_masks[color_name]
            for c in range(3):
                color_overlay[:,:,c][color_mask > 0] = display_color[c]
            
            # Blend with original image for better visualization
            alpha = 0.7  # Transparency
            blended = cv2.addWeighted(image, 1-alpha, color_overlay, alpha, 0)
            
            # Show in subplot
            ax = fig.add_subplot(1, len(detected_colors) + 1, i + 1)
            ax.imshow(blended)
            ax.set_title(f"{color_name.capitalize()}: {percentage:.1f}%")
            ax.axis("off")
        
        plt.suptitle(f"Color Analysis: {color_score} colors detected", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return color_score
