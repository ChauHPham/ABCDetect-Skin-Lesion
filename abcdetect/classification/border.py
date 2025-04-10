import numpy as np
import matplotlib.pyplot as plt
import cv2 

def calculate_border_score(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> int:
    """Calculate the border score for the lesion according to ABCD rule.

    The border score is evaluated through the presence of sharp distinct edges or gradual indistinct edges.
    Legions are divided into 8 segments, and each segment is processed using Canny Edge Detection.
    Distinct edges are given a score of 1 and indistinct edges are given a score of 0.
    The maximum border score is 8 and minimum border score is 0, which is normalized by the number of quadrants
    to calculate the final TDS contribution score. 

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization graph for detected border (weak and strong edges).

    Returns:
        Border score (0 to 8) based on the presence of distinct edges.
    """
    # Optional: Segmented mask may interfere with border detection
    # Overlay binary mask on top of RGB legion image
    image = cv2.bitwise_and(image,mask)
    
    # Visualize image with mask on 
    cv2.imshow("AND", image)

    # Set legion image to grayscale
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set lesion image as 8 bit
    newimage = grayimage.astype(np.uint8)

    # Greyscale image of segmented legion
    cv2.imshow("AND", newimage)

    # Slice image into 8 segments to evaluate border score for each 
    # Height and width are floats 
    height, width = newimage.shape
    image_segments = []
    for i in range(0, height, height // 2):
        for j in range(0, width, width // 2):
            segment = newimage[i:i + height // 2, j:j + width // 2]
            image_segments.append(segment)
    
    # Calculate the edges and obtain the score for each segment
    border_score = 0
    for segment in image_segments:
        segment_score = segment_edge_detection(segment)
        # Add up the score of each segment for total border score 
        border_score += segment_score
    
    if show_graph:
            # Visualize the whole image after canny edge detection
            # Adjust thresholds to be the same values as the segment edge detection function
            low_threshold = 70
            high_threshold = 100 
            cannyedges = cv2.Canny(newimage, low_threshold, high_threshold)
            plt.imshow(cannyedges, cmap = 'gray')
            plt.show()
         
    return border_score


def segment_edge_detection (image_segment):
        """ 
        Implement canny edge detection in 1 segment of the legion image.
        Returns 1 if distinct edges found, returns 0 if no distinct edges found.
        """
        # Find high gradient edges of segmented image
        # Adjust lower and higher thresholds for stronger edges 
        low_threshold = 70
        high_threshold = 100 
        image_edges = cv2.Canny(image_segment, low_threshold, high_threshold)

        # Compute perimeter to area ratio for segments + edges found
        image_contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(image_contours[0])

        # Compare against border threshold value to differentiate distinct VS indistinct edges
        # Border threshold value should be adjusted based on validated border strength 
        border_threshold = 0.2
        if contour_area >= border_threshold:
            # Distinct border in this segment found, return score of +1
            return 1
        else:
            # Indistinct border in this segment found, return score of 0
            return 0
    
def rgb2gray (image):
    """ Set RGB image of legion to grayscale for edge detection"""
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Run as demo script 
if __name__ == '__main__':
    import sys

    # Testing image 
    img = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024306.jpg")
    mask = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024306_segmentation.png")

    # Placeholder image 
    # img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # mask = np.zeros((256, 256), dtype=np.uint8)

    # Calculate color score.
    score = calculate_border_score(img, mask, show_graph=True)
    print("TDS Border Contribution score:", score)

    sys.exit(0)