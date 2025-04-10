import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2 

def calculate_border(image: np.ndarray, mask: np.ndarray, *, show_graph: bool = False) -> float:
    """Calculate the border score for the lesion according to ABCD rule.

    The border score is evaluated through the presence of sharp distinct edges or gradual indistinct edges.
    Legions are divided into 8 segments, and each segment is processed using Canny Edge Detection.
    Distinct edges are given a score of 1 and indistinct edges are given a score of 0.

    Args:
        image: RGB image of the lesion as a NumPy array.
        mask: Binary mask (same size as image) delineating the lesion.
        show_graph: If True, display visualization graph for detected border (weak and strong edges).

    Returns:
        TDS contribution score for border classification.

    """
    # Set legion image to grayscale
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set lesion image as 8 bit
    newimage = grayimage.astype(np.uint8)

    # Slice image into 8 segments to prepare for edge detection 
    height, width = newimage.shape
    image_segments = []
    for i in range(0, height, height // 2):
        for j in range(0, width, width // 2):
            segment = newimage[i:i + height // 2, j:j + width // 2]
            image_segments.append(segment)


    # Bug fixing and checking rn! 
    # Check if all legion borders are distinct or indistinct and return a min score/max score
    image_edges = cv2.Canny(newimage, 20, 100)
    # Compute perimeter to area ratio for edges found
    image_contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = cv2.contourArea(image_contours[0][0])

    # Visualize the border detection of each segment 
    plt.imshow(image_edges,cmap = 'gray')
    plt.show()

    # Compare against border threshold value to differentiate distinct VS indistinct edges
    border_threshold = 0.2
    if contour_area > border_threshold * ((image.shape[0]*image.shape[1])):
        # Distinct border in this segment found, return score of +1
        print("Found distinct edge")
    else:
         # Indistinct border in this segment found, return score of 0
        print("No distinct edge")

    # Calculate the edges and obtain the score for each segment
    # segment_score = [segment_edge_detection(segment) for segment in image_segments]
    border_score = 0
    for segment in image_segments:
        segment_score = segment_edge_detection(segment)
        # Add up the score of each segment for total border score 
        border_score += segment_score
         
    # Normalize the border score between (0,1) as contribution score for TDS 
    tds_score = border_score / 8

    return tds_score

def segment_edge_detection (image_segment):
        """ 
        Implement canny edge detection in 1 segment of the legion image.
        Returns 1 if distinct edges found, returns 0 if no distinct edges found.
        """
        # Find high gradient edges of segmented image
        # Lower threshold (for weak edges): 20
        # Higher threshold (for definite edges): 100 
        image_edges = cv2.Canny(image_segment, 20, 150)
        # Compute perimeter to area ratio for edges found
        image_contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(image_contours[0][0])

        # Visualize the border detection of each segment 
        plt.imshow(image_edges, cmap = 'gray')
        plt.show()

        # Compare against border threshold value to differentiate distinct VS indistinct edges
        border_threshold = 0.2
        if contour_area > border_threshold * ((image_segment.shape[0]*image_segment.shape[1])):
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
    #img = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024306.jpg")
    # mask = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024306_segmentation.png")

    # Placeholder image 
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)

    # Calculate color score.
    score = calculate_border(img, mask, show_graph=True)
    print("TDS Border Contribution score:", score)

    # Exit gracefully.
    sys.exit(0)