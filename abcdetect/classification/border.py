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

    # Ensure image and mask are the same size
    if image.shape[:2] != mask.shape[:2]:
        target_shape = (mask.shape[1], mask.shape[0])
        image = cv2.resize(image, target_shape)

    # Expand binary mask to 3 channels
    binary_mask = mask > 0.5
    binary_mask_3c = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    # Apply mask to image
    maskedimage = image * binary_mask_3c 

    # Set legion image to grayscale
    grayimage = cv2.cvtColor(maskedimage, cv2.COLOR_BGR2GRAY)

    # Set lesion image as 8 bit
    newimage = grayimage.astype(np.uint8)

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
    
    # Visualization of the border detection procedure 
    if show_graph:
            # Adjust thresholds to be the same values as the segment edge detection function
            low_threshold = 70
            high_threshold = 150 
            cannyedges = cv2.Canny(newimage, low_threshold, high_threshold, 1, L2gradient= True)
            # Plot images all on the same graph with titles
            fig, axes = plt.subplots(1, 4, figsize=(12, 4), dpi = 80)
            axes[0].imshow(image)
            axes[0].set_title('Original Image', fontsize= 'x-large')
            axes[0].axis('off')
            axes[1].imshow(maskedimage)
            axes[1].set_title('Mask Overlay Image', fontsize= 'x-large')
            axes[1].axis('off')
            axes[2].imshow(grayimage, cmap='gray')
            axes[2].set_title('Grayscale Image', fontsize= 'x-large')
            axes[2].axis('off')
            axes[3].imshow(cannyedges, cmap='gray')
            axes[3].set_title('Canny Edge Detection', fontsize= 'x-large')
            axes[3].axis('off')
            fig.text(0.5, 0.1, 'Border Score: {}'.format(border_score), fontsize= 'large', horizontalalignment='center', wrap=True ) 
            plt.tight_layout()
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
        high_threshold = 150
        image_edges = cv2.Canny(image_segment, low_threshold, high_threshold, 1, L2gradient= True)

        # Compute perimeter to area ratio for segments + edges found
        image_contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(image_contours[0])

        # Compare against border threshold value to differentiate distinct VS indistinct edges
        # Border threshold value should be adjusted based on validated border strength 
        border_threshold = 2.5
        if contour_area > 0 and contour_area < border_threshold:
            # Strong distinct border in this segment found
            return 2
        elif contour_area == border_threshold:
           # Weak distinct border in this segment found
           return 1
        else:
            # Indistinct border in this segment found
            return 0
    
def rgb2gray (image):
    """ Set RGB image of legion to grayscale for edge detection"""
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Run as demo script 
if __name__ == '__main__':
    import sys

    # Test 1 
    # Opencv reads image in BGR -> must convert to RGB  
    # img = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024545.jpg")[:,:,::-1]
    # mask = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024545_segmentation.png")[:,:,::-1]

    # Test 2 
    # img2 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024644.jpg")[:,:,::-1]
    # mask2 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024644_segmentation.png")[:,:,::-1]
    
    # Test 3 
    # img3 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024732.jpg")[:,:,::-1]
    # mask3 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024732_segmentation.png")[:,:,::-1]
    
    # Test 4 
    # img4 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024323.jpg")[:,:,::-1]
    # mask4 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024323_segmentation.png")[:,:,::-1]
    
    # Test 5 
    # img5 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024349.jpg")[:,:,::-1]
    # mask5 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024349_segmentation.png")[:,:,::-1]

    # Test 6 
    # img6 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024563.jpg")[:,:,::-1]
    # mask6 = cv2.imread(r"D:\School\Classes\CMPT419\project\Skin Lesion Sample Set-20250410T015009Z-001\Skin Lesion Sample Set\ISIC_0024563_segmentation.png")[:,:,::-1]
    
    # Placeholder image 
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)[:,:,::-1]
    mask = np.zeros((256, 256), dtype=np.uint8)[:,:,::-1]

    # Calculate border score for each demo lesion.
    score = calculate_border_score(img, mask, show_graph=True)
    print("TDS Border Contribution score:", score)

    # Additional Border Score Tests
    # score2 = calculate_border_score(img2, mask2, show_graph=True)
    # print("TDS Border Contribution score:", score2)
    # score3 = calculate_border_score(img3, mask3, show_graph=True)
    # print("TDS Border Contribution score:", score3)
    # score4 = calculate_border_score(img4, mask4, show_graph=True)
    # print("TDS Border Contribution score:", score4)
    # score5 = calculate_border_score(img5, mask5, show_graph=True)
    # print("TDS Border Contribution score:", score5)
    # score6 = calculate_border_score(img6, mask6, show_graph=True)
    # print("TDS Border Contribution score:", score6)

    sys.exit(0)