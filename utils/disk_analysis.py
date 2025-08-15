import torch
import numpy as np
from typing import Tuple

# -------------------- get statistics of the disk dataset -------------------- #
import cv2
def gain_division(img):
    '''img is an image matrix of shape (H, W), bounded either by (0, 1) or (0, 255)'''
    kernelSize = 30
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    localMax = cv2.morphologyEx(img, cv2.MORPH_CLOSE, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)
    localMin = cv2.morphologyEx(img, cv2.MORPH_OPEN, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)
    
    gainDivision = np.where(localMax == 0, 0, ((img-localMin)/(localMax-localMin)))
    # Clip the values to [0,255]
    gainDivision = np.clip((255 * gainDivision), 0, 255)
    return gainDivision

def find_circle_center(img):
    # Apply Gaussian blur to reduce noise and improve edge detection
    img = cv2.GaussianBlur(img, (5, 5), 2)
    # img = cv2.medianBlur(img, 5)
    
    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.4, minDist=50,
                               param1=100, param2=10, minRadius=8, maxRadius=10)

    if circles is not None:
        # Convert the (x, y, r) coordinates of the detected circles into integers
        # circles = np.round(circles[0, :]).astype("int")

        # Assuming you're detecting only one circle
        x, y, r = circles[0][0]
        return ((x, y), img)
    else:
        return None  # No circles found

def calc_avg_bkg_intensity_from_center(
    img: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float = 11.,
    transition_width: float = 0.,
):
    (img_size, img_size) = img.shape
    (c_x, c_y) = center
    
    inner_radius = outer_radius - transition_width

    # Create a meshgrid for the x, y coordinates
    y, x = np.ogrid[:img_size, :img_size]

    # Calculate the radial distance for each pixel from the center
    r = np.sqrt((y - c_y) ** 2 + (x - c_x) ** 2)

    # Initialize the mask as zeros
    mask = np.zeros((img_size, img_size))

    # Apply the conditions to the mask
    mask[r > outer_radius] = img[r > outer_radius]  # Outside the outer radius, set to 0
    
    # count the number of pixels that aren't in the mask
    n_outside_mask = np.count_nonzero(mask[r > outer_radius])
    
    return torch.tensor(mask, dtype=torch.float).unsqueeze(0), n_outside_mask


def infer_disk_statistics(img: np.ndarray):
    img_gain = gain_division(img).astype(np.uint8)
    (center, img_blur) = find_circle_center(img_gain)
    masked_img, n_outside_mask = calc_avg_bkg_intensity_from_center(img, center, outer_radius=11)
    avg_i_bkg = (torch.sum(masked_img) / n_outside_mask).item()
    return center, avg_i_bkg, img_gain, img_blur, masked_img