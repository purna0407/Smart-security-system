import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def apply_high_pass_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def apply_edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_filtered = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel_filtered)

def apply_unsharp_masking(image, kernel_size=(9, 9), sigma=10.0, amount=1.5):
    gaussian_blur = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, amount, gaussian_blur, 1 - amount, 0)

def saveFilteredImg(ImgName, imgObj, filter_type):
    output_folder = os.path.join("Filtered_Images", filter_type)
    os.makedirs(output_folder, exist_ok=True)
    img_output_path = os.path.join(output_folder, ImgName)
    cv2.imwrite(img_output_path, imgObj)

def process(imagepath):
    original_image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    assert original_image is not None, "File could not be read, check with os.path.exists()"

    median_filtered_image = apply_median_filter(original_image)
    high_pass_filtered_image = apply_high_pass_filter(original_image)
    histogram_equalized_image = histogram_equalization(original_image)
    edge_detected_image = apply_edge_detection(original_image)
    sobel_filtered_image = apply_sobel_filter(original_image)
    unsharp_masked_image = apply_unsharp_masking(original_image)

    image_name = os.path.basename(imagepath)
    saveFilteredImg('original_' + image_name, original_image, 'Original')
    saveFilteredImg('median_filtered_' + image_name, median_filtered_image, 'Median_Filtered')
    saveFilteredImg('high_pass_filtered_' + image_name, high_pass_filtered_image, 'High_Pass_Filtered')
    saveFilteredImg('histogram_equalized_' + image_name, histogram_equalized_image, 'Histogram_Equalized')
    saveFilteredImg('edge_detected_' + image_name, edge_detected_image, 'Edge_Detected')
    saveFilteredImg('sobel_filtered_' + image_name, sobel_filtered_image, 'Sobel_Filtered')
    saveFilteredImg('unsharp_masked_' + image_name, unsharp_masked_image, 'Unsharp_Masked')

if __name__ == "__main__":
    process(r"C:\Users\racha\OneDrive\Desktop\DIP\captured_images\image_1712892236.jpg")
