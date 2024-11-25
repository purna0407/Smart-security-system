import cv2 as cv
import numpy as np
import os
from datetime import datetime

D0_highpass = 50  # Cut-off frequency for High-Pass Filter

# Apply High-Pass Filter using Ideal Filter (FFT)
def apply_high_pass_filter(image, D0_highpass):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    hpf_mask = np.ones((rows, cols), np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist <= D0_highpass:
                hpf_mask[i, j] = 0

    hpf_filtered = fshift * hpf_mask
    hpf_inverse_shifted = np.fft.ifftshift(hpf_filtered)
    restored_image = np.fft.ifft2(hpf_inverse_shifted)
    return np.abs(restored_image)

# Perform Histogram Equalization
def histogram_equalization(image):
    image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv.equalizeHist(image_yuv[:,:,0])  # Equalize the luminance channel
    equalized_image = cv.cvtColor(image_yuv, cv.COLOR_YUV2BGR)
    return equalized_image

# Apply Median Filter
def apply_median_filter(image, filter_size=5):
    return cv.medianBlur(image, filter_size)

# Apply Sobel Filter for edge detection
def apply_sobel_filter(image):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    return cv.magnitude(sobelx, sobely)

# Apply Edge Detection (Canny)
def apply_edge_detection(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)
    ret, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lower = 0.5 * ret
    upper = 1.5 * ret
    canny = cv.Canny(image, lower, upper)
    return canny

# Apply Unsharp Masking
def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Save filtered images
def saveFilteredImg(ImgName, imgObj, filter_type):
    output_folder = os.path.join("Filtered_Images", filter_type)
    os.makedirs(output_folder, exist_ok=True)
    img_output_path = os.path.join(output_folder, ImgName)
    cv.imwrite(img_output_path, imgObj)

# Process the image and apply various filters
def process(imagepath):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_image = cv.imread(imagepath)
    original_image_RGB = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    gray_img = cv.imread(imagepath, cv.IMREAD_GRAYSCALE)
    
    assert gray_img is not None, "File could not be read, check with os.path.exists()"
    
    D0_highpass = 50  # Cut-off frequency for High-Pass Filter
    
    # Apply high-pass filter
    hpf_filtered_image = apply_high_pass_filter(gray_img, D0_highpass)
    
    # Perform histogram equalization
    histogram_equalized_image = histogram_equalization(original_image)
    
    # Perform Median Filter
    median_filtered_image = apply_median_filter(gray_img, 5)
    
    # Apply Edge Detection (Canny)
    edge_detected_image = apply_edge_detection(gray_img)
    
    # Apply Sobel Filter
    sobel_filtered_image = apply_sobel_filter(gray_img)
    
    # Apply Unsharp Masking
    unsharp_masked_image = apply_unsharp_mask(original_image)
    
    # Create output directories
    filter_folders = ["original_images", "median_filtered_images", "highpass_images", 
                      "histogram_images", "edge_detected_images", "sobel_images", "unsharp_images"]
    
    for folder in filter_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Save images to corresponding folders
    original_image_path = os.path.join("original_images", f'Original_image_{timestamp}.jpg')
    cv.imwrite(original_image_path, original_image)
    
    # Save filtered images
    saveFilteredImg(f'median_filtered_image_{timestamp}.jpg', median_filtered_image, 'Median_Filtered')
    saveFilteredImg(f'hpf_filtered_image_{timestamp}.jpg', hpf_filtered_image, 'HPF')
    saveFilteredImg(f'histogram_equalized_image_{timestamp}.jpg', histogram_equalized_image, 'Histogram_Equalized')
    saveFilteredImg(f'edge_detected_image_{timestamp}.jpg', edge_detected_image, 'Edge_Detected')
    saveFilteredImg(f'sobel_filtered_image_{timestamp}.jpg', sobel_filtered_image, 'Sobel_Filtered')
    saveFilteredImg(f'unsharp_masked_image_{timestamp}.jpg', unsharp_masked_image, 'Unsharp_Masked')
    
    # Print the output paths
    print(f"Filtered images have been saved to respective folders.")
    
if __name__ == "__main__":
    process(r"C:\path\to\your\image.jpg")  # Replace with the path to your image
