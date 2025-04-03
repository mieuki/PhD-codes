#%%
import numpy as np
import scipy as sp
from scipy.signal import convolve2d
from scipy.fft import fft, fftfreq, fft2, fftshift
from scipy.fft import ifft, ifft2, ifftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import os

#%%
#Converting images to grayscale and saving in a folder

# Path to the folder containing images
folder_path = r"C:\Users\Muskan\Desktop\collimator\data\tiff_files\nef_files"

""" # Create a folder to save grayscale images
output_folder = os.path.join(folder_path, 'grayscale_images')
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif') or filename.endswith('.tif'):
        # Read the image
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, gray_img)

print("All images have been converted to grayscale and saved.")
 """
# %%
# Knife edge targets analysis

# Path to the image file
image_path = r"C:\Users\Muskan\Desktop\collimator\data\tiff_files\grayscale_images\Knife_edge_target\DSC_0099.tif"

# Read the image
image = cv2.imread(image_path)
image = image.astype(np.float32)/255.0 #convert to float and normalize

# Display the original image 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Image dimensions (height, width, channels)
#image.shape = height, width, channels
#print(f"Image dimensions: Height={height}, Width={width}, Channels={channels}")

# Define the region of interest (ROI) - (x, y, width, height)
roi = (2170, 800, 3000, 3000)

# Crop the region of interest from the image
x, y, w, h = roi
cropped_image = image[y:y+h, x:x+w]

# Display the cropped grayscale region
plt.imshow(cropped_image, cmap='gray')
plt.title('Cropped Region')
plt.show()


""" # %%
# Function to detect edges using the Canny edge detection algorithm with configurable thresholds
def detect_edges(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# Set the thresholds for Canny edge detection
low_threshold = 100
high_threshold = 200

# Detect edges in the cropped image
edges = detect_edges(cropped_image, low_threshold, high_threshold)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.show()

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    edge_region = cropped_image[y:y+h, x:x+w]

    #display
    plt.imshow(edge_region, cmap='gray')
    plt.title('Detected Edge Region')
    plt.show()
else:
    print("No contours found.")
# Find the largest contour , assuming it is the edge we are interested in 
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
edge_region = cropped_image[y:y+h, x:x+w]

#display
plt.imshow(edge_region, cmap='gray')
plt.title('Detected Edge Region')
plt.show()

 """
# %%

# Calculate the line spread function (ESF) from the edge region
def calculate_esf(edge_region):
    # Calculate the sum of pixel values along the vertical axis
    esf = np.mean(edge_region, axis=0)
    return esf 

# Calculate LSF from the edge region
esf = calculate_esf(cropped_image)

# Plot the ESF
plt.plot(esf)
plt.title('Edge Spread Function (LSF)')
plt.xlabel('Pixel Position')
plt.ylabel('Pixel Intensity')
plt.show()

# %%
def calculate_lsf(esf):
    # Calculate the derivative of the ESF to obtain the LSF
    lsf = np.diff(esf)
    return lsf

lsf = calculate_lsf(esf)

plt.plot(lsf)
plt.title('Line Spread Function (LSF)')
plt.show()
# %%
def calculate_mtf(esf):
    # Calculate the MTF by taking the Fourier Transform of the LSF
    mtf = np.abs(np.fft.fft(esf))
    return mtf 
mtf = calculate_mtf(esf) 
frequencies = np.fft.fftfreq(len(mtf))
plt.plot(frequencies[:len(frequencies)//2], mtf[:len(mtf)//2])
plt.title('Modulation Transfer Function (MTF)')
plt.xlabel('Spatial Frequency')
plt.ylabel('MTF')
plt.show()
# %%

#%%
from PIL import Image
import numpy as np

# Open the image
img = Image.open(r"C:\Users\Muskan\Desktop\collimator\data\tiff_files\grayscale_images\Knife_edge_target\DSC_0099.tif")

# Get image dimensions
width, height = img.size
print(f"Image size: {width} x {height}")

# Extract pixel values using getpixel()
x, y = 100, 50  # Example coordinates
pixel_value_getpixel = img.getpixel((x, y))
print(f"Pixel value at ({x}, {y}) using getpixel(): {pixel_value_getpixel}")

# Extract pixel values using load()
pixels = img.load()
pixel_value_load = pixels[x, y]
print(f"Pixel value at ({x}, {y}) using load(): {pixel_value_load}")

# Convert image to NumPy array
pixel_array = np.array(img)
print(f"Shape of pixel array: {pixel_array.shape}")
print(f"Pixel value at ({x}, {y}) in array: {pixel_array[y, x]}")

# Example of iterating through pixels (printing a few for demonstration)
print("Sample of pixel values:")
for y in range(min(10, height)):
    for x in range(min(10, width)):
        print(f"Pixel at ({x}, {y}): {img.getpixel((x, y))}", end=" ")
    print()
