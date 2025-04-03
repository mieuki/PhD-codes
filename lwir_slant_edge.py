#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy as sp
from scipy.ndimage import gaussian_filter

# path to the knife edge folder
#image_path = r"C:\Users\Muskan\Desktop\collimator\data\lwir_tamarisk_data\knife_edge_45C.tif"
#image_path = r"C:\Users\Muskan\Desktop\collimator\data\lwir_tamarisk_data\knife_edge_40C.tif"
#image_path = r"C:\Users\Muskan\Desktop\collimator\data\lwir_tamarisk_data\knife_edge_35C.tif"
#image_path = r"C:\Users\Muskan\Desktop\collimator\data\lwir_tamarisk_data\knife_edge_30C.tif"
image_path = r"C:\Users\Muskan\Desktop\collimator\data\lwir_tamarisk_data\knife_edge_50C.tif"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Rotate the image by 90 degrees
rot_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

plt.imshow(rot_image, cmap='gray')
plt.title('Rotated Image')
plt.show()

# %%
# Crop the image
#roi = (210, 320, 40, 2) #everything else
roi = (230, 320, 40, 2) #50C
x, y, w, h = roi
crop = rot_image[y:y+h, x:x+w]
plt.imshow(crop, cmap='gray')
plt.title('Cropped rot Image')
plt.show()
cropx, cropy = crop.shape
print('Cropped Image Shape:', crop.shape)
print('Cropped Image Size', crop.size)
print('Cropped Image Data Type:', crop.dtype)

# %%
esf = np.mean(crop, axis=0)
smooth_esf = gaussian_filter(esf, sigma=2) 
#plt.plot(esf, color='blue')
plt.plot(smooth_esf, linestyle = '--', marker = '.', color = 'blue', label = 'ESF') 
plt.title('ESF of the cropped image')
plt.xlabel('Pixel')
plt.ylabel('DN')
plt.legend()        
plt.show()

# %%
lsf = np.diff(smooth_esf)
norm_lsf = lsf / np.max(lsf)
plt.plot(norm_lsf, linestyle = '--', marker = '.', color='blue', label = 'LSF Norm')
plt.title('LSF')
plt.ylabel('Normalised differential values')
plt.xlabel('Pixel')
plt.legend()
plt.show()

#%%
mtf = np.fft.fftshift(np.fft.fft((lsf)))
mtf_norm = np.abs(mtf) / np.max(np.abs(mtf))
plt.plot(mtf_norm,  linestyle = '--', marker = '.', color='blue', label = 'MTF')
plt.title('MTF')    
d = 1.0  # Set the sample spacing to 1.0 or the appropriate value
freq = np.fft.fftfreq(len(lsf), d=d)
#plt.plot(np.abs(mtf), color='blue')  
plt.title('MTF')
plt.ylabel('Normalised value')
plt.xlabel('Spatial Frequency')
plt.xlim(18, 30)
plt.legend()
plt.show()
# %%
