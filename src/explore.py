import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
img_path = "data/raw/matthew_gels_2/matthew_gels_2/images/10.tif"
mask_path = "data/raw/matthew_gels_2/matthew_gels_2/masks/10.tif"

# Load
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Show
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Gel Image")
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Band Mask")
plt.tight_layout()
plt.savefig("reports/first_look.png")
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Saved to reports/first_look.png")
