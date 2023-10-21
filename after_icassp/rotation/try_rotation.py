import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

def rotate_image_linear(original_image, angle_degrees):
    rows, cols = original_image.shape
    alpha = -np.deg2rad(angle_degrees)  # Negative because y-axis is reversed in image coordinates

    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])  # Rotation matrix

    # Create a grid for the original image
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Apply the inverse rotation to the grid
    coords = np.linalg.inv(R) @ np.vstack([x.flatten() - cols / 2, y.flatten() - rows / 2])
    x_rot = coords[0].reshape(rows, cols) + cols / 2
    y_rot = coords[1].reshape(rows, cols) + rows / 2

    # Use bilinear interpolation
    rotated_image = map_coordinates(original_image, [y_rot, x_rot], order=1, mode='constant', cval=0)

    return rotated_image

# Read the image using matplotlib
img = plt.imread("../../img/lena_gray.png")

# Ensure it's grayscale (this step is necessary if the image has an alpha channel or is RGB)
if img.ndim > 2:
    img = img[:, :, 0]

rotated_img = rotate_image_linear(img, 20)

# Display the rotated image
plt.imshow(rotated_img, cmap='gray')
plt.axis('off')
plt.show()
