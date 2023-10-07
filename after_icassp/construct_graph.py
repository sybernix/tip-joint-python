import numpy as np
import matplotlib.pyplot as plt

def getXYMaps(originalImage, angle_degrees):
    rows, cols = originalImage.shape[:2]
    alpha = -np.deg2rad(angle_degrees)  # Negative because the y-axis is reversed in image coordinates

    # Rotation matrix
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])

    # Create a grid for the original image
    X, Y = np.meshgrid(np.arange(1, cols+1), np.arange(1, rows+1))

    # Apply the inverse rotation to the grid
    coords = np.linalg.solve(R, np.vstack([(X - cols/2).ravel(), (Y - rows/2).ravel()]))

    x_rot = (coords[0] + cols/2).reshape(rows, cols)
    y_rot = (coords[1] + rows/2).reshape(rows, cols)

    return x_rot, y_rot

# Define image dimensions
width = 10
height = 10

# Create a 10x10 matrix filled with ones (white)
img_patch = np.ones((height, width))

# Set the left half to black (zero)
img_patch[:, :width//2] = 0

# Display the image patch
plt.imshow(img_patch, cmap='gray')
plt.show()

# Add Gaussian noise
noise_mean = 0
noise_variance = 0.01  # Adjust this value according to the desired noise level
noisy_img_patch = img_patch + np.random.normal(noise_mean, np.sqrt(noise_variance), img_patch.shape)

# Clip values outside [0, 1] range if any due to noise
noisy_img_patch = np.clip(noisy_img_patch, 0, 1)

# Display the noisy image patch
plt.imshow(noisy_img_patch, cmap='gray')
plt.show()
