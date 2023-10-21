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
p_w = 10
p_h = 10
angle_degrees = 20

# Create a 10x10 matrix filled with ones (white)
img_patch = np.ones((p_h, p_w))

# Set the left half to black (zero)
img_patch[:, :p_w // 2] = 0

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

mapx, mapy = getXYMaps(noisy_img_patch, angle_degrees)

x_min = int(np.floor(mapx.min()))
y_min = int(np.floor(mapy.min()))

x_max = int(np.ceil(mapx.max()))
y_max = int(np.ceil(mapy.max()))

img_patch = noisy_img_patch[y_min:y_max, x_min:x_max]
ip_h, ip_w = img_patch.shape

theta = np.zeros((p_h * p_w, ip_h * ip_w))

for i in range(p_h):
    for j in range(p_w):
        theta_r = i * p_w + j

        x = mapx[i, j]  - x_min
        y = mapy[i, j] - y_min

        l = int(np.floor(x))
        t = int(np.floor(y))

        a = x - l
        b = y - t

        theta_lt_loc = (t - 1) * ip_w + l
        theta_rt_loc = theta_lt_loc + 1

        theta_lb_loc = t * ip_w + l
        theta_rb_loc = theta_lb_loc + 1

        theta[theta_r, theta_lt_loc] = (1 - b) * (1 - a)
        theta[theta_r, theta_rt_loc] = (1 - b) * a
        theta[theta_r, theta_lb_loc] = b * (1 - a)
        theta[theta_r, theta_rb_loc] = b * a

a = 1
