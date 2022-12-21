import numpy as np
from numpy import matmul as mul

def linear_multiple_interpolate(img):
    theta1 = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    theta2 = np.array([[0.5, 0, 0.5, 0], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0, 0.5]])

    h, w = img.shape
    interpolated_img = np.zeros((2 * h - 1, 2 * w - 1))

    for row in range(int(h - 1)):
        for col in range(int(w - 1)):
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            interp1 = mul(theta1, flattened_patch)
            interp2 = mul(theta2, flattened_patch)
            interpolated_patch = np.zeros((3, 3))

            interpolated_patch[0, 0] = img_patch[0, 0]
            interpolated_patch[0, 2] = img_patch[0, 1]
            interpolated_patch[2, 0] = img_patch[1, 0]
            interpolated_patch[2, 2] = img_patch[1, 1]

            interpolated_patch[0, 1] = interp1[0]
            interpolated_patch[2, 1] = interp1[1]

            interpolated_patch[1, 0] = interp2[0]
            interpolated_patch[1, 1] = interp2[1]
            interpolated_patch[1, 2] = interp2[2]

            interpolated_img[row * 2:row * 2 + 3, col * 2:col * 2 + 3] = interpolated_patch
    return interpolated_img
