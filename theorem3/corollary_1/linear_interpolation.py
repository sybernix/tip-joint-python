import numpy as np
import matplotlib.pyplot as plt


def interpolateLinear(img, theta):
    Im = np.identity(4)

    linear_interpolator = np.concatenate((Im, theta))

    h, w = img.shape
    interpolated_img = np.zeros((h, 2 * w - 1))

    for i in range(int(h / 2)):
        for j in range(int(w-1)):
            row = i * 2
            col = j
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            x = np.matmul(linear_interpolator, flattened_patch)

            x_reshaped = np.zeros((2, 3))
            x_reshaped[0, 0] = x[0, 0]
            x_reshaped[0, 1] = x[4, 0]
            x_reshaped[0, 2] = x[1, 0]
            x_reshaped[1, 0] = x[2, 0]
            x_reshaped[1, 1] = x[5, 0]
            x_reshaped[1, 2] = x[3, 0]
            interpolated_img[row:row+2, col+j:col+j+3] = x_reshaped
    return interpolated_img
