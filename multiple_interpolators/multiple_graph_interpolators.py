import numpy as np
from utils import get_project_root
from numpy import matmul as mul

def graph_multiple_interpolate(img, m, n1, n2, gamma):
    H = np.zeros((m, m + n1 + n2))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[3, 3] = 1
    HT = H.transpose()

    H1 = H / np.sqrt(2)
    H1T = H1.transpose()

    H2 = H / np.sqrt(2)
    H2T = H1.transpose()

    theta1 = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    theta1T = theta1.transpose()

    A1mn = 2 * mul(theta1T, np.linalg.pinv(mul(theta1, theta1T)))
    A1 = np.zeros((m + n1 + n2, m + n1 + n2))
    A1[0:m, m:m + n1] = A1mn
    A1T = A1.transpose()

    temp1 = mul(H1T, H1) - 2 * mul(mul(A1T, H1T), H1) + mul(mul(mul(A1T, H1T), H1), A1)

    theta2 = np.array([[0.5, 0, 0.5, 0], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0, 0.5]])
    theta2T = theta2.transpose()

    A2mn = 2 * mul(theta2T, np.linalg.pinv(mul(theta2, theta2T)))
    A2 = np.zeros((m + n1 + n2, m + n1 + n2))
    A2[0:m, m + n1:m + n1 + n2] = A2mn
    A2T = A2.transpose()

    temp2 = mul(H2T, H2) - 2 * mul(mul(A2T, H2T), H2) + mul(mul(mul(A2T, H2T), H2), A2)

    coeff = mul(HT, H) + gamma * (temp1 + temp2)
    inv_coeff = np.linalg.pinv(coeff)

    h, w = img.shape
    interpolated_img = np.zeros((2 * h - 1, 2 * w - 1))

    for row in range(int(h - 1)):
        for col in range(int(w - 1)):
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            x = mul(inv_coeff, mul(HT, flattened_patch))
            interpolated_patch = x * (1 + gamma)
            interpolated_patch_reshaped = np.zeros((3, 3))
            interpolated_patch_reshaped[0, 0] = interpolated_patch[0, 0]
            interpolated_patch_reshaped[0, 2] = interpolated_patch[1, 0]
            interpolated_patch_reshaped[2, 0] = interpolated_patch[2, 0]
            interpolated_patch_reshaped[2, 2] = interpolated_patch[3, 0]

            interpolated_patch_reshaped[0, 1] = interpolated_patch[4, 0]
            interpolated_patch_reshaped[2, 1] = interpolated_patch[5, 0]

            interpolated_patch_reshaped[1, 0] = interpolated_patch[6, 0]
            interpolated_patch_reshaped[1, 1] = interpolated_patch[7, 0]
            interpolated_patch_reshaped[1, 2] = interpolated_patch[8, 0]

            interpolated_img[row * 2:row * 2 + 3, col * 2:col * 2 + 3] = interpolated_patch_reshaped
    return interpolated_img