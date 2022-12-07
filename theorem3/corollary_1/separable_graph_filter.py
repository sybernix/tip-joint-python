import matplotlib.pyplot as plt
import numpy as np

def graph_denoise_interpolate(img):
    m = 4
    n = 2
    gamma = 0.4

    # initialize denoiser
    psi = np.full((m, m), 1 / m)
    mu = 0.5
    laplacian = (1 / mu) * (np.linalg.pinv(psi) - (1 + gamma) * np.identity(m))

    # initialize interpolator
    theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    thetaT = theta.transpose()
    Amn = 2 * np.matmul(thetaT, np.linalg.pinv(np.matmul(theta, thetaT)))
    A = np.zeros((m + n, m + n))
    A[0:m, m:m + n] = Amn
    AT = A.transpose()

    H = np.zeros((m, m + n))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[3, 3] = 1
    HT = H.transpose()

    temp = np.matmul(HT, H) + gamma * (np.matmul(HT, H) + np.matmul(np.matmul(np.matmul(AT, HT), H), A)
                                       - 2 * np.matmul(np.matmul(AT, HT), H)) + mu * np.matmul(np.matmul(HT, laplacian),
                                                                                               H)

    h, w = img.shape
    filtered_img = np.zeros((h, 2 * w - 1))

    for i in range(int(h / 2)):
        for j in range(int(w - 1)):
            row = i * 2
            col = j
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            p = np.linalg.pinv(temp)
            q = np.matmul(p, HT)

            x = np.matmul(q, flattened_patch)
            interpolated_patch = x * (1 + gamma)
            interpolated_patch_reshaped = np.zeros((2, 3))
            interpolated_patch_reshaped[0, 0] = interpolated_patch[0, 0]
            interpolated_patch_reshaped[0, 1] = interpolated_patch[4, 0]
            interpolated_patch_reshaped[0, 2] = interpolated_patch[1, 0]
            interpolated_patch_reshaped[1, 0] = interpolated_patch[2, 0]
            interpolated_patch_reshaped[1, 1] = interpolated_patch[5, 0]
            interpolated_patch_reshaped[1, 2] = interpolated_patch[3, 0]
            filtered_img[row:row + 2, col + j:col + j + 3] = interpolated_patch_reshaped
    return filtered_img
