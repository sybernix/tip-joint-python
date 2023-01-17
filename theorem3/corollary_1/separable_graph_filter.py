import matplotlib.pyplot as plt
import numpy as np

def graph_denoise_interpolate(img, psi, theta):
    m = 4
    n = 2
    gamma = 0.4

    # initialize denoiser
    # psi = np.full((m, m), 1 / m)
    # psi = np.array([[0.312816957047989, 0.247249820875324, 0.260255415738962, 0.179677806337725],
    #                 [0.247249820875324, 0.247335656093526, 0.278941892119199, 0.226472630911952],
    #                 [0.260255415738962, 0.278941892119199, 0.201335672588002, 0.259467019553837],
    #                 [0.179677806337726, 0.226472630911952, 0.259467019553837, 0.334382543196486]])
    mu = 0.5
    laplacian = (1 / mu) * (np.linalg.inv(psi) - (1 + gamma) * np.identity(m))

    # initialize interpolator
    # theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    thetaT = theta.transpose()
    Amn = 2 * np.matmul(thetaT, np.linalg.inv(np.matmul(theta, thetaT)))
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
    p = np.linalg.inv(temp)
    q = np.matmul(p, HT)

    h, w = img.shape
    filtered_img = np.zeros((h, 2 * w - 1))

    for i in range(int(h / 2)):
        for j in range(int(w - 1)):
            row = i * 2
            col = j
            img_patch = img[row:row + 2, col:col + 2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            interpolated_patch = np.matmul(q, flattened_patch)
            # interpolated_patch = x * (1 + gamma)
            interpolated_patch_reshaped = np.zeros((2, 3))
            interpolated_patch_reshaped[0, 0] = interpolated_patch[0, 0]
            interpolated_patch_reshaped[0, 1] = interpolated_patch[4, 0]
            interpolated_patch_reshaped[0, 2] = interpolated_patch[1, 0]
            interpolated_patch_reshaped[1, 0] = interpolated_patch[2, 0]
            interpolated_patch_reshaped[1, 1] = interpolated_patch[5, 0]
            interpolated_patch_reshaped[1, 2] = interpolated_patch[3, 0]
            filtered_img[row:row + 2, col + j:col + j + 3] = interpolated_patch_reshaped
    return filtered_img
