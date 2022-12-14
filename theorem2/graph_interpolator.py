import numpy as np
from utils import conjgrad
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg

def interpolateGraphFilter(img, use_cg=False):
    m = 4
    n = 2
    gamma = 0.4

    theta = np.array([[1/2, 1/2, 0, 0], [0, 0, 1/2, 1/2]])
    thetaT = theta.transpose()
    H = np.zeros((m, m + n))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[3, 3] = 1
    HT = H.transpose()
    Amn = 2 * np.matmul(thetaT, np.linalg.pinv(np.matmul(theta, thetaT)))
    A = np.zeros((m + n, m + n))
    A[0:m, m:m+n] = Amn
    AT = A.transpose()

    temp = np.matmul(HT, H) + gamma * (np.matmul(HT, H) + np.matmul(np.matmul(np.matmul(AT, HT), H), A)
                                       - 2 * np.matmul(np.matmul(AT, HT), H))

    if use_cg == False:
        p = np.linalg.inv(temp)
        q = np.matmul(p, HT)

    h, w = img.shape
    interpolated_img = np.zeros((h, 2 * w - 1))

    for i in range(int(h / 2)):
        for j in range(int(w-1)):
            row = i * 2
            col = j
            img_patch = img[row:row+2, col:col+2]
            flattened_patch = np.reshape(img_patch, (4, 1))

            b = np.matmul(HT, flattened_patch)

            if use_cg ==  False:
                x = np.matmul(q, flattened_patch)
            else:
                # x = conjgrad(temp, b, np.random.rand(len(temp[0]), 1))
                x, exit_code = cg(temp, b)
                x = np.reshape(x, (6, 1))

            interpolated_patch = x * (1 + gamma)
            interpolated_patch_reshaped = np.zeros((2, 3))
            interpolated_patch_reshaped[0, 0] = interpolated_patch[0, 0]
            interpolated_patch_reshaped[0, 1] = interpolated_patch[4, 0]
            interpolated_patch_reshaped[0, 2] = interpolated_patch[1, 0]
            interpolated_patch_reshaped[1, 0] = interpolated_patch[2, 0]
            interpolated_patch_reshaped[1, 1] = interpolated_patch[5, 0]
            interpolated_patch_reshaped[1, 2] = interpolated_patch[3, 0]
            interpolated_img[row:row+2, col+j:col+j+3] = interpolated_patch_reshaped
    return interpolated_img