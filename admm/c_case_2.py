import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt
from utils import get_project_root
from tqdm import tqdm

m = 4           # num of original pixels
n = 2           # num of interpolated pixels
gamma = 0.4     # interpolator parameter
kappa = 0.2     # denoiser parameter
rho = 0.7       # ADMM parameter

img_path = str(get_project_root()) + "/img/lena_gray.png"

img = plt.imread(img_path)
plt.imshow(img, cmap="gray")
plt.show()

# 0. Define delection matrices
H = np.zeros((m, m + n))
for i in range(m):
    H[i, i] = 1
HT = H.transpose()

G = np.zeros((n, m+n))
for i in range(n):
    G[i, m+i] = 1
GT = G.transpose()

# 1. Define interpolator
theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])      # must be full row-rank
thetaT = theta.transpose()
Amn = 2 * mul(thetaT, inv(mul(theta, thetaT)))
A = np.zeros((m + n, m + n))
A[0:m, m:m + n] = Amn
AT = A.transpose()

# 2. Define denoiser
psi = [[0.4375, 0.5625], [0.5625, 0.4375]]      # must be non-negative, symmetric, doubly-stochastic, invertible
Lbar = (inv(psi) - np.identity(n))/kappa

h, w = img.shape
linear_filtered_img = np.zeros((h, int(w + w/2)))
admm_filtered_img = np.zeros((h, int(w + w/2)))

for i in tqdm(range(int(h / 2))):
    for j in range(int(w / 2)):
        row = i * 2
        col = j * 2
        img_patch = img[row:row + 2, col:col + 2]
        flattened_patch = np.reshape(img_patch, (4, 1))

        # 3. compute linear system solution
        coeff_linear = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H)) + kappa * mul(mul(GT, Lbar), G)
        x0 = mul(mul(inv(coeff_linear), HT), flattened_patch) * (1 + gamma)
        x0_reshaped = np.zeros((2, 3))
        x0_reshaped[0, 0] = x0[0, 0]
        x0_reshaped[0, 1] = x0[4, 0]
        x0_reshaped[0, 2] = x0[1, 0]
        x0_reshaped[1, 0] = x0[2, 0]
        x0_reshaped[1, 1] = x0[5, 0]
        x0_reshaped[1, 2] = x0[3, 0]
        linear_filtered_img[row:row + 2, j * 3:j * 3 + 3] = x0_reshaped

        # 4. Implement ADMM iterations
        done = False
        iter = 0
        lamda = np.zeros((m + n, 1))
        coeff_m = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H))
        z = mul(mul(inv(coeff_m), HT), flattened_patch) * (1 + gamma)

        while not done:
            iter = iter + 1

            # compute x
            coeff_x = mul(HT, H) + kappa * mul(mul(GT, Lbar), G) + (rho / 2) * np.identity(m + n)
            x = mul(inv(coeff_x), (mul(HT, flattened_patch) - lamda/2 + (rho/2) * z))

            # compute z
            coeff_z = gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H)) + (rho / 2) * np.identity(m + n)
            z = mul(inv(coeff_z), (lamda/2 + (rho/2) * x))

            # update lamda
            lamda = lamda + rho * (x - z)

            if np.sum(np.abs(x - z)) < 0.001:
                done = True
            if iter > 100:
                print("large iter")
                done = True

        x_norm = x * (1 + gamma)
        x_norm_reshaped = np.zeros((2, 3))
        x_norm_reshaped[0, 0] = x_norm[0, 0]
        x_norm_reshaped[0, 1] = x_norm[4, 0]
        x_norm_reshaped[0, 2] = x_norm[1, 0]
        x_norm_reshaped[1, 0] = x_norm[2, 0]
        x_norm_reshaped[1, 1] = x_norm[5, 0]
        x_norm_reshaped[1, 2] = x_norm[3, 0]
        admm_filtered_img[row:row + 2, j * 3:j * 3 + 3] = x_norm_reshaped

plt.imshow(linear_filtered_img, cmap="gray")
plt.show()

plt.imshow(admm_filtered_img, cmap="gray")
plt.show()

difference = admm_filtered_img - linear_filtered_img
ssd1 = np.sum(difference ** 2)
print(ssd1)

print("hi")