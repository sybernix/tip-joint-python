import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv as inv
from scipy.sparse.linalg import cg

m = 4           # num of original pixels
n = 2           # num of interpolated pixels
gamma = 0.5     # interpolator parameter
kappa = 0.2     # denoiser parameter
rho = 1000       # ADMM parameter

img_patch = np.array([[153, 156], [157, 158]])
flattened_patch = np.reshape(img_patch, (4, 1))

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
psi = [[0.8, 0.2], [0.2, 0.8]]      # must be non-negative, symmetric, doubly-stochastic, invertible
Lbar = (inv(psi) - np.identity(n))/kappa

# 3. compute linear system solution
coeff_linear = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H)) + kappa * mul(mul(GT, Lbar), G)
# x0 = mul(mul(inv(coeff_linear), HT), flattened_patch) * (1 + gamma)

M21 = coeff_linear[m:m+n, 0:m]
M22 = coeff_linear[m:m+n, m:m+n] # M22 is SPD
x01 = flattened_patch/(1+gamma)
x02_, x02_exit_code = cg(M22, mul(-M21, x01))
x02 = np.reshape(x02_, (n, 1))
x0 = np.concatenate((x01, x02), axis=0) * (1+gamma)

# 4. Implement ADMM iterations
done = False
iter = 0
lamda = np.zeros((m + n, 1))
coeff_m = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H))
z = mul(mul(inv(coeff_m), HT), flattened_patch)

while not done:
    iter = iter + 1

    # compute x
    coeff_x = mul(HT, H) + kappa * mul(mul(GT, Lbar), G) + (rho / 2) * np.identity(m + n)
    # x = mul(inv(coeff_x), (mul(HT, flattened_patch) - lamda/2 + (rho/2) * z))
    x, exit_code = cg(coeff_x, (mul(HT, flattened_patch) - lamda/2 + (rho/2) * z))
    x = np.reshape(x, (m+n, 1))

    if exit_code != 0:
        print("x exit code not zero in iteration: " + str(iter))

    # compute z
    coeff_z = gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H)) + (rho / 2) * np.identity(m + n)
    # z = mul(inv(coeff_z), (lamda/2 + (rho/2) * x))
    M21_z = coeff_z[m:m + n, 0:m]
    M22_z = coeff_z[m:m + n, m:m + n]  # M22 is SPD
    z01 = (lamda[0:m] / 2 + (rho/2) * x[0:m]) / (gamma + rho/2)
    z02_, exit_code_z = cg(M22_z, mul(-M21_z, z01) + lamda[m:m+n]/2 + (rho/2) * x[m:m+n])
    z02 = np.reshape(z02_, (n, 1))
    z = np.concatenate((x01, x02), axis=0)
    # z, exit_code_z = cg(coeff_z, (lamda/2 + (rho/2) * x))
    # z = np.reshape(z, (m+n, 1))

    if exit_code_z != 0:
        print("z exit code not zero in iteration: " + str(iter))

    # update lamda
    lamda = lamda + rho * (x - z)

    if np.sum(np.abs(x - z)) < 0.001 or iter > 10000:
        done = True

x_norm = x * (1+ gamma)

print("hi")