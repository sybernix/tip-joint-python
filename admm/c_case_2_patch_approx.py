import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv as inv
from scipy.sparse.linalg import cg


def getQ(psi, rho, n, a=0.5):
    term = (a ** -1) + rho / 2 - 1
    fa = term ** -1
    fda = (term ** -2) * (a ** -2)
    fdda = 2 * (term ** -3) * (a ** -4) - 2 * (term ** -2) * (a ** -3)
    Q = (fa - a * fda + (a ** 2) * fdda / 2) * np.identity(n) + (fda - a * fdda) * psi + (fdda/2) * mul(psi, psi)
    return Q


m = 4  # num of original pixels
n = 2  # num of interpolated pixels
gamma = 0.4  # interpolator parameter
kappa = 0.5  # denoiser parameter
rho = 10000  # ADMM parameter

img_patch = np.array([[153, 156], [157, 158]])
flattened_patch = np.reshape(img_patch, (4, 1))

# 0. Define delection matrices
H = np.zeros((m, m + n))
for i in range(m):
    H[i, i] = 1
HT = H.transpose()

G = np.zeros((n, m + n))
for i in range(n):
    G[i, m + i] = 1
GT = G.transpose()

# 1. Define interpolator
theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])  # must be full row-rank
thetaT = theta.transpose()
Amn = 2 * mul(thetaT, inv(mul(theta, thetaT)))
A = np.zeros((m + n, m + n))
A[0:m, m:m + n] = Amn
AT = A.transpose()

# 2. Define denoiser
psi = np.array([[0.8, 0.2], [0.2, 0.8]])  # must be non-negative, symmetric, doubly-stochastic, invertible
Lbar = (inv(psi) - np.identity(n)) / kappa

# 3. compute linear system solution
coeff_linear = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H)) + kappa * mul(
    mul(GT, Lbar), G)
# x0 = mul(mul(inv(coeff_linear), HT), flattened_patch) * (1 + gamma)

M21 = coeff_linear[m:m + n, 0:m]
M22 = coeff_linear[m:m + n, m:m + n]  # M22 is SPD
x01 = flattened_patch / (1 + gamma)
x02_, x02_exit_code = cg(M22, mul(-M21, x01))
x02 = np.reshape(x02_, (n, 1))
x0 = np.concatenate((x01, x02), axis=0) * (1 + gamma)

# 4. Implement ADMM iterations
done = False
iter = 0
lamda = np.zeros((m + n, 1))
coeff_m = mul(HT, H) + gamma * (mul(HT, H) + mul(mul(mul(AT, HT), H), A) - 2 * mul(mul(AT, HT), H))
z = mul(mul(inv(coeff_m), HT), flattened_patch)

coeff_z = np.zeros((m + n, m + n))
coeff_z[:m, :m] = np.identity(m) / gamma
coeff_z[m:m + n, :m] = theta / gamma
coeff_z[m:m + n, m:m + n] = mul(theta, thetaT) / (4 * gamma)

# x = np.zeros((m+n, 1)) + 1
x = z
Q = getQ(psi, rho, n)

while not done:
    iter = iter + 1

    # compute x
    x_m = (1 / (1 + rho / 2)) * (flattened_patch - lamda[:m] / 2 + (rho / 2) * z[:m])
    # x_n = mul(psi, (-lamda[m:m + n] / 2 + (rho / 2) * z[m:m + n] + (1 - rho / 2) * x[m:m + n]))
    x_n = mul(Q, (-lamda[m:m + n] / 2 + (rho / 2) * z[m:m + n]))
    x = np.concatenate((x_m, x_n), axis=0)

    # compute z
    z = mul(coeff_z, lamda / 2 + (rho / 2) * (x - z))

    # update lamda
    lamda = lamda + rho * (x - z)
    # if np.sum(np.abs(x - z)) < 0.002 and rho < 1000:
    #     print(iter)
    #     rho = rho + iter / 1000

    if np.sum(np.abs(x - z)) < 0.001 or iter > 10000:
        done = True

x_norm = x * (1 + gamma)

print("hi")
