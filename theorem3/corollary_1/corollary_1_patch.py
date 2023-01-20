import numpy as np

img_patch = np.array([[153, 156], [157, 158]])
flattened_patch = np.reshape(img_patch, (4, 1))

psi = np.array([[0.312816957047989, 0.247249820875324, 0.260255415738962, 0.179677806337725],
                    [0.247249820875324, 0.247335656093526, 0.278941892119199, 0.226472630911952],
                    [0.260255415738962, 0.278941892119199, 0.201335672588002, 0.259467019553837],
                    [0.179677806337726, 0.226472630911952, 0.259467019553837, 0.334382543196486]])

# graph filtering
m = 4
n = 2
gamma = 0.4

mu = 0.3
laplacian = (1 / mu) * (np.linalg.pinv(psi) - (1 + gamma) * np.identity(m))

# initialize interpolator
theta = np.array([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
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

graph_filtered_output = np.matmul(q, flattened_patch)

# linear operations
linear_denoised = np.matmul(psi, flattened_patch)
linear_denoised_interp = np.matmul(theta, linear_denoised)

print("hi")
