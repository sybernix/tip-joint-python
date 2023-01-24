import numpy as np
from scipy.sparse.linalg import cg

n = 20 # size of invertible matrix I wish to generate
A = np.random.rand(n, n)
mx = np.sum(np.abs(A), axis=1)
np.fill_diagonal(A, mx)

b = np.random.rand(n, 1)

# A = np.array([[1, 2], [2, 1]])
# b = np.array([[5], [4]])

x = np.matmul(np.linalg.inv(A), b)

x_cg_temp, exit_code = cg(A, b)

x_cg = np.reshape(x_cg_temp, (n, 1))

diff = np.sum(np.abs(x - x_cg))

print("hi")