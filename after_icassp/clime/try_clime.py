import numpy as np
from scipy.optimize import linprog

# Problem dimensions
p = 3  # Example size of the matrix (modify as needed)

# Generate a random positive definite covariance matrix with min eigenvalue > 1
min_eigenvalue = 0
while min_eigenvalue <= 1:
    rand_matrix = np.random.randn(p, p)
    Sigma_n = np.dot(rand_matrix.T, rand_matrix)  # Symmetric and positive definite
    min_eigenvalue = np.min(np.linalg.eigvals(Sigma_n))

# Sigma_n = np.array([[7.540806986998345, 0.592408550809672, -1.451203589946607],
#                     [0.592408550809672, 3.689352020526934, -2.179130782121223],
#                     [-1.451203589946607, -2.179130782121223, 3.002917878382207]])

# Lambda value for constraint
lambda_n = 1

# Identity matrix
I_p = np.eye(p)

# Objective function: Minimize sum of absolute values of Omega elements
f = np.ones(p**2).T  # Linear objective function (L1 norm)

# Equality constraints (for matrix multiplication Σn * Ω = I)
Aeq = np.kron(Sigma_n, np.eye(p))  # Kronecker product for matrix multiplication
beq = I_p.flatten()  # Vectorize identity matrix

# Inequality constraints (for ||Σn * Ω - I||∞ ≤ λn)
A = np.vstack([Aeq, -Aeq])  # Combine positive and negative constraints
b = lambda_n * np.ones(2 * p**2)  # Upper and lower bounds for infinity norm constraint
b = b.T

# Solve the linear program
# res = linprog(c=f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='highs', options={'disp': True})

res = linprog(c=f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='interior-point', options={'disp': True, 'tol': 1e-5, 'maxiter': 1000})

# Reshape the solution vector back to matrix form
Omega_vec = res.x
Omega = Omega_vec.reshape((p, p))

# Display the result
print("Estimated Omega:")
print(Omega)

# Verify the result
print("Sigma_n * Omega:")
print(np.dot(Sigma_n, Omega))
