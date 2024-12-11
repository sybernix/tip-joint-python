import numpy as np
from scipy.optimize import linprog


def clime_inv(C):
    """
    Solves the CLIME problem for every column of matrix C.
    Inputs:
        C: Covariance matrix (N x N)
    Output:
        c_inv: Solution matrix (N x N), where each column corresponds to the solution for a column of C.
    """
    # Get the size of the matrix
    N = C.shape[0]

    # Initialize the solution matrix
    c_inv = np.zeros((N, N))

    # Solve for each column
    for ii in range(N):
        print(f"Solving for column {ii + 1}...")
        c_inv[:, ii] = inverse_solver(C, ii)

    print("All columns solved.")
    return c_inv


def inverse_solver(C, ii):
    """
    Computes the inverse of matrix C using LP for the column index ii.
    Inputs:
        C: Input matrix (should be invertible).
        ii: Column index for which the inverse column is computed.
    Output:
        l_i: Solution vector for column ii.
    """
    # Identity matrix and objective coefficients
    emat = np.eye(C.shape[0])
    c = np.ones(C.shape[0])  # Objective to minimize the sum of residuals

    # Linear programming constraints
    A_eq = np.vstack([C, -C])  # Combine constraints for upper and lower bounds
    b_eq = np.concatenate([emat[:, ii], -emat[:, ii]])  # Corresponding RHS

    # Solve the LP using SciPy's linprog
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs', options={'disp': False})

    # Check feasibility
    if not res.success:
        raise ValueError(f"Inverse computation failed for column {ii}: LP is infeasible.")

    # Extract the solution vector
    return res.x


if __name__ == "__main__":
    # Example covariance matrix
    C = np.array([[2.0, -1.0, 0.0],
                  [-1.0, 2.0, -1.0],
                  [0.0, -1.0, 2.0]])

    # Solve for the inverse using CLIME
    c_inv = clime_inv(C)
    print("Computed inverse:")
    print(np.dot(C,c_inv))
