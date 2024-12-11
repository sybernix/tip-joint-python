from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent


def conjgrad(A, b, x):
    """
    https://gist.github.com/glederrey/7fe6e03bbc85c81ed60f3585eea2e073
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(np.sum((rsnew**2))) < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


import numpy as np


def prepare_psi(filter_kernel, patch_height, patch_width, kernel_halfwidth):
    """
    Constructs the Psi matrix for patch processing.

    Parameters:
    filter_kernel (numpy.ndarray): Gaussian filter or any other kernel.
    patch_height (int): Height of the patch.
    patch_width (int): Width of the patch.
    kernel_halfwidth (int): Half-width of the kernel.

    Returns:
    numpy.ndarray: The prepared Psi matrix.
    """
    # Initialize Psi matrix with zeros
    psi = np.zeros((patch_height * patch_width, patch_height * patch_width))

    # Loop over each position in the patch
    for row in range(patch_height):
        for col in range(patch_width):
            # Create a temporary patch padded with zeros
            temp_patch = np.zeros((patch_height + 2 * kernel_halfwidth, patch_width + 2 * kernel_halfwidth))

            # Insert the filter kernel at the correct position
            temp_patch[row: row + 2 * kernel_halfwidth + 1, col: col + 2 * kernel_halfwidth + 1] = filter_kernel

            # Extract the relevant region from the padded patch
            cropped_patch = temp_patch[kernel_halfwidth: kernel_halfwidth + patch_height,
                            kernel_halfwidth: kernel_halfwidth + patch_width]

            # Flatten the cropped patch and assign it to the Psi matrix
            psi[row * patch_width + col, :] = cropped_patch.T.flatten()

    # Add a small value for numerical stability
    psi += 1e-8

    # Normalize Psi matrix using Sinkhorn-Knopp normalization
    psi = sinkhorn_knopp(psi)

    return psi


def sinkhorn_knopp(matrix, max_iter=1000, tol=1e-6):
    """
    Normalizes a matrix using the Sinkhorn-Knopp algorithm to make it doubly stochastic.

    Parameters:
    matrix (numpy.ndarray): Input matrix.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.

    Returns:
    numpy.ndarray: Normalized doubly stochastic matrix.
    """
    row_sum = np.ones(matrix.shape[0])
    col_sum = np.ones(matrix.shape[1])

    for _ in range(max_iter):
        prev_matrix = matrix.copy()
        matrix = matrix / np.sum(matrix, axis=1, keepdims=True)  # Row normalization
        matrix = matrix / np.sum(matrix, axis=0, keepdims=True)  # Column normalization

        # Check for convergence
        if np.linalg.norm(matrix - prev_matrix, ord='fro') < tol:
            break

    return matrix
