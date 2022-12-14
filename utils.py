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
