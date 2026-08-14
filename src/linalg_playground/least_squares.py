import numpy as np


def fit_least_squares(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return np.linalg.solve(AtA, Atb)


def polynomial_design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)
