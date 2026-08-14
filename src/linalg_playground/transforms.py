import numpy as np


def build_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def build_scale(sx, sy):
    return np.array([[sx, 0.0], [0.0, sy]])


def build_shear(kx, ky):
    return np.array([[1.0, kx], [ky, 1.0]])


def apply_transform(points, A):
    return points @ A.T


def eigen_directions(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigenvalues))
    return eigenvalues[order], eigenvectors[:, order]
