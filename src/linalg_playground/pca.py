import numpy as np


def pca(X):
    mean = X.mean(axis=0)
    X_centered = X - mean
    n = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(-eigenvalues)
    return mean, eigenvalues[order], eigenvectors[:, order]


def project(X, mean, eigenvectors, n_components):
    return (X - mean) @ eigenvectors[:, :n_components]
