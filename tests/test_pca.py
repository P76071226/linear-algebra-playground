import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from linalg_playground.pca import pca, project


def test_pca_matches_manual_covariance_and_eigh():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    mean, eigenvalues, _ = pca(X)

    cov = np.cov(X, rowvar=False)
    expected_vals, _ = np.linalg.eigh(cov)
    order = np.argsort(-expected_vals)

    assert np.allclose(mean, X.mean(axis=0))
    assert np.allclose(eigenvalues, expected_vals[order])


def test_pca_matches_sklearn_explained_variance_ratio():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    _, eigenvalues, _ = pca(X)

    sklearn_pca = SklearnPCA(n_components=4)
    sklearn_pca.fit(X)
    explained_ratio = eigenvalues / eigenvalues.sum()

    assert np.allclose(explained_ratio, sklearn_pca.explained_variance_ratio_, atol=1e-6)


def test_project_reduces_dimensionality():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    mean, _, eigenvectors = pca(X)
    projected = project(X, mean, eigenvectors, n_components=2)
    assert projected.shape == (30, 2)


def test_pca_recovers_a_known_principal_axis():
    rng = np.random.default_rng(4)
    n = 500
    t = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    direction = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
    perpendicular = np.array([-direction[1], direction[0]])
    X = np.outer(t, direction) + np.outer(noise, perpendicular)

    _, _, eigenvectors = pca(X)
    top_component = eigenvectors[:, 0]
    alignment = abs(np.dot(top_component, direction))
    assert alignment > 0.99
