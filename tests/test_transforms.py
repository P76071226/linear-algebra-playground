import numpy as np

from linalg_playground.transforms import (
    apply_transform,
    build_rotation,
    build_shear,
    eigen_directions,
)


def test_rotation_preserves_norms():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(20, 2))
    A = build_rotation(np.pi / 3)
    transformed = apply_transform(points, A)
    assert np.allclose(
        np.linalg.norm(points, axis=1), np.linalg.norm(transformed, axis=1)
    )


def test_apply_transform_matches_direct_matrix_vector_product():
    A = build_rotation(np.pi / 2)
    points = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = apply_transform(points, A)
    expected = np.array([A @ p for p in points])
    assert np.allclose(result, expected)


def test_single_axis_shear_preserves_area():
    A = build_shear(0.7, 0.0)
    assert np.isclose(np.linalg.det(A), 1.0)


def test_shear_determinant_matches_closed_form():
    A = build_shear(0.4, 0.3)
    assert np.isclose(np.linalg.det(A), 1 - 0.4 * 0.3)


def test_symmetric_matrix_eigenvectors_are_orthogonal():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    _, eigenvectors = eigen_directions(A)
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
    assert np.isclose(np.dot(v1, v2), 0.0, atol=1e-10)


def test_eigen_directions_satisfy_the_eigenvalue_equation():
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    eigenvalues, eigenvectors = eigen_directions(A)
    for i in range(2):
        assert np.allclose(A @ eigenvectors[:, i], eigenvalues[i] * eigenvectors[:, i])
