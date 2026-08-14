import numpy as np

from linalg_playground.least_squares import fit_least_squares, polynomial_design_matrix


def test_fit_matches_np_linalg_lstsq():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(30, 3))
    x_true = np.array([2.0, -1.0, 0.5])
    b = A @ x_true + rng.normal(scale=0.1, size=30)

    x_hat = fit_least_squares(A, b)
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(x_hat, x_lstsq, atol=1e-8)


def test_fit_recovers_an_exact_noise_free_line():
    x = np.linspace(0, 10, 20)
    y = 3.0 * x + 2.0
    A = np.column_stack([np.ones_like(x), x])
    coeffs = fit_least_squares(A, y)
    assert np.allclose(coeffs, [2.0, 3.0], atol=1e-8)


def test_polynomial_design_matrix_shape_and_values():
    x = np.array([0.0, 1.0, 2.0])
    A = polynomial_design_matrix(x, degree=2)
    expected = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 4.0]])
    assert A.shape == (3, 3)
    assert np.allclose(A, expected)
