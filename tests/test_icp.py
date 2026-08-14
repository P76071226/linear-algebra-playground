import numpy as np

from linalg_playground.icp import best_rigid_transform, icp, nearest_neighbors


def test_best_rigid_transform_recovers_a_known_transform():
    rng = np.random.default_rng(6)
    P = rng.normal(size=(20, 3))
    theta = 0.4
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([1.0, -0.5, 0.2])
    Q = P @ R_true.T + t_true

    R_est, t_est = best_rigid_transform(P, Q)
    assert np.allclose(R_est, R_true, atol=1e-8)
    assert np.allclose(t_est, t_true, atol=1e-8)


def test_nearest_neighbors_finds_the_closest_index():
    target = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
    source = np.array([[5.2, 5.1], [0.1, -0.1]])
    indices = nearest_neighbors(source, target)
    assert list(indices) == [2, 0]


def test_icp_recovers_a_known_rigid_transform():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(200, 3)) * np.array([3.0, 1.0, 2.0])
    theta = 0.3
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([0.5, -0.2, 0.1])
    target = source @ R_true.T + t_true

    _, R_est, t_est, errors = icp(source, target, max_iterations=50)
    assert errors[-1] < 1e-6
    assert np.allclose(R_est, R_true, atol=1e-3)
    assert np.allclose(t_est, t_true, atol=1e-3)
