"""Small linear Kalman-filter building blocks."""

import numpy as np


def predict(
    state: np.ndarray,
    covariance: np.ndarray,
    transition: np.ndarray,
    process_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Gaussian state estimate through a linear motion model."""
    predicted_state = transition @ state
    predicted_covariance = transition @ covariance @ transition.T + process_noise
    return predicted_state, predicted_covariance


def update(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: np.ndarray,
    observation: np.ndarray,
    measurement_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one linear measurement with a predicted Gaussian state.

    The covariance uses the Joseph form, which better preserves symmetry and
    positive semidefiniteness under floating-point arithmetic.
    """
    innovation = measurement - observation @ predicted_state
    innovation_covariance = (
        observation @ predicted_covariance @ observation.T + measurement_noise
    )
    kalman_gain = np.linalg.solve(
        innovation_covariance, observation @ predicted_covariance
    ).T

    state = predicted_state + kalman_gain @ innovation
    identity = np.eye(predicted_covariance.shape[0])
    residual_transform = identity - kalman_gain @ observation
    covariance = (
        residual_transform @ predicted_covariance @ residual_transform.T
        + kalman_gain @ measurement_noise @ kalman_gain.T
    )
    return state, covariance, kalman_gain, innovation
