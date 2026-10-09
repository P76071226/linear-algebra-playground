import numpy as np

from linalg_playground.kalman import predict, update


def test_predict_propagates_state_and_covariance():
    state = np.array([2.0, 3.0])
    covariance = np.diag([4.0, 1.0])
    transition = np.array([[1.0, 0.5], [0.0, 1.0]])
    process_noise = np.diag([0.1, 0.2])

    predicted_state, predicted_covariance = predict(
        state, covariance, transition, process_noise
    )

    assert np.allclose(predicted_state, [3.5, 3.0])
    assert np.allclose(
        predicted_covariance,
        transition @ covariance @ transition.T + process_noise,
    )


def test_update_moves_toward_measurement_and_reduces_position_uncertainty():
    predicted_state = np.array([0.0, 1.0])
    predicted_covariance = np.diag([4.0, 1.0])
    observation = np.array([[1.0, 0.0]])
    measurement_noise = np.array([[1.0]])

    state, covariance, gain, innovation = update(
        predicted_state,
        predicted_covariance,
        np.array([2.0]),
        observation,
        measurement_noise,
    )

    assert 0.0 < state[0] < 2.0
    assert covariance[0, 0] < predicted_covariance[0, 0]
    assert np.allclose(gain, [[0.8], [0.0]])
    assert np.allclose(innovation, [2.0])


def test_uncertainty_grows_when_measurements_are_missing():
    state = np.array([0.0, 1.0])
    covariance = np.diag([0.5, 0.2])
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process_noise = 0.0001 * np.array([[0.25, 0.5], [0.5, 1.0]])
    position_variances = [covariance[0, 0]]

    for _ in range(5):
        state, covariance = predict(
            state, covariance, transition, process_noise
        )
        position_variances.append(covariance[0, 0])

    assert np.all(np.diff(position_variances) > 0.0)


def test_constant_velocity_filter_improves_noisy_position_measurements():
    rng = np.random.default_rng(7)
    dt = 1.0
    steps = 100
    truth = 0.7 * np.arange(steps) * dt
    measurements = truth + rng.normal(scale=2.0, size=steps)

    transition = np.array([[1.0, dt], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])
    process_noise = 0.0001 * np.array([[0.25, 0.5], [0.5, 1.0]])
    measurement_noise = np.array([[4.0]])
    state = np.array([0.0, 0.0])
    covariance = np.diag([10.0, 10.0])
    estimates = []

    for measurement in measurements:
        state, covariance = predict(state, covariance, transition, process_noise)
        state, covariance, _, _ = update(
            state,
            covariance,
            np.array([measurement]),
            observation,
            measurement_noise,
        )
        estimates.append(state.copy())
        assert np.allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)

    estimates = np.asarray(estimates)
    measurement_rmse = np.sqrt(np.mean((measurements - truth) ** 2))
    estimated_rmse = np.sqrt(np.mean((estimates[:, 0] - truth) ** 2))

    assert estimates.shape == (steps, 2)
    assert estimated_rmse < measurement_rmse
