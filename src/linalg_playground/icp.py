import numpy as np
from scipy.spatial import cKDTree


def nearest_neighbors(source, target):
    tree = cKDTree(target)
    _, indices = tree.query(source)
    return indices


def best_rigid_transform(P, Q):
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_centered = P - p_mean
    Q_centered = Q - q_mean

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0] * (P.shape[1] - 1) + [d])
    R = Vt.T @ D @ U.T

    t = q_mean - R @ p_mean
    return R, t


def icp(source, target, max_iterations=50, tolerance=1e-6):
    current = source.copy()
    R_total = np.eye(source.shape[1])
    t_total = np.zeros(source.shape[1])
    errors = []

    for _ in range(max_iterations):
        indices = nearest_neighbors(current, target)
        matched_target = target[indices]

        R, t = best_rigid_transform(current, matched_target)
        current = current @ R.T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        error = np.mean(np.sum((current - matched_target) ** 2, axis=1))
        errors.append(error)
        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tolerance:
            break

    return current, R_total, t_total, errors
