"""One-off script to generate hero images for the README."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from linalg_playground.transforms import apply_transform, eigen_directions
from linalg_playground.pca import pca
from linalg_playground.least_squares import fit_least_squares
from linalg_playground.icp import icp

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_transform_image():
    A = np.array([[2.0, 1.0], [1.0, 1.5]])
    angles = np.linspace(0, 2 * np.pi, 200)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    transformed = apply_transform(circle, A)
    eigvals, eigvecs = eigen_directions(A)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(transformed[:, 0], transformed[:, 1], s=4, color="lightgray")
    for i in range(2):
        v = np.real(eigvecs[:, i] * eigvals[i])
        ax.arrow(0, 0, float(v[0]), float(v[1]), head_width=0.1, length_includes_head=True, color=f"C{i}")
        ax.arrow(0, 0, float(-v[0]), float(-v[1]), head_width=0.1, length_includes_head=True, color=f"C{i}")
    ax.set_aspect("equal")
    ax.set_title("Eigenvectors of a linear transform")
    fig.savefig(OUT_DIR / "01_transform.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_pca_image():
    rng = np.random.default_rng(42)
    n = 400
    t = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    direction = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
    perpendicular = np.array([-direction[1], direction[0]])
    cloud = np.outer(t, direction) + np.outer(noise, perpendicular)
    mean, eigenvalues, eigenvectors = pca(cloud)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=8, alpha=0.5)
    for i in range(2):
        v = np.real(eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2)
        ax.arrow(float(mean[0]), float(mean[1]), float(v[0]), float(v[1]), head_width=0.15, length_includes_head=True, color=f"C{i+1}")
    ax.set_aspect("equal")
    ax.set_title("PCA: principal directions")
    fig.savefig(OUT_DIR / "02_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_least_squares_image():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    y = 3.0 * x + 2.0 + rng.normal(scale=2.5, size=x.shape)
    A = np.column_stack([np.ones_like(x), x])
    coeffs = fit_least_squares(A, y)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=15, alpha=0.6)
    ax.plot(x, A @ coeffs, color="tab:red")
    ax.set_title("Least-squares fit")
    fig.savefig(OUT_DIR / "03_least_squares.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_icp_image():
    source = np.load(Path(__file__).resolve().parent.parent / "data" / "bunny_subsampled.npy")
    theta = np.radians(25)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.array([0.05, -0.03, 0.02])
    target = source @ R_true.T + t_true
    aligned, _, _, _ = icp(source, target, max_iterations=50)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*aligned.T, s=2, color="tab:blue", label="aligned source")
    ax.scatter(*target.T, s=2, color="tab:orange", label="target")
    ax.set_title("ICP: aligned bunny point clouds")
    ax.legend()
    fig.savefig(OUT_DIR / "04_icp.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_transform_image()
    save_pca_image()
    save_least_squares_image()
    save_icp_image()
    print("Saved images to", OUT_DIR)
