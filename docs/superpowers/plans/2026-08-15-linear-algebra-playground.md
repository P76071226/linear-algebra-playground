# Linear Algebra Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a four-notebook portfolio repo (`linear-algebra-playground`) that makes matrix transforms/eigenvectors, PCA, least squares, and ICP registration visually and numerically concrete, with hand-implemented core linear algebra backed by pytest.

**Architecture:** A `src/linalg_playground` package holds hand-implemented, pytest-covered core functions (one module per project). `notebooks/` holds the narrative layer (markdown explanation + calls into the package + matplotlib visualization, built via `nbformat` and executed via `nbconvert`). `data/bunny_subsampled.npy` is a committed, pre-fetched point cloud so notebooks run offline.

**Tech Stack:** Python (`uv`-managed), NumPy, SciPy (`cKDTree` only), Matplotlib, `ipywidgets`/`ipympl`, scikit-learn (reference cross-checks + one real dataset), Jupyter, pytest, GitHub Actions.

## Global Constraints

- Python `>=3.10`, dependency management via `uv` (`pyproject.toml` + `uv.lock`).
- Core linear-algebra logic must be hand-implemented in `src/linalg_playground/*.py`: rotation/scale/shear matrix builders, PCA's covariance + eigendecomposition, least squares' normal-equation solve, ICP's SVD-based rigid transform. High-level solvers (`np.linalg.lstsq`, `sklearn.decomposition.PCA`, etc.) are used **only** in tests, to cross-check the hand-implemented result.
- `scipy.spatial.cKDTree` is used as-is for ICP's nearest-neighbor search — it's a utility, not core linear algebra, so it is not hand-rolled.
- Every hand-implemented function gets a pytest test cross-checking it against a NumPy/SciPy/scikit-learn reference or a known closed-form ground truth.
- Notebooks import functions from `linalg_playground` — no duplicated math logic inside notebook cells.
- Author: Kai Yao. Repo name: `linear-algebra-playground`. GitHub username: `P76071226`. License: MIT.
- Deviations from the spec's exact file tree (both needed for the spec's own requirements, not scope creep): a `scripts/fetch_bunny.py` (one-time data fetch, since the spec requires `data/bunny_subsampled.npy` to exist) and `scripts/make_readme_images.py` (one-time README image generation, since the spec requires an illustrative image per project in the README).

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/linalg_playground/__init__.py`
- Create: `.gitignore`
- Create: `LICENSE`
- Create (empty dirs via `.gitkeep` not needed — created implicitly when files are added): `tests/`, `notebooks/`, `data/`, `scripts/`, `docs/images/`, `.github/workflows/`

**Interfaces:**
- Produces: an importable `linalg_playground` package (empty for now) that later tasks add modules to; a working `uv sync` environment all later tasks assume is already set up.

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p src/linalg_playground tests notebooks data scripts docs/images .github/workflows
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "linalg-playground"
version = "0.1.0"
description = "Small visual projects for building linear algebra intuition: matrix transforms, PCA, least squares, and ICP registration."
authors = [{ name = "Kai Yao" }]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "ipympl>=0.9",
    "ipywidgets>=8.1",
    "scikit-learn>=1.4",
    "jupyter>=1.0",
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/linalg_playground"]
```

- [ ] **Step 3: Create the package `__init__.py`**

```python
```

(empty file — write zero bytes to `src/linalg_playground/__init__.py`)

- [ ] **Step 4: Sync the environment and verify imports**

```bash
uv sync
uv run python -c "import numpy, scipy, matplotlib, sklearn, ipywidgets, pytest, linalg_playground; print('ok')"
```

Expected: `ok` printed, `uv.lock` created in the repo root.

- [ ] **Step 5: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
*.egg-info/
.DS_Store
```

- [ ] **Step 6: Write `LICENSE`**

```
MIT License

Copyright (c) 2026 Kai Yao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/linalg_playground/__init__.py .gitignore LICENSE
git commit -m "Scaffold project: uv-managed pyproject, package skeleton, license"
```

---

### Task 2: Transform Playground core module (`transforms.py`)

**Files:**
- Create: `src/linalg_playground/transforms.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Produces:
  - `build_rotation(theta: float) -> np.ndarray` — shape `(2, 2)`
  - `build_scale(sx: float, sy: float) -> np.ndarray` — shape `(2, 2)`
  - `build_shear(kx: float, ky: float) -> np.ndarray` — shape `(2, 2)`
  - `apply_transform(points: np.ndarray, A: np.ndarray) -> np.ndarray` — `points` is `(N, 2)`, returns `(N, 2)`
  - `eigen_directions(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(eigenvalues (2,), eigenvectors (2,2) as columns)`, sorted by descending `|eigenvalue|`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transforms.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `linalg_playground.transforms` doesn't exist yet.

- [ ] **Step 3: Implement `transforms.py`**

```python
# src/linalg_playground/transforms.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: all 6 tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/linalg_playground/transforms.py tests/test_transforms.py
git commit -m "Implement transform builders and eigenvector helper, with tests"
```

---

### Task 3: Transform Playground notebook

**Files:**
- Create: `notebooks/01_transform_playground.ipynb`

**Interfaces:**
- Consumes: `build_rotation`, `build_scale`, `build_shear`, `apply_transform`, `eigen_directions` from Task 2.

- [ ] **Step 1: Generate the notebook**

```bash
uv run python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = r"""# Matrix Transformations & Eigenvectors

A matrix isn't just a grid of numbers -- it's a description of how to move every point in space. This notebook makes that concrete: pick a 2x2 matrix, watch what it does to a grid of points and a unit circle, then find the one or two directions that matrix leaves unchanged (its eigenvectors)."""

md2 = r"""## The math

A matrix $A$ acts on a point $x$ by matrix-vector multiplication:

$$x' = Ax$$

Three basic building blocks:

- **Rotation** by angle $\theta$: $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
- **Scale** by $(s_x, s_y)$: $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$
- **Shear** by $(k_x, k_y)$: $\begin{bmatrix} 1 & k_x \\ k_y & 1 \end{bmatrix}$

An **eigenvector** $v$ of $A$ is a direction that $A$ only stretches, never rotates:

$$Av = \lambda v$$

for some scalar $\lambda$ (the eigenvalue). Every other direction gets rotated by $A$; eigenvectors don't."""

code1 = r"""import numpy as np
import matplotlib.pyplot as plt
from linalg_playground.transforms import (
    build_rotation, build_scale, build_shear, apply_transform, eigen_directions,
)"""

code2 = r"""def unit_circle_points(n=200):
    angles = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def grid_points(lim=2.0, step=0.5):
    xs = np.arange(-lim, lim + step, step)
    ys = np.arange(-lim, lim + step, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def plot_before_after(A, title=""):
    circle = unit_circle_points()
    grid = grid_points()
    before = np.vstack([grid, circle])
    after = apply_transform(before, A)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, pts, label in zip(axes, [before, after], ["Before", "After"]):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color="tab:blue")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(label)
    fig.suptitle(title)
    plt.show()
    return fig"""

code3 = r"""plot_before_after(build_rotation(np.pi / 4), "Rotation by 45 degrees")
plot_before_after(build_scale(2.0, 0.5), "Scale (x2, x0.5)")
plot_before_after(build_shear(0.6, 0.0), "Shear (kx=0.6)")"""

md3 = r"""## Where do eigenvectors point?

Below, the gray dots trace the unit circle after being transformed by $A$; the colored arrows are the eigenvector directions, scaled by their eigenvalues. Watch how every other direction on the circle bends, but the arrows keep pointing the same way (only their length changes)."""

code4 = r"""def plot_eigen_directions(A, title=""):
    eigvals, eigvecs = eigen_directions(A)
    circle = unit_circle_points()
    transformed = apply_transform(circle, A)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(transformed[:, 0], transformed[:, 1], s=4, color="lightgray", label="transformed unit circle")

    if not np.iscomplexobj(eigvecs):
        for i in range(eigvecs.shape[1]):
            v = eigvecs[:, i]
            scale = eigvals[i]
            ax.arrow(0, 0, v[0] * scale, v[1] * scale, head_width=0.1,
                      length_includes_head=True, color=f"C{i}", label=f"eigenvector {i} (lambda={eigvals[i]:.2f})")
            ax.arrow(0, 0, -v[0] * scale, -v[1] * scale, head_width=0.1,
                      length_includes_head=True, color=f"C{i}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    plt.show()
    print("eigenvalues:", eigvals)
    return fig


symmetric_A = np.array([[2.0, 1.0], [1.0, 1.5]])
plot_eigen_directions(symmetric_A, "Symmetric matrix: orthogonal eigenvectors")"""

md4 = r"""## What about a pure rotation?

A rotation matrix (other than 0 degrees or 180 degrees) has no *real* eigenvector -- every direction actually turns. NumPy will report complex eigenvalues in that case, which is itself a useful signal: complex eigenvalues mean "this transform rotates, it doesn't just stretch along fixed axes.\""""

code5 = r"""rotation_eigvals, _ = eigen_directions(build_rotation(np.pi / 3))
print("Rotation eigenvalues (complex - no real eigenvector):", rotation_eigvals)"""

md5 = r"""## Try it yourself

The cell below gives you sliders for the four entries of $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$. Drag them and watch the grid, the transformed unit circle, and the eigenvector arrows update live.

**Note:** the sliders only respond when this notebook is run locally (`uv run jupyter lab`) -- GitHub's notebook preview renders the last-saved static output only. The static plots above already show the same idea for a few representative matrices."""

code6 = r"""%matplotlib widget
import ipywidgets as widgets

fig, ax = plt.subplots(figsize=(5, 5))


def update(a=1.0, b=0.0, c=0.0, d=1.0):
    ax.clear()
    A = np.array([[a, b], [c, d]])
    circle = unit_circle_points()
    transformed = apply_transform(circle, A)
    ax.scatter(transformed[:, 0], transformed[:, 1], s=4, color="lightgray")

    eigvals, eigvecs = eigen_directions(A)
    if not np.iscomplexobj(eigvecs):
        for i in range(eigvecs.shape[1]):
            v = eigvecs[:, i]
            scale = eigvals[i]
            ax.arrow(0, 0, v[0] * scale, v[1] * scale, head_width=0.1,
                      length_includes_head=True, color=f"C{i}")
            ax.arrow(0, 0, -v[0] * scale, -v[1] * scale, head_width=0.1,
                      length_includes_head=True, color=f"C{i}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    fig.canvas.draw_idle()


widgets.interact(
    update,
    a=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.0),
    b=widgets.FloatSlider(min=-3, max=3, step=0.1, value=0.0),
    c=widgets.FloatSlider(min=-3, max=3, step=0.1, value=0.0),
    d=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.0),
)"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(md4),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(md5),
    nbf.v4.new_code_cell(code6),
]

nbf.write(nb, "notebooks/01_transform_playground.ipynb")
print("written")
PY
```

Expected: `written` printed, `notebooks/01_transform_playground.ipynb` exists.

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs without error**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_transform_playground.ipynb
```

Expected: command exits `0` with no `Traceback` in the output. If the `%matplotlib widget` cell errors under headless execution, replace `%matplotlib widget` with `%matplotlib inline` for the purposes of the committed executed output (the local-run instructions in the markdown cell already tell the user to switch backends when running interactively) — keep the source cell as `%matplotlib widget` for local use, but note this fallback only if nbconvert actually fails on it.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_transform_playground.ipynb
git commit -m "Add transform playground notebook"
```

---

### Task 4: PCA core module (`pca.py`)

**Files:**
- Create: `src/linalg_playground/pca.py`
- Test: `tests/test_pca.py`

**Interfaces:**
- Produces:
  - `pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `X` is `(N, D)`; returns `(mean (D,), eigenvalues (D,) sorted descending, eigenvectors (D, D) columns sorted to match)`
  - `project(X: np.ndarray, mean: np.ndarray, eigenvectors: np.ndarray, n_components: int) -> np.ndarray` — returns `(N, n_components)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pca.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_pca.py -v
```

Expected: `ModuleNotFoundError` — `linalg_playground.pca` doesn't exist yet.

- [ ] **Step 3: Implement `pca.py`**

```python
# src/linalg_playground/pca.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_pca.py -v
```

Expected: all 4 tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/linalg_playground/pca.py tests/test_pca.py
git commit -m "Implement PCA via covariance and eigendecomposition, with tests"
```

---

### Task 5: PCA notebook

**Files:**
- Create: `notebooks/02_pca_point_cloud.ipynb`

**Interfaces:**
- Consumes: `pca`, `project` from Task 4.

- [ ] **Step 1: Generate the notebook**

```bash
uv run python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = r"""# PCA: Finding the Directions Your Data Actually Varies Along

Principal Component Analysis answers one question: if you had to describe a cloud of points using as few directions as possible, which directions would you pick? This notebook builds PCA from a covariance matrix and its eigenvectors, first on a synthetic 2D cloud where you can see the answer, then on a real 30-feature dataset where you can't."""

md2 = r"""## The math

Given data $X \in \mathbb{R}^{N \times D}$ (rows are samples):

1. Center it: $\tilde{X} = X - \bar{x}$
2. Compute the covariance matrix: $C = \frac{1}{N-1}\tilde{X}^T\tilde{X}$
3. Eigendecompose $C$: $Cv_i = \lambda_i v_i$
4. Sort eigenvectors by eigenvalue, descending. The top eigenvector points along the direction of maximum variance; each eigenvalue is how much variance lies along that direction.

Projecting onto the top $k$ eigenvectors gives the best $k$-dimensional summary of the data (in a least-squares sense)."""

code1 = r"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from linalg_playground.pca import pca, project"""

code2 = r"""rng = np.random.default_rng(42)
n = 400
t = rng.normal(size=n)
noise = rng.normal(scale=0.3, size=n)
direction = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
perpendicular = np.array([-direction[1], direction[0]])
cloud = np.outer(t, direction) + np.outer(noise, perpendicular)

mean, eigenvalues, eigenvectors = pca(cloud)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cloud[:, 0], cloud[:, 1], s=8, alpha=0.5)
for i in range(2):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    ax.arrow(mean[0], mean[1], v[0], v[1], head_width=0.15,
              length_includes_head=True, color=f"C{i+1}",
              label=f"PC{i+1} (variance={eigenvalues[i]:.2f})")
ax.set_aspect("equal")
ax.legend()
ax.set_title("Principal directions of a synthetic point cloud")
plt.show()"""

md3 = r"""## A real dataset: 30 features down to 2

The Breast Cancer Wisconsin dataset has 30 measured features per sample. We can't plot 30 dimensions, but we can project onto the top 2 principal components and see how much of the data's structure survives."""

code3 = r"""data = load_breast_cancer()
X, y = data.data, data.target

mean, eigenvalues, eigenvectors = pca(X)
projected = project(X, mean, eigenvectors, n_components=2)

explained = eigenvalues[:2].sum() / eigenvalues.sum()
print(f"Top 2 components explain {explained:.1%} of total variance")

fig, ax = plt.subplots(figsize=(6, 6))
for label, name in zip([0, 1], data.target_names):
    mask = y == label
    ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=0.6, label=name)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.set_title("Breast cancer dataset projected onto its top 2 principal components")
plt.show()"""

md4 = r"""## Takeaway

PCA is a change of basis: instead of describing points using the original feature axes, we describe them using the eigenvectors of the covariance matrix -- axes chosen so the first few capture as much of the data's spread as possible. That's the same eigenvector idea from the transform playground notebook, just applied to a covariance matrix instead of an arbitrary transform."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(md4),
]

nbf.write(nb, "notebooks/02_pca_point_cloud.ipynb")
print("written")
PY
```

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs without error**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_pca_point_cloud.ipynb
```

Expected: exits `0`, no `Traceback`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_pca_point_cloud.ipynb
git commit -m "Add PCA notebook"
```

---

### Task 6: Least Squares core module (`least_squares.py`)

**Files:**
- Create: `src/linalg_playground/least_squares.py`
- Test: `tests/test_least_squares.py`

**Interfaces:**
- Produces:
  - `fit_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray` — `A` is `(N, D)`, `b` is `(N,)`, returns `(D,)`
  - `polynomial_design_matrix(x: np.ndarray, degree: int) -> np.ndarray` — `x` is `(N,)`, returns `(N, degree + 1)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_least_squares.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_least_squares.py -v
```

Expected: `ModuleNotFoundError` — `linalg_playground.least_squares` doesn't exist yet.

- [ ] **Step 3: Implement `least_squares.py`**

```python
# src/linalg_playground/least_squares.py
import numpy as np


def fit_least_squares(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return np.linalg.solve(AtA, Atb)


def polynomial_design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_least_squares.py -v
```

Expected: all 3 tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/linalg_playground/least_squares.py tests/test_least_squares.py
git commit -m "Implement least-squares normal-equation solver, with tests"
```

---

### Task 7: Least Squares notebook

**Files:**
- Create: `notebooks/03_least_squares_fitting.ipynb`

**Interfaces:**
- Consumes: `fit_least_squares`, `polynomial_design_matrix` from Task 6.

- [ ] **Step 1: Generate the notebook**

```bash
uv run python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = r"""# Least Squares: Fitting What Doesn't Exactly Fit

Real measurements are noisy, so $Ax = b$ usually has no exact solution. Least squares finds the $x$ that gets closest -- and "closest" turns out to mean projecting $b$ onto the column space of $A$. This notebook builds that solver from the normal equations, then uses it on Anscombe's quartet to show why you should always plot your data."""

md2 = r"""## The math

We want the $x$ minimizing $\|Ax - b\|^2$. Setting the gradient to zero gives the **normal equations**:

$$A^TA\,x = A^Tb$$

Geometrically, $Ax$ can only ever land somewhere in the column space of $A$. The best we can do is pick the point in that column space closest to $b$ -- i.e., the orthogonal projection of $b$ onto $\text{col}(A)$. We can check this after solving: the residual $b - A\hat{x}$ should be orthogonal to every column of $A$."""

code1 = r"""import numpy as np
import matplotlib.pyplot as plt
from linalg_playground.least_squares import fit_least_squares, polynomial_design_matrix"""

code2 = r"""rng = np.random.default_rng(0)
x = np.linspace(0, 10, 40)
y_true = 3.0 * x + 2.0
y = y_true + rng.normal(scale=2.5, size=x.shape)

A = np.column_stack([np.ones_like(x), x])
coeffs = fit_least_squares(A, y)
print(f"Fitted: y = {coeffs[1]:.2f} x + {coeffs[0]:.2f}  (true: y = 3.00 x + 2.00)")

residual = y - A @ coeffs
print("Residual dot columns of A (should be ~0):", A.T @ residual)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, s=15, alpha=0.6, label="noisy measurements")
ax.plot(x, A @ coeffs, color="tab:red", label="least-squares fit")
ax.plot(x, y_true, color="gray", linestyle="--", label="true line")
ax.legend()
ax.set_title("Least-squares line fit")
plt.show()"""

code3 = r"""rng = np.random.default_rng(1)
x2 = np.linspace(-3, 3, 60)
y2_true = 0.5 * x2 ** 2 - x2 + 1
y2 = y2_true + rng.normal(scale=1.0, size=x2.shape)

A2 = polynomial_design_matrix(x2, degree=2)
coeffs2 = fit_least_squares(A2, y2)
print("Fitted polynomial coefficients (c0, c1, c2):", coeffs2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x2, y2, s=15, alpha=0.6, label="noisy measurements")
order = np.argsort(x2)
ax.plot(x2[order], (A2 @ coeffs2)[order], color="tab:red", label="least-squares fit (degree 2)")
ax.legend()
ax.set_title("Least-squares polynomial fit")
plt.show()"""

md3 = r"""## Anscombe's quartet

These four datasets were constructed by Francis Anscombe (1973) to have nearly identical summary statistics -- including the exact same least-squares fit -- despite looking completely different. Least squares alone can't tell them apart; only plotting can."""

code4 = r"""anscombe_x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
anscombe_y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
anscombe_x2 = anscombe_x1.copy()
anscombe_y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
anscombe_x3 = anscombe_x1.copy()
anscombe_y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
anscombe_x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
anscombe_y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

datasets = [
    (anscombe_x1, anscombe_y1),
    (anscombe_x2, anscombe_y2),
    (anscombe_x3, anscombe_y3),
    (anscombe_x4, anscombe_y4),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (xs, ys) in zip(axes.ravel(), datasets):
    A_i = np.column_stack([np.ones_like(xs), xs])
    coeffs_i = fit_least_squares(A_i, ys)
    print(f"slope={coeffs_i[1]:.3f} intercept={coeffs_i[0]:.3f}")
    order_i = np.argsort(xs)
    ax.scatter(xs, ys)
    ax.plot(xs[order_i], (A_i @ coeffs_i)[order_i], color="tab:red")
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)
fig.suptitle("Anscombe's quartet: same fitted line, very different data")
plt.tight_layout()
plt.show()"""

md4 = r"""## Takeaway

The normal equations turn "find the best-fit line" into a small linear system. But the same fitted line can come from very different underlying data -- the fit doesn't know it's being fooled. Always look at the plot, not just the coefficients."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(md4),
]

nbf.write(nb, "notebooks/03_least_squares_fitting.ipynb")
print("written")
PY
```

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs without error**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_least_squares_fitting.ipynb
```

Expected: exits `0`, no `Traceback`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_least_squares_fitting.ipynb
git commit -m "Add least squares notebook"
```

---

### Task 8: Stanford Bunny data fetch

**Files:**
- Create: `scripts/fetch_bunny.py`
- Create (generated, then committed): `data/bunny_subsampled.npy`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `data/bunny_subsampled.npy`, a `(N, 3)` `float64` NumPy array with `1000 <= N <= 2500`, loadable via `np.load`. Task 9 and Task 10 depend on this file existing.

- [ ] **Step 1: Write the failing data test**

```python
# tests/test_data.py
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bunny_subsampled.npy"


def test_bunny_point_cloud_exists_with_expected_shape():
    assert DATA_PATH.exists(), f"missing {DATA_PATH} -- run scripts/fetch_bunny.py"
    points = np.load(DATA_PATH)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert 1000 <= points.shape[0] <= 2500
    assert points.dtype == np.float64
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_data.py -v
```

Expected: `FAIL` — `data/bunny_subsampled.npy` doesn't exist yet.

- [ ] **Step 3: Write the fetch script**

```python
# scripts/fetch_bunny.py
"""One-off script to download the Stanford Bunny reconstruction and save a
point cloud as data/bunny_subsampled.npy.

Run with: uv run python scripts/fetch_bunny.py
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

BUNNY_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
MEMBER_NAME = "bunny/reconstruction/bun_zipper_res3.ply"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "bunny_subsampled.npy"


def parse_ascii_ply_vertices(text):
    lines = text.splitlines()
    vertex_count = None
    header_end = None
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i
            break
    if vertex_count is None or header_end is None:
        raise ValueError("Could not parse PLY header")

    points = np.empty((vertex_count, 3), dtype=np.float64)
    for row, line in enumerate(lines[header_end + 1 : header_end + 1 + vertex_count]):
        x, y, z = line.split()[:3]
        points[row] = (float(x), float(y), float(z))
    return points


def main():
    print(f"Downloading {BUNNY_URL} ...")
    with urllib.request.urlopen(BUNNY_URL, timeout=60) as response:
        archive_bytes = response.read()

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        member = tar.extractfile(MEMBER_NAME)
        if member is None:
            raise FileNotFoundError(MEMBER_NAME)
        text = member.read().decode("ascii")

    points = parse_ascii_ply_vertices(text)
    print(f"Parsed {points.shape[0]} vertices")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, points)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the fetch script**

```bash
uv run python scripts/fetch_bunny.py
```

Expected: prints `Parsed 1889 vertices` and `Saved to .../data/bunny_subsampled.npy`.

- [ ] **Step 5: Run the test to verify it passes**

```bash
uv run pytest tests/test_data.py -v
```

Expected: `PASS`.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_bunny.py data/bunny_subsampled.npy tests/test_data.py
git commit -m "Add Stanford Bunny fetch script and committed point cloud data"
```

---

### Task 9: ICP core module (`icp.py`)

**Files:**
- Create: `src/linalg_playground/icp.py`
- Test: `tests/test_icp.py`

**Interfaces:**
- Consumes: `data/bunny_subsampled.npy` (Task 8), for the notebook only — this task's tests use synthetic data.
- Produces:
  - `nearest_neighbors(source: np.ndarray, target: np.ndarray) -> np.ndarray` — `source` `(N, D)`, `target` `(M, D)`, returns int indices `(N,)` into `target`
  - `best_rigid_transform(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `P`, `Q` both `(N, D)`, returns `(R (D, D), t (D,))` such that `R @ P[i] + t ≈ Q[i]`
  - `icp(source: np.ndarray, target: np.ndarray, max_iterations: int = 50, tolerance: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]` — returns `(aligned_source (N, D), R_total (D, D), t_total (D,), errors)`, where `errors` is the mean-squared nearest-neighbor error per iteration and `R_total, t_total` is the composed transform such that `source @ R_total.T + t_total ≈ aligned_source`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_icp.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_icp.py -v
```

Expected: `ModuleNotFoundError` — `linalg_playground.icp` doesn't exist yet.

- [ ] **Step 3: Implement `icp.py`**

```python
# src/linalg_playground/icp.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_icp.py -v
```

Expected: all 3 tests `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/linalg_playground/icp.py tests/test_icp.py
git commit -m "Implement ICP (SVD rigid transform + nearest-neighbor loop), with tests"
```

---

### Task 10: ICP notebook

**Files:**
- Create: `notebooks/04_icp_registration.ipynb`

**Interfaces:**
- Consumes: `icp`, `best_rigid_transform` from Task 9; `data/bunny_subsampled.npy` from Task 8.

- [ ] **Step 1: Generate the notebook**

```bash
uv run python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = r"""# ICP: Aligning Two Point Clouds with SVD

Iterative Closest Point (ICP) answers: given two point clouds that are the same shape but in different positions/orientations, what rotation and translation aligns one onto the other? This notebook implements it from scratch on the Stanford Bunny and shows it recovering a known rigid transform."""

md2 = r"""## The math

Given two matched point sets $P$ and $Q$ (same length, corresponding points), the rotation $R$ and translation $t$ minimizing $\sum_i \|Rp_i + t - q_i\|^2$ has a closed form (the Kabsch / orthogonal Procrustes solution):

1. Center both sets on their centroids: $\bar p, \bar q$
2. Cross-covariance: $H = \tilde P^T \tilde Q$
3. SVD: $H = U\Sigma V^T$
4. $R = V\,\text{diag}(1,\dots,1,\det(VU^T))\,U^T$ (the $\det$ term fixes a reflection SVD can produce)
5. $t = \bar q - R\bar p$

But real point clouds usually aren't matched point-for-point -- we don't know which point in $Q$ corresponds to which in $P$. ICP works around that by alternating:

1. Guess correspondences: match each point in $P$ to its **nearest neighbor** in $Q$
2. Solve for the best $(R, t)$ given those guessed correspondences
3. Apply it, and repeat until the alignment stops improving"""

code1 = r"""import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from linalg_playground.icp import icp, best_rigid_transform

source = np.load("../data/bunny_subsampled.npy")
print("bunny point cloud shape:", source.shape)"""

code2 = r"""theta = np.radians(25)
R_true = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta), np.cos(theta), 0.0],
    [0.0, 0.0, 1.0],
])
t_true = np.array([0.05, -0.03, 0.02])

target = source @ R_true.T + t_true"""

code3 = r"""aligned, R_est, t_est, errors = icp(source, target, max_iterations=50)
print(f"Converged in {len(errors)} iterations, final mean squared error: {errors[-1]:.2e}")
print("R_est:\n", R_est)
print("R_true:\n", R_true)
print("t_est:", t_est, " t_true:", t_true)"""

code4 = r"""fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection="3d")
ax1.scatter(*source.T, s=2, color="tab:blue", label="source")
ax1.scatter(*target.T, s=2, color="tab:orange", label="target")
ax1.set_title("Before alignment")
ax1.legend()

ax2 = fig.add_subplot(132, projection="3d")
ax2.scatter(*aligned.T, s=2, color="tab:blue", label="aligned source")
ax2.scatter(*target.T, s=2, color="tab:orange", label="target")
ax2.set_title("After ICP alignment")
ax2.legend()

ax3 = fig.add_subplot(133)
ax3.plot(errors, marker="o")
ax3.set_yscale("log")
ax3.set_xlabel("iteration")
ax3.set_ylabel("mean squared error (log scale)")
ax3.set_title("Convergence")

plt.tight_layout()
plt.show()"""

md3 = r"""## Takeaway

The same SVD-based "best rotation aligning two point sets" idea from the closed-form Procrustes solution powers the whole iterative loop -- ICP just re-guesses correspondences and re-solves it until the guess stops changing. This is the same rigid-transform math behind real-world point cloud registration in robotics and 3D scanning."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(md3),
]

nbf.write(nb, "notebooks/04_icp_registration.ipynb")
print("written")
PY
```

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs without error**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_icp_registration.ipynb
```

Expected: exits `0`, no `Traceback`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_icp_registration.ipynb
git commit -m "Add ICP registration notebook"
```

---

### Task 11: CI workflow

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `pyproject.toml`/`uv.lock` (Task 1) and all `tests/*.py` (Tasks 2, 4, 6, 8, 9).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/tests.yml
name: tests

on:
  push:
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.11
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest -v
```

- [ ] **Step 2: Validate the YAML is well-formed**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); print('valid yaml')"
```

Expected: `valid yaml`. (If `PyYAML` isn't available, `python -c "import json,sys; sys.exit(0)"`-style tools won't help for YAML — instead just visually confirm indentation is consistent with the block above; do not skip this check silently.)

- [ ] **Step 3: Run the full test suite locally as a final pre-CI sanity check**

```bash
uv run pytest -v
```

Expected: all tests from Tasks 2, 4, 6, 8, 9 `PASS` (16 tests total: 6 transforms + 4 pca + 3 least_squares + 1 data + 3 icp).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "Add GitHub Actions workflow running pytest via uv"
```

---

### Task 12: README and hero images

**Files:**
- Create: `scripts/make_readme_images.py`
- Create (generated, then committed): `docs/images/01_transform.png`, `docs/images/02_pca.png`, `docs/images/03_least_squares.png`, `docs/images/04_icp.png`
- Create: `README.md`

**Interfaces:**
- Consumes: all `src/linalg_playground/*.py` modules (Tasks 2, 4, 6, 9) and `data/bunny_subsampled.npy` (Task 8).

- [ ] **Step 1: Write the image-generation script**

```python
# scripts/make_readme_images.py
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
        v = eigvecs[:, i] * eigvals[i]
        ax.arrow(0, 0, v[0], v[1], head_width=0.1, length_includes_head=True, color=f"C{i}")
        ax.arrow(0, 0, -v[0], -v[1], head_width=0.1, length_includes_head=True, color=f"C{i}")
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
        v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
        ax.arrow(mean[0], mean[1], v[0], v[1], head_width=0.15, length_includes_head=True, color=f"C{i+1}")
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
```

- [ ] **Step 2: Run the script and verify the images exist**

```bash
uv run python scripts/make_readme_images.py
ls -la docs/images/
```

Expected: `Saved images to .../docs/images`, and `01_transform.png`, `02_pca.png`, `03_least_squares.png`, `04_icp.png` all present.

- [ ] **Step 3: Write `README.md`**

```markdown
# Linear Algebra Playground

Four small, visual projects that make core linear algebra concepts tangible: matrix transforms and eigenvectors, PCA, least squares, and SVD-based rigid point cloud registration (ICP). Each notebook pairs a short conceptual explanation with a from-scratch NumPy implementation and a plot you can actually see the math in.

[![tests](https://github.com/P76071226/linear-algebra-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/P76071226/linear-algebra-playground/actions/workflows/tests.yml)

## Setup

```bash
git clone https://github.com/P76071226/linear-algebra-playground.git
cd linear-algebra-playground
uv sync
uv run jupyter lab
```

## The four projects, in build order

### 01 — [Transform Playground](notebooks/01_transform_playground.ipynb)
Drag the entries of a 2x2 matrix and watch a grid of points and a unit circle deform in real time, with eigenvectors overlaid to show which directions the matrix leaves unchanged.

![Transform playground](docs/images/01_transform.png)

### 02 — [PCA](notebooks/02_pca_point_cloud.ipynb)
Find the directions a point cloud actually varies along, from a hand-built covariance-and-eigenvector implementation — first on synthetic data, then on a real 30-feature medical dataset reduced to 2D.

![PCA](docs/images/02_pca.png)

### 03 — [Least Squares](notebooks/03_least_squares_fitting.ipynb)
Fit noisy data by solving the normal equations from scratch, then use Anscombe's quartet to see why the same fitted line can hide very different underlying data.

![Least squares](docs/images/03_least_squares.png)

### 04 — [ICP Registration](notebooks/04_icp_registration.ipynb)
Align two copies of the Stanford Bunny point cloud — one rotated and translated from the other — using an SVD-based rigid transform solver wrapped in an iterative nearest-neighbor loop.

![ICP](docs/images/04_icp.png)

## Note on interactive notebooks

Some notebooks use `ipywidgets`/`ipympl` sliders, which only respond when run locally (`uv run jupyter lab`) — GitHub's notebook preview shows the last-saved static output only. Each interactive section also has a static plot alongside it so the idea comes across either way.

## Running the tests

```bash
uv run pytest -v
```

Every hand-implemented core function (rotation/scale/shear builders, PCA's covariance + eigendecomposition, the least-squares normal-equation solver, ICP's SVD-based rigid transform) is cross-checked against NumPy/SciPy/scikit-learn reference implementations or a known closed-form ground truth.

## License

MIT — see [LICENSE](LICENSE).
```

- [ ] **Step 4: Commit**

```bash
git add scripts/make_readme_images.py docs/images README.md
git commit -m "Add README with project roadmap and generated hero images"
```

---

## Post-plan: pushing to GitHub

Not part of this plan's tasks (creating/pushing to a remote is a user-confirmed action, not an autonomous one): once all 12 tasks are done and `uv run pytest -v` is green locally, create the GitHub repo (e.g. `gh repo create P76071226/linear-algebra-playground --public --source=. --remote=origin`) and push, only after explicit confirmation from Kai.
