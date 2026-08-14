# Linear Algebra Playground — Design

## Purpose

Four small, visual projects that make core linear algebra concepts (matrix
transforms, eigenvectors, PCA, least squares, SVD-based rigid registration)
tangible instead of formula-only. Built as a GitHub portfolio piece, done in
learning order: transform playground → PCA → least squares → ICP.

## Repo layout

```
linear-algebra-playground/
├── README.md
├── LICENSE                      # MIT
├── pyproject.toml               # uv-managed deps
├── uv.lock
├── .gitignore
├── src/linalg_playground/
│   ├── __init__.py
│   ├── transforms.py            # 01
│   ├── pca.py                   # 02
│   ├── least_squares.py         # 03
│   └── icp.py                   # 04
├── tests/
│   ├── test_transforms.py
│   ├── test_pca.py
│   ├── test_least_squares.py
│   └── test_icp.py
├── notebooks/
│   ├── 01_transform_playground.ipynb
│   ├── 02_pca_point_cloud.ipynb
│   ├── 03_least_squares_fitting.ipynb
│   └── 04_icp_registration.ipynb
├── data/
│   └── bunny_subsampled.npy     # pre-downloaded, subsampled Stanford Bunny (~1-2k pts)
└── .github/workflows/tests.yml
```

Core linear-algebra logic lives in `src/linalg_playground/` as pure,
testable functions. Notebooks import from this package and are the
"narrative" layer: markdown explanation (concept + formula + intuition
plot, medium depth — no full derivations) + calls into the package +
visualization. This keeps the hand-implemented steps testable by pytest
instead of living only inside notebook cells.

## Per-project design

### 01 — Transform Playground (`transforms.py`)

- Hand-implemented: `build_rotation(theta)`, `build_scale(sx, sy)`,
  `build_shear(kx, ky)`, `apply_transform(points, A)`.
- Notebook: transform a point grid + unit circle; `ipywidgets` sliders for
  matrix entries `a, b, c, d`; overlay eigenvectors (via `np.linalg.eig`)
  as arrows to show which directions stay fixed under the transform.
- Tests: pure rotation preserves norms; symmetric matrix eigenvectors are
  orthogonal; shear matrices have det = 1 (area-preserving).

### 02 — PCA Point Cloud (`pca.py`)

- Hand-implemented: `pca(X)` — center data, compute covariance as
  `X_centered.T @ X_centered / N`, then `np.linalg.eigh` (symmetric-matrix
  solver, more stable) for eigenvectors/eigenvalues, sorted descending by
  eigenvalue.
- Data: (a) synthetic elongated 2D point cloud for the principal-axis
  arrow visualization; (b) real dataset via `sklearn.datasets` (breast
  cancer or digits) projected to the top 2 PCs, plotted to show class
  separation.
- Tests: cross-check against `np.cov` + `np.linalg.eigh` done manually,
  and against `sklearn.decomposition.PCA`'s explained_variance_ratio_.

### 03 — Least Squares Fitting (`least_squares.py`)

- Hand-implemented: `fit_least_squares(A, b)` — build the normal equation
  `A^T A x = A^T b`, solve with `np.linalg.solve` (not explicit inverse,
  noted in markdown for numerical stability).
- Data: (a) synthetic noisy line/polynomial data, with a plot illustrating
  "no exact solution → project b onto column space of A"; (b) Anscombe's
  quartet as the real-data example (same fitted line, very different
  underlying distributions — small, no download needed).
- Tests: cross-check against `np.linalg.lstsq`.

### 04 — ICP Registration (`icp.py`)

- Hand-implemented: `best_rigid_transform(P, Q)` — centroids, cross
  covariance, `np.linalg.svd`, compose `R, t` with the reflection
  (det-sign) correction; `icp(source, target, max_iter)` main loop.
- Uses `scipy.spatial.cKDTree` for nearest-neighbor correspondence search
  (a tool, not core linear algebra, so not hand-rolled).
- Data: `data/bunny_subsampled.npy`, a pre-downloaded and subsampled
  (~1000-2000 points) Stanford Bunny point cloud, committed to the repo so
  notebooks run offline. A known `(R_true, t_true)` is applied to produce
  a source cloud; ICP must recover it.
- Visualization: before/after overlay of source vs. target, and an
  error-vs-iteration convergence curve.
- Tests: `best_rigid_transform` exactly recovers a known `R, t` on a small
  hand-built point set; full `icp()` converges below an error threshold on
  the bunny data.

## Tooling

- Dependency management: `uv` (`pyproject.toml` + `uv.lock`).
- Dependencies: `numpy`, `scipy`, `matplotlib`, `ipympl`, `ipywidgets`,
  `scikit-learn`, `jupyter`, `pytest`.
- Setup: `uv sync && uv run jupyter lab`.

## CI

- `.github/workflows/tests.yml` runs `uv run pytest` on push/PR.
- Only `tests/` run in CI — notebooks are not executed there (slow, and
  contain interactive widgets that don't make sense headless).
- README gets a CI badge.

## README

- One paragraph on the repo's purpose (visual projects building linear
  algebra intuition).
- Learning roadmap listing the four projects in build order (01→04).
- Per-project: one illustrative image/GIF, one-sentence description, link
  to its notebook.
- CI badge, MIT license note, `uv` setup instructions.

## Known limitation: interactive widgets on GitHub

`ipywidgets`/`ipympl` sliders don't render as interactive in GitHub's
static notebook preview. Each notebook's intro markdown and the README
note that sliders require running the notebook locally (`uv run jupyter
lab`), and each interactive section also includes a static before/after
comparison plot so the point still comes across without cloning.
