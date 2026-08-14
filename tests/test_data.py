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
