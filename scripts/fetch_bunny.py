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
