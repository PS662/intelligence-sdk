#!/usr/bin/env python
"""Write the iris samples to data/.

Sample data is not committed; this script regenerates it from scikit-learn's bundled
copy. ``iris.csv`` is the plain dataset. ``iris_extra.csv`` is a small augmentation
set (jittered rows with their own ``sample_id`` column) used to demonstrate
combining datasets. Column names follow the same cleaning the SDK's library loader
applies, so file and library recipes on iris see identical columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IRIS = DATA_DIR / "iris.csv"
IRIS_EXTRA = DATA_DIR / "iris_extra.csv"


def clean(name: str) -> str:
    return name.replace(" (cm)", "").replace(" ", "_").replace("/", "_")


def _iris_frame(target: str = "species") -> pd.DataFrame:
    bunch = load_iris()
    frame = pd.DataFrame(bunch.data, columns=[clean(c) for c in bunch.feature_names])
    frame[target] = [str(bunch.target_names[i]) for i in bunch.target]
    return frame


def write_iris_csv(path: Path = IRIS, target: str = "species") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _iris_frame(target).to_csv(path, index=False)
    return path


def write_iris_extra_csv(path: Path = IRIS_EXTRA, target: str = "species", n: int = 30, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = _iris_frame(target)
    rows = base.sample(n=n, random_state=seed).reset_index(drop=True)
    features = [c for c in rows.columns if c != target]
    rows[features] = (rows[features] + rng.normal(0, 0.05, size=(n, len(features)))).round(2)
    rows.insert(0, "sample_id", range(1000, 1000 + n))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    print(f"wrote {write_iris_csv(out_dir / 'iris.csv')}")
    print(f"wrote {write_iris_extra_csv(out_dir / 'iris_extra.csv')}")
