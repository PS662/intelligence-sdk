#!/usr/bin/env python
"""Write the iris sample to data/iris.csv.

Sample data is not committed; this script regenerates it from scikit-learn's bundled
copy. Column names follow the same cleaning the SDK's library loader applies, so a
``source: file`` recipe on this CSV and a ``source: library`` recipe on ``iris`` see
identical columns (a test pins that).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"


def clean(name: str) -> str:
    return name.replace(" (cm)", "").replace(" ", "_").replace("/", "_")


def write_iris_csv(path: Path = DEFAULT_PATH, target: str = "species") -> Path:
    bunch = load_iris()
    frame = pd.DataFrame(bunch.data, columns=[clean(c) for c in bunch.feature_names])
    frame[target] = [str(bunch.target_names[i]) for i in bunch.target]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = write_iris_csv(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH)
    print(f"wrote {out}")
