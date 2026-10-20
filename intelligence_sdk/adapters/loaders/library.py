"""``source: library`` loader for scikit-learn's bundled datasets.

Version = MD5 of the canonically serialized frame (sorted columns, no index), with
the library version recorded as provenance. Same contract as the CSV loader, so the
runner never knows which source it is looking at.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import sklearn
from sklearn import datasets as skd

from intelligence_sdk.core.filters import apply_filter
from intelligence_sdk.core.recipe import LibraryDataset
from intelligence_sdk.core.registry import LibraryName
from intelligence_sdk.core.types import DatasetVersion, LoadedDataset

ROW_ID = "row_id"

_LOADERS = {
    LibraryName.IRIS: skd.load_iris,
    LibraryName.WINE: skd.load_wine,
    LibraryName.BREAST_CANCER: skd.load_breast_cancer,
}


def _clean(name: str) -> str:
    return name.replace(" (cm)", "").replace(" ", "_").replace("/", "_")


class LibraryLoader:
    def load(self, definition: LibraryDataset) -> LoadedDataset:
        bunch = _LOADERS[definition.name]()
        frame = pd.DataFrame(bunch.data, columns=[_clean(c) for c in bunch.feature_names])
        frame[definition.target] = [str(bunch.target_names[i]) for i in bunch.target]
        frame.index = pd.RangeIndex(len(frame), name=ROW_ID)

        canonical = frame.sort_index(axis=1).to_csv(index=False).encode()
        digest = hashlib.md5(canonical).hexdigest()  # noqa: S324
        version = DatasetVersion(
            source="library",
            locator=definition.name.value,
            hash=digest,
            n_rows=len(frame),
            columns=[str(c) for c in frame.columns],
            provenance={"hash_algorithm": "md5(canonical csv)", "library": "scikit-learn", "library_version": sklearn.__version__},
        )
        frame = apply_filter(frame, definition.filter, "dataset.filter")
        return LoadedDataset(frame=frame, target=definition.target, version=version)
