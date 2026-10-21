"""``source: library`` loader for scikit-learn's bundled datasets."""

from __future__ import annotations

import pandas as pd
from sklearn import datasets as skd

from intelligence_sdk.core.recipe import LibraryDataset
from intelligence_sdk.core.registry import LibraryName
from intelligence_sdk.core.types import Dataset
from intelligence_sdk.interfaces import DataLoader

ROW_ID = "row_id"

_LOADERS = {
    LibraryName.IRIS: skd.load_iris,
    LibraryName.WINE: skd.load_wine,
    LibraryName.BREAST_CANCER: skd.load_breast_cancer,
}


def _clean(name: str) -> str:
    return name.replace(" (cm)", "").replace(" ", "_").replace("/", "_")


class LibraryLoader(DataLoader):
    def load(self, spec: LibraryDataset, name: str) -> Dataset:
        bunch = _LOADERS[spec.dataset]()
        frame = pd.DataFrame(bunch.data, columns=[_clean(c) for c in bunch.feature_names])
        frame[spec.target] = [str(bunch.target_names[i]) for i in bunch.target]
        frame.index = pd.RangeIndex(len(frame), name=ROW_ID)
        return Dataset(name=name, frame=frame, target=spec.target)
