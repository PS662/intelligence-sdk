from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from intelligence_sdk.core.types import Dataset

if TYPE_CHECKING:  # schema types only for annotations; no runtime import of core.recipe
    from intelligence_sdk.core.recipe import FileDataset, LibraryDataset, SavedDataset


class DataLoader(ABC):
    """Materializes one dataset spec into a raw ``Dataset``.

    Contract: the frame is indexed by unique row ids (positional ``row_id`` or the
    configured id column). Filters, split and folds are applied afterwards by the
    resolver; only a saved dataset arrives with its recorded split and folds.
    """

    @abstractmethod
    def load(self, spec: FileDataset | LibraryDataset | SavedDataset, name: str) -> Dataset: ...
