from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from intelligence_sdk.core.types import LoadedDataset

if TYPE_CHECKING:  # schema types only for annotations; no runtime import of core.recipe
    from intelligence_sdk.core.recipe import FileDataset, LibraryDataset


class DataLoader(ABC):
    """Materializes one dataset definition.

    Contract: the returned frame is indexed by stable row identifiers (original file
    index or the configured id column), the definition-level filter is already
    applied, and ``version`` identifies the *unfiltered* source content.
    """

    @abstractmethod
    def load(self, definition: FileDataset | LibraryDataset) -> LoadedDataset: ...
