from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from intelligence_sdk.core.types import LoadedDataset

if TYPE_CHECKING:  # schema types only for annotations; no runtime import of core.recipe
    from intelligence_sdk.core.recipe import FileDataset, LibraryDataset


class DataLoader(Protocol):
    """Materializes one dataset definition.

    Contract: the returned frame is indexed by stable row identifiers (original file
    index or the configured id column), the definition-level filter is already
    applied, and ``version`` identifies the *unfiltered* source content.
    """

    def load(self, definition: "FileDataset | LibraryDataset") -> LoadedDataset: ...
