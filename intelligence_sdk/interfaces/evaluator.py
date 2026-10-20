from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from intelligence_sdk.core.types import PredictionResult

if TYPE_CHECKING:
    from intelligence_sdk.core.registry import Scoring


class Evaluator(Protocol):
    """Scores predictions against truth. Keyed on ``PredictionResult.result_type`` so
    one array-based implementation serves every backend; a backend that must score
    GPU-resident tensors or streams implements this protocol instead."""

    def evaluate(
        self, y_true: np.ndarray, prediction: PredictionResult, metrics: list["Scoring"] | None = None
    ) -> dict[str, Any]: ...
