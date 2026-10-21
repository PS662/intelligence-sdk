from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from intelligence_sdk.core.types import EstimatorKind, PredictionResult

if TYPE_CHECKING:
    from intelligence_sdk.core.recipe import StepSpec
    from intelligence_sdk.core.registry import Model, Preprocessor


class ModelBackend(ABC):
    """Turns registry names into a trainable estimator and back into predictions.

    The backend contracts to return a ``PredictionResult`` whose ``result_type`` says
    what the frame holds; that is what lets one array-based evaluator serve every
    backend. A torch backend is another subclass.
    """

    @abstractmethod
    def param_names(self, model: Model) -> set[str]:
        """Valid hyperparameter names for ``model`` - used for did-you-mean at resolve time."""
        ...

    @abstractmethod
    def step_param_names(self, step: Preprocessor) -> set[str]:
        """Valid parameter names for a preprocessing step."""
        ...

    @abstractmethod
    def estimator_kind(self, model: Model) -> EstimatorKind:
        """Classifier or regressor, known statically (no data needed)."""
        ...

    @abstractmethod
    def build(self, preprocessing: list[StepSpec], model: Model, params: dict[str, Any], seed: int) -> Any: ...

    @abstractmethod
    def fit(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> Any: ...

    @abstractmethod
    def predict(self, estimator: Any, X: pd.DataFrame) -> PredictionResult: ...

    @abstractmethod
    def save(self, estimator: Any, model_dir: Path, metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def load(self, model_dir: Path) -> Any: ...
