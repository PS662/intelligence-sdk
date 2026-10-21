"""Names the SDK owns. Config refers to components only by these enums; a backend
maps them to concrete library classes. Config can therefore never name an arbitrary
import path - the registry is the allowlist."""

from __future__ import annotations

from enum import StrEnum

from intelligence_sdk.core.types import EstimatorKind


class Preprocessor(StrEnum):
    STANDARD_SCALER = "standard_scaler"
    MIN_MAX_SCALER = "min_max_scaler"
    SIMPLE_IMPUTER = "simple_imputer"


class Model(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"
    KNN = "knn"
    SVC = "svc"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"


class Backend(StrEnum):
    """Model backends the runtime can build. Selected recipe-wide in ``runtime.backend``."""

    SKLEARN = "sklearn"


class ArtifactStoreKind(StrEnum):
    """Artifact stores the runtime can build. Selected in ``runtime.artifact_store``."""

    LOCAL = "local"


class LibraryName(StrEnum):
    """Datasets a ``source: library`` definition may name."""

    IRIS = "iris"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"


class Scoring(StrEnum):
    """Metrics with a known direction and a known task kind. Restricting ``scoring`` to
    this set lets grid search rank candidates correctly and lets the resolve phase
    reject a classification metric on a regressor before any data loads."""

    ACCURACY = "accuracy"
    F1_MACRO = "f1_macro"
    PRECISION_MACRO = "precision_macro"
    RECALL_MACRO = "recall_macro"
    NEG_RMSE = "neg_rmse"
    NEG_MAE = "neg_mae"
    R2 = "r2"

    @property
    def kind(self) -> EstimatorKind:
        return _SCORING_KIND[self]

    @property
    def higher_is_better(self) -> bool:
        # Every member is oriented so that higher is better (hence neg_rmse / neg_mae).
        return True


_SCORING_KIND = {
    Scoring.ACCURACY: EstimatorKind.CLASSIFIER,
    Scoring.F1_MACRO: EstimatorKind.CLASSIFIER,
    Scoring.PRECISION_MACRO: EstimatorKind.CLASSIFIER,
    Scoring.RECALL_MACRO: EstimatorKind.CLASSIFIER,
    Scoring.NEG_RMSE: EstimatorKind.REGRESSOR,
    Scoring.NEG_MAE: EstimatorKind.REGRESSOR,
    Scoring.R2: EstimatorKind.REGRESSOR,
}

DEFAULT_SCORING = {
    EstimatorKind.CLASSIFIER: [Scoring.ACCURACY, Scoring.F1_MACRO, Scoring.PRECISION_MACRO, Scoring.RECALL_MACRO],
    EstimatorKind.REGRESSOR: [Scoring.NEG_RMSE, Scoring.NEG_MAE, Scoring.R2],
}
