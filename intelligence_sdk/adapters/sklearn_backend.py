"""scikit-learn implementation of the ``ModelBackend`` port.

This is the only module that names scikit-learn estimator classes. Config refers to
components by registry enum; the tables below are the allowlist that maps them.
A torch backend would be a sibling module subclassing the same interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from intelligence_sdk.core.recipe import StepSpec
from intelligence_sdk.core.registry import Model, Preprocessor
from intelligence_sdk.core.types import EstimatorKind, PredictionResult, ResultType
from intelligence_sdk.interfaces import ModelBackend

MODEL_FILE = "model.joblib"
META_FILE = "model.json"

_PREPROCESSORS: dict[Preprocessor, type] = {
    Preprocessor.STANDARD_SCALER: StandardScaler,
    Preprocessor.MIN_MAX_SCALER: MinMaxScaler,
    Preprocessor.SIMPLE_IMPUTER: SimpleImputer,
}

_MODELS: dict[Model, tuple[type, EstimatorKind]] = {
    Model.LOGISTIC_REGRESSION: (LogisticRegression, EstimatorKind.CLASSIFIER),
    Model.RANDOM_FOREST: (RandomForestClassifier, EstimatorKind.CLASSIFIER),
    Model.DECISION_TREE: (DecisionTreeClassifier, EstimatorKind.CLASSIFIER),
    Model.KNN: (KNeighborsClassifier, EstimatorKind.CLASSIFIER),
    Model.SVC: (SVC, EstimatorKind.CLASSIFIER),
    Model.GRADIENT_BOOSTING: (GradientBoostingClassifier, EstimatorKind.CLASSIFIER),
    Model.LINEAR_REGRESSION: (LinearRegression, EstimatorKind.REGRESSOR),
}


class SklearnBackend(ModelBackend):
    # ---- static knowledge (no data needed) ----
    def param_names(self, model: Model) -> set[str]:
        return set(_MODELS[model][0]().get_params())

    def step_param_names(self, step: Preprocessor) -> set[str]:
        return set(_PREPROCESSORS[step]().get_params())

    def estimator_kind(self, model: Model) -> EstimatorKind:
        return _MODELS[model][1]

    # ---- lifecycle ----
    def build(self, preprocessing: list[StepSpec], model: Model, params: dict[str, Any], seed: int) -> Pipeline:
        steps = [(spec.step.value, _PREPROCESSORS[spec.step](**spec.params)) for spec in preprocessing]
        cls, _ = _MODELS[model]
        estimator = cls(**params)
        if "random_state" in estimator.get_params() and "random_state" not in params:
            estimator.set_params(random_state=seed)
        steps.append(("model", estimator))
        return Pipeline(steps)

    def fit(self, estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        return estimator.fit(X, y)

    def predict(self, estimator: Pipeline, X: pd.DataFrame) -> PredictionResult:
        frame = pd.DataFrame({"prediction": estimator.predict(X)}, index=X.index)
        if is_classifier(estimator):
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
                for i, cls in enumerate(estimator.classes_):
                    frame[f"proba_{cls}"] = proba[:, i]
            return PredictionResult(frame=frame, result_type=ResultType.CLASS_LABEL)
        return PredictionResult(frame=frame, result_type=ResultType.CONTINUOUS)

    def save(self, estimator: Pipeline, model_dir: Path, metadata: dict[str, Any]) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, model_dir / MODEL_FILE)
        meta = {"backend": "sklearn", "artifact": MODEL_FILE, **metadata}
        (model_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str))

    def load(self, model_dir: Path) -> Pipeline:
        return joblib.load(model_dir / MODEL_FILE)
