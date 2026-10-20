"""Default ``Evaluator``: array-based metrics keyed on ``ResultType``.

Because every backend contracts to return predictions as arrays with a declared
result type, this one implementation serves all of them. A backend that must score
GPU-resident tensors or streams would implement the Evaluator port itself.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skm

from intelligence_sdk.core.registry import DEFAULT_SCORING, Scoring
from intelligence_sdk.core.types import EstimatorKind, PredictionResult, ResultType

_CLASSIFICATION = {
    Scoring.ACCURACY: lambda t, p: skm.accuracy_score(t, p),
    Scoring.F1_MACRO: lambda t, p: skm.f1_score(t, p, average="macro", zero_division=0),
    Scoring.PRECISION_MACRO: lambda t, p: skm.precision_score(t, p, average="macro", zero_division=0),
    Scoring.RECALL_MACRO: lambda t, p: skm.recall_score(t, p, average="macro", zero_division=0),
}
_REGRESSION = {
    Scoring.NEG_RMSE: lambda t, p: -float(np.sqrt(skm.mean_squared_error(t, p))),
    Scoring.NEG_MAE: lambda t, p: -skm.mean_absolute_error(t, p),
    Scoring.R2: lambda t, p: skm.r2_score(t, p),
}


class DefaultEvaluator:
    def evaluate(
        self, y_true: np.ndarray, prediction: PredictionResult, metrics: list[Scoring] | None = None
    ) -> dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = prediction.values
        if prediction.result_type == ResultType.CLASS_LABEL:
            table, kind = _CLASSIFICATION, EstimatorKind.CLASSIFIER
        elif prediction.result_type == ResultType.CONTINUOUS:
            table, kind = _REGRESSION, EstimatorKind.REGRESSOR
        else:
            raise NotImplementedError(f"DefaultEvaluator cannot score {prediction.result_type.value} outputs")

        chosen = metrics or DEFAULT_SCORING[kind]
        bad = [m for m in chosen if m.kind != kind]
        if bad:
            raise ValueError(f"metric(s) {[m.value for m in bad]} do not apply to {kind.value} outputs")

        out: dict[str, Any] = {m.value: float(table[m](y_true, y_pred)) for m in chosen}
        out["n_samples"] = int(len(y_true))
        if kind == EstimatorKind.CLASSIFIER:
            labels = sorted({str(v) for v in np.concatenate([y_true.astype(str), y_pred.astype(str)])})
            cm = skm.confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
            out["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}
        return out
