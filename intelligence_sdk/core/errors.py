"""User-facing errors. Every error names the config path that failed and, where we
can compute one, a did-you-mean suggestion. The CLI prints these on one line and
never shows a traceback for them."""
from __future__ import annotations

import difflib
from collections.abc import Iterable


def did_you_mean(word: str, candidates: Iterable[str], cutoff: float = 0.6) -> str | None:
    matches = difflib.get_close_matches(str(word), [str(c) for c in candidates], n=1, cutoff=cutoff)
    return matches[0] if matches else None


class RecipeError(Exception):
    """A single problem with a recipe, anchored to a dotted config path."""

    def __init__(self, path: str, message: str, suggestion: str | None = None):
        self.path = path
        self.message = message
        self.suggestion = suggestion
        super().__init__(str(self))

    def __str__(self) -> str:
        text = f"{self.path}: {self.message}" if self.path else self.message
        if self.suggestion:
            text += f" (did you mean '{self.suggestion}'?)"
        return text


class RecipeValidationError(RecipeError):
    """Several problems reported together so the user fixes them in one pass."""

    def __init__(self, errors: list[RecipeError]):
        self.errors = errors
        first = errors[0] if errors else RecipeError("", "invalid recipe")
        super().__init__(first.path, first.message, first.suggestion)

    def __str__(self) -> str:
        if len(self.errors) == 1:
            return RecipeError.__str__(self.errors[0])
        lines = [f"recipe has {len(self.errors)} problems:"]
        lines += [f"  - {RecipeError.__str__(e)}" for e in self.errors]
        return "\n".join(lines)


class DatasetVersionMismatch(RecipeError):
    """Test asked for the holdout of a model but the dataset on disk is not the one the
    model was trained on. Re-deriving a split on different data is silent leakage."""
