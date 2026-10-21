"""Recipe schema and loader.

A recipe is a YAML mapping with two reserved keys - ``datasets`` (a registry of named
dataset definitions) and ``runtime`` (which implementations run it) - and any number
of user-named *sections*. Each section carries a ``type`` that selects its schema;
new section types are added as members of the ``Section`` union.

Validation is fail-fast and every error names the config path that failed, with a
did-you-mean suggestion when one can be computed.

Section listing order drives implicit dependency chaining and topological tie-breaks
(see ``core.dag``); ``yaml.safe_load`` preserves the order the user wrote.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from intelligence_sdk.core.errors import RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.registry import ArtifactStoreKind, Backend, LibraryName, Model, Preprocessor, Scoring

RESERVED_KEYS = {"datasets", "runtime"}
SECTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


# --------------------------------------------------------------------------- base


class StrictModel(BaseModel):
    """Unknown keys are rejected by pydantic (``extra="forbid"``); the error converter
    below turns each one into a config-path error with a did-you-mean suggestion
    drawn from the failing model's own fields, while still reporting every other
    problem in the recipe in the same pass."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------ runtime


class RuntimeSpec(StrictModel):
    """Which implementations run the recipe. Optional, with defaults. Recipe-wide
    because a test must load a model with the backend that saved it."""

    backend: Backend = Backend.SKLEARN
    artifact_store: ArtifactStoreKind = ArtifactStoreKind.LOCAL


_runtime_adapter = TypeAdapter(RuntimeSpec)


# ----------------------------------------------------------------------- datasets


class FilterSpec(StrictModel):
    """Row filter by id. ``include`` is applied before ``exclude``. Each may be a path
    to a one-column CSV of ids or an inline list of ids."""

    include: str | list[int | str] | None = None
    exclude: str | list[int | str] | None = None


class FileDataset(StrictModel):
    source: Literal["file"]
    path: str
    target: str
    id_column: str | None = None
    filter: FilterSpec | None = None


class LibraryDataset(StrictModel):
    source: Literal["library"]
    name: LibraryName
    target: str
    filter: FilterSpec | None = None


DatasetDef = Annotated[FileDataset | LibraryDataset, Field(discriminator="source")]


class DatasetRef(StrictModel):
    """One entry of a section's ``datasets`` list: a registry reference
    (``dataset_name``) or an inline definition (``dataset``), never both."""

    dataset_name: str | None = None
    dataset: DatasetDef | None = None
    filter: FilterSpec | None = None
    save: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DatasetRef:
        if (self.dataset_name is None) == (self.dataset is None):
            raise ValueError("set exactly one of 'dataset_name' (registry reference) or 'dataset' (inline definition)")
        return self


def _exactly_one_dataset(refs: list[DatasetRef] | None, field: str = "datasets") -> None:
    if refs is None:
        return
    if len(refs) == 0:
        raise ValueError(f"'{field}' must list exactly one dataset")
    if len(refs) > 1:
        raise ValueError(f"'{field}' lists {len(refs)} datasets; multi-dataset sections are not supported yet")


# --------------------------------------------------------------------- components


class StepSpec(StrictModel):
    step: Preprocessor
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(StrictModel):
    name: Model
    params: dict[str, Any] = Field(default_factory=dict)

    def grid_params(self) -> dict[str, list[Any]]:
        """Parameters whose value is a list - the grid-search trigger."""
        return {k: v for k, v in self.params.items() if isinstance(v, list)}


class SplitSpec(StrictModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = True


class SearchSpec(StrictModel):
    cv: int = Field(default=3, ge=2)
    scoring: Scoring = Scoring.ACCURACY


# ----------------------------------------------------------------------- sections


class SectionBase(StrictModel):
    depends_on: list[str] = Field(default_factory=list)
    output_dir: str = "artifacts"


class TrainSection(SectionBase):
    type: Literal["train"]
    datasets: list[DatasetRef]
    model: ModelSpec
    preprocessing: list[StepSpec] = Field(default_factory=list)
    split: SplitSpec = Field(default_factory=SplitSpec)
    search: SearchSpec = Field(default_factory=SearchSpec)
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> TrainSection:
        _exactly_one_dataset(self.datasets)
        return self


class TestSection(SectionBase):
    """Scores a trained model.

    Model source: exactly one of an explicit ``depends_on`` naming a train section, or
    ``model_path``. Dataset: omitted (inherited from the model's recorded dataset,
    hash-checked, default ``rows: holdout``) or explicit (``rows: all`` by default,
    never refused - this is the inference-on-new-data path).
    """

    type: Literal["test"]
    model_path: str | None = None
    datasets: list[DatasetRef] | None = None
    rows: Literal["holdout", "all"] | None = None

    @model_validator(mode="after")
    def _check(self) -> TestSection:
        has_path = self.model_path is not None
        has_dep = len(self.depends_on) > 0
        if has_path == has_dep:
            raise ValueError(
                "a test section needs exactly one model source: either 'model_path' or an explicit "
                "'depends_on' naming a train section"
            )
        _exactly_one_dataset(self.datasets)
        return self


class EvalSection(SectionBase):
    type: Literal["eval"]
    datasets: list[DatasetRef]
    candidates: list[ModelSpec] = Field(min_length=1)
    preprocessing: list[StepSpec] = Field(default_factory=list)
    folds: int = Field(default=5, ge=2)
    stratify: bool = True
    metrics: list[Scoring] | None = None
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> EvalSection:
        _exactly_one_dataset(self.datasets)
        for i, cand in enumerate(self.candidates):
            lists = cand.grid_params()
            if lists:
                raise ValueError(
                    f"candidates[{i}].params.{next(iter(lists))}: eval candidates take scalar hyperparameters; "
                    "use a train section with a list value to grid-search"
                )
        return self


Section = Annotated[TrainSection | TestSection | EvalSection, Field(discriminator="type")]
SECTION_TYPES = ("train", "test", "eval")

_section_adapter = TypeAdapter(Section)
_dataset_adapter = TypeAdapter(DatasetDef)


# ------------------------------------------------------------------------- recipe


class Recipe(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = "<memory>"
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    datasets: dict[str, FileDataset | LibraryDataset] = Field(default_factory=dict)
    sections: dict[str, TrainSection | TestSection | EvalSection] = Field(default_factory=dict)

    @property
    def order(self) -> list[str]:
        """Section names in listing order."""
        return list(self.sections)

    def resolve_dataset(self, ref: DatasetRef, path: str) -> FileDataset | LibraryDataset:
        if ref.dataset is not None:
            return ref.dataset
        name = ref.dataset_name
        if name not in self.datasets:
            raise RecipeError(
                f"{path}.dataset_name",
                f"unknown dataset '{name}'; registry has: {', '.join(self.datasets) or '(none)'}",
                did_you_mean(name, self.datasets),
            )
        return self.datasets[name]


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    if not path.exists():
        raise RecipeError("", f"recipe file not found: {path}")
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise RecipeError("", f"recipe is not valid YAML: {exc}") from exc
    return parse_recipe(data, source=str(path))


def parse_recipe(data: Any, source: str = "<memory>") -> Recipe:
    if not isinstance(data, dict) or not data:
        raise RecipeError("", "recipe must be a non-empty mapping of section names to blocks")

    errors: list[RecipeError] = []
    datasets: dict[str, Any] = {}
    sections: dict[str, Any] = {}

    runtime = RuntimeSpec()
    if "runtime" in data:
        try:
            runtime = _runtime_adapter.validate_python(data["runtime"] or {})
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix="runtime", root=RuntimeSpec))

    raw_datasets = data.get("datasets") or {}
    if not isinstance(raw_datasets, dict):
        errors.append(RecipeError("datasets", "must be a mapping of dataset name to definition"))
        raw_datasets = {}
    for name, raw in raw_datasets.items():
        try:
            datasets[name] = _dataset_adapter.validate_python(raw)
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix=f"datasets.{name}", root=DatasetDef))

    section_items = [(k, v) for k, v in data.items() if k not in RESERVED_KEYS]
    if not section_items:
        errors.append(RecipeError("", "recipe has no sections; add at least one block with a 'type'"))
    for name, raw in section_items:
        if not SECTION_NAME_RE.match(str(name)):
            errors.append(
                RecipeError(
                    str(name),
                    "section names must be identifiers matching [A-Za-z_][A-Za-z0-9_-]* "
                    "(they become artifact directory names); 'datasets' and 'runtime' are reserved keys",
                )
            )
            continue
        if not isinstance(raw, dict):
            errors.append(RecipeError(name, "section must be a mapping with a 'type' field"))
            continue
        if "type" not in raw:
            errors.append(RecipeError(f"{name}.type", f"missing; one of {', '.join(SECTION_TYPES)}"))
            continue
        try:
            sections[name] = _section_adapter.validate_python(raw)
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix=name, root=Section))

    if errors:
        raise RecipeValidationError(errors)

    recipe = Recipe(source=source, runtime=runtime, datasets=datasets, sections=sections)
    _check_dataset_refs(recipe, errors)
    if errors:
        raise RecipeValidationError(errors)
    return recipe


def _check_dataset_refs(recipe: Recipe, errors: list[RecipeError]) -> None:
    for name, section in recipe.sections.items():
        refs = getattr(section, "datasets", None) or []
        for i, ref in enumerate(refs):
            try:
                recipe.resolve_dataset(ref, f"{name}.datasets[{i}]")
            except RecipeError as exc:
                errors.append(exc)


# ------------------------------------------------------------- error conversion


_QUOTED = re.compile(r"'([^']*)'")


def _unwrap(annotation: Any) -> list[Any]:
    """Candidate model classes reachable from a type annotation (through Annotated,
    Optional/Union, list[...] and dict[str, ...])."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _unwrap(get_args(annotation)[0])
    if origin in (Union, UnionType):
        out: list[Any] = []
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            out.extend(_unwrap(arg))
        return out
    if origin in (list, dict):
        args = get_args(annotation)
        return _unwrap(args[-1]) if args else []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def _fields_at(root: Any, loc: tuple[Any, ...]) -> set[str] | None:
    """Field names of the model that pydantic was validating at ``loc``.

    Pydantic's loc interleaves field names, list/dict keys and, for discriminated
    unions, the chosen tag. We walk the annotations the same way.
    """
    candidates = _unwrap(root)
    for item in loc:
        if not candidates:
            return None
        if isinstance(item, int):
            continue  # list index: the element type was already unwrapped
        tagged = [c for c in candidates if _tag_of(c) == item]
        if tagged:
            candidates = tagged
            continue
        if len(candidates) == 1 and isinstance(item, str) and item in candidates[0].model_fields:
            candidates = _unwrap(candidates[0].model_fields[item].annotation)
            continue
        if isinstance(item, str) and not any(item in c.model_fields for c in candidates):
            continue  # dict key (e.g. a dataset name) - element type already unwrapped
        return None
    if len(candidates) == 1:
        return set(candidates[0].model_fields)
    return None


def _tag_of(model_cls: Any) -> str | None:
    for disc in ("type", "source"):
        field = model_cls.model_fields.get(disc)
        if field is not None and get_origin(field.annotation) is Literal:
            return get_args(field.annotation)[0]
    return None


def _loc_to_path(prefix: str, loc: tuple[Any, ...]) -> str:
    parts = [prefix] if prefix else []
    for item in loc:
        # Pydantic inserts the chosen union tag (e.g. 'train') into loc; it is not a config key.
        if isinstance(item, str) and item in SECTION_TYPES + ("file", "library"):
            continue
        if isinstance(item, int):
            parts[-1] = f"{parts[-1]}[{item}]" if parts else f"[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)


def _convert(exc: ValidationError, prefix: str, root: Any) -> list[RecipeError]:
    out: list[RecipeError] = []
    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        etype = err.get("type", "")
        msg = err.get("msg", "invalid value")
        ctx = err.get("ctx", {}) or {}
        value = err.get("input")
        suggestion = None

        if etype == "value_error":
            msg = msg.removeprefix("Value error, ")
            m2 = re.match(r"(candidates\[\d+\]\.params\.\w+): (.*)", msg)
            if m2:
                loc = loc + tuple(m2.group(1).split("."))
                msg = m2.group(2)
        elif etype == "enum":
            expected = _QUOTED.findall(str(ctx.get("expected", "")))
            suggestion = did_you_mean(str(value), expected)
            msg = f"'{value}' is not a known value; choose one of: {', '.join(expected)}"
        elif etype == "union_tag_invalid":
            tag = ctx.get("tag")
            expected = _QUOTED.findall(str(ctx.get("expected_tags", "")))
            loc = loc + (ctx.get("discriminator", "type").strip("'"),)
            suggestion = did_you_mean(str(tag), expected)
            msg = f"'{tag}' is not a known type; choose one of: {', '.join(expected)}"
        elif etype == "union_tag_not_found":
            disc = ctx.get("discriminator", "'type'").strip("'")
            loc = loc + (disc,)
            msg = f"missing required field '{disc}'"
        elif etype == "missing":
            msg = "missing required field"
        elif etype == "extra_forbidden":
            siblings = _fields_at(root, loc[:-1])
            msg = f"unknown field '{loc[-1]}'"
            if siblings:
                suggestion = did_you_mean(str(loc[-1]), siblings)
                msg += f"; valid fields: {', '.join(sorted(siblings))}"

        out.append(RecipeError(_loc_to_path(prefix, loc), msg, suggestion))
    return out


def parse_dataset_def(data: dict[str, Any], path: str = "dataset") -> FileDataset | LibraryDataset:
    """Re-validate a dataset definition recorded in a run's ``dataset.json``."""
    try:
        return _dataset_adapter.validate_python(data)
    except ValidationError as exc:
        raise RecipeValidationError(_convert(exc, prefix=path, root=DatasetDef)) from exc
