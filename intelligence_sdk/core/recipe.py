"""Recipe schema and loader.

A recipe is a YAML mapping with two reserved keys - ``datasets`` (a registry of named
dataset specs) and ``runtime`` (which implementations run it) - and any number of
user-named *nodes*. Each node carries a ``type`` that selects its schema: ``dataset``,
``train``, ``test`` or ``eval``. New node types are added as members of the ``Node``
union.

Datasets are referential. A node's ``datasets`` list names registry entries or
dataset nodes (``dataset_name``) or defines datasets inline (``name`` + spec). Every
spec carries its own operations (filter, split, folds, save) and every node carries
the same operations at its root, applied to the consolidated dataset it receives.

Validation is fail-fast and every error names the config path that failed, with a
did-you-mean suggestion when one can be computed. Node listing order is kept.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, ClassVar, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, ValidationError, model_validator

from intelligence_sdk.core.errors import RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.registry import ArtifactStoreKind, Backend, LibraryName, Model, Preprocessor, Scoring

RESERVED_KEYS = {"datasets", "runtime"}
NODE_TYPES = ("dataset", "train", "test", "eval")
SOURCE_TYPES = ("file", "library", "dataset")
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class StrictModel(BaseModel):
    """Unknown keys are rejected by pydantic (``extra="forbid"``); the error converter
    below turns each one into a config-path error with a did-you-mean suggestion."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------ runtime


class RuntimeSpec(StrictModel):
    """Which implementations run the recipe. Optional, with defaults. Recipe-wide
    because a test must load a model with the backend that saved it."""

    backend: Backend = Backend.SKLEARN
    artifact_store: ArtifactStoreKind = ArtifactStoreKind.LOCAL


# --------------------------------------------------------------- dataset options


class SplitRef(StrictModel):
    """Filter by a recorded split instead of an id list. Only a saved dataset
    (``source: dataset``) carries a split to refer to."""

    split: Literal["train", "test"]


FilterIds = str | list[int | str] | SplitRef


class FilterSpec(StrictModel):
    """Row filter by id. ``include`` is applied before ``exclude``. Each is a path to a
    one-column CSV of ids, an inline list of ids, or ``{split: train|test}``."""

    include: FilterIds | None = None
    exclude: FilterIds | None = None

    def uses_split(self) -> bool:
        return isinstance(self.include, SplitRef) or isinstance(self.exclude, SplitRef)


class SplitSpec(StrictModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = True


class FoldsSpec(StrictModel):
    k: int = Field(default=5, ge=2)
    stratify: bool = True


class SaveSpec(StrictModel):
    """Where to save a dataset. When ``path`` already holds one: ``overwrite`` rebuilds
    it, ``reuse`` loads it instead of building, neither is an error."""

    path: str
    overwrite: bool = False
    reuse: bool = False

    @model_validator(mode="after")
    def _one_policy(self) -> SaveSpec:
        if self.overwrite and self.reuse:
            raise ValueError("set at most one of 'overwrite' and 'reuse'")
        return self


class OpsMixin(StrictModel):
    """Operations applied, in this order, to the dataset they are declared on."""

    filter: FilterSpec | None = None
    split: SplitSpec | None = None
    folds: FoldsSpec | None = None
    seed: int = 42


class DatasetOps(OpsMixin):
    save_dataset: SaveSpec | None = None

    @model_validator(mode="after")
    def _split_filter_needs_saved_source(self) -> DatasetOps:
        if self.filter is not None and self.filter.uses_split() and getattr(self, "source", None) != "dataset":
            raise ValueError("filter: a 'split' reference is only valid on a dataset with source: dataset")
        return self


class FileDataset(DatasetOps):
    source: Literal["file"]
    path: str
    target: str
    id_column: str | None = None


class LibraryDataset(DatasetOps):
    source: Literal["library"]
    dataset: LibraryName
    target: str


class SavedDataset(DatasetOps):
    """A dataset previously saved by this tool (``dataset.csv`` + ``metadata.json``)."""

    source: Literal["dataset"]
    path: str


DatasetSpec = Annotated[FileDataset | LibraryDataset | SavedDataset, Field(discriminator="source")]
AnyDataset = FileDataset | LibraryDataset | SavedDataset


# ------------------------------------------------------------------ dataset entries


class RefEntry(StrictModel):
    """Reference to a registry entry or a dataset node by name."""

    _tag: ClassVar[str] = "ref"
    dataset_name: str


class InlineEntry(BaseModel):
    """A dataset defined in place: ``name`` plus the fields of a dataset spec."""

    _tag: ClassVar[str] = "inline"
    model_config = ConfigDict(extra="forbid")
    name: str
    spec: DatasetSpec

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if isinstance(data, dict) and "spec" not in data:
            data = dict(data)
            name = data.pop("name", None)
            if name is None:
                hint = did_you_mean("dataset_name", [str(k) for k in data], cutoff=0.75)
                if hint:
                    raise ValueError(f"unknown field '{hint}' (did you mean 'dataset_name'?)")
                raise ValueError("an inline dataset needs a 'name' (or use 'dataset_name' to reference one)")
            return {"name": name, "spec": data}
        return data


def _entry_kind(value: Any) -> str:
    return "ref" if isinstance(value, dict) and "dataset_name" in value else "inline"


DatasetEntry = Annotated[
    Annotated[RefEntry, Tag("ref")] | Annotated[InlineEntry, Tag("inline")],
    Discriminator(_entry_kind),
]


# -------------------------------------------------------------------- components


class StepSpec(StrictModel):
    step: Preprocessor
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(StrictModel):
    name: Model
    params: dict[str, Any] = Field(default_factory=dict)

    def grid_params(self) -> dict[str, list[Any]]:
        """Parameters whose value is a list - the search trigger."""
        return {k: v for k, v in self.params.items() if isinstance(v, list)}


class SearchSpec(StrictModel):
    """Cross-validated search over list-valued hyperparameters, folded over the
    rows the model is fitted on."""

    cv: int = Field(default=3, ge=2)
    scoring: Scoring = Scoring.ACCURACY


# ----------------------------------------------------------------------- outputs


class OutputSpec(StrictModel):
    save_path: str
    overwrite: bool = False


class DatasetOutput(OutputSpec):
    """A dataset node's output is a saved dataset, with the same policy as ``save_dataset``."""

    reuse: bool = False

    @model_validator(mode="after")
    def _one_policy(self) -> DatasetOutput:
        if self.overwrite and self.reuse:
            raise ValueError("set at most one of 'overwrite' and 'reuse'")
        return self


class TrainOutput(OutputSpec):
    pass


class TestOutput(OutputSpec):
    pass


class EvalOutput(OutputSpec):
    pass


# ------------------------------------------------------------------------- nodes


class NodeBase(OpsMixin):
    """Every node consumes a ``datasets`` list, consolidates it into one dataset and
    applies its root-level operations to that dataset before acting."""

    datasets: list[DatasetEntry] = Field(min_length=1)
    ignore_duplicates: bool = False

    @model_validator(mode="after")
    def _root_filter_has_no_split_ref(self) -> NodeBase:
        if self.filter is not None and self.filter.uses_split():
            raise ValueError("filter: a 'split' reference is only valid inside a source: dataset entry")
        return self

    def entry_names(self) -> list[str]:
        return [e.name for e in self.datasets if isinstance(e, InlineEntry)]

    def references(self) -> list[str]:
        return [e.dataset_name for e in self.datasets if isinstance(e, RefEntry)]


class DatasetNode(NodeBase):
    """Consolidates its datasets and saves the result. Runs when referenced or when
    named with ``--node``."""

    type: Literal["dataset"]
    recipe_output: DatasetOutput | None = None


class ActionNode(NodeBase):
    depends_on: list[str] = Field(default_factory=list)
    save_dataset: SaveSpec | None = None


class TrainNode(ActionNode):
    type: Literal["train"]
    model: ModelSpec
    preprocessing: list[StepSpec] = Field(default_factory=list)
    hyperparam_search: SearchSpec = Field(default_factory=SearchSpec)
    recipe_output: TrainOutput | None = None


class TestNode(ActionNode):
    """Scores a trained model on every row of its dataset. The model comes from
    exactly one of an explicit ``depends_on`` naming a train node, or ``model_path``."""

    type: Literal["test"]
    model_path: str | None = None
    recipe_output: TestOutput | None = None

    @model_validator(mode="after")
    def _one_model_source(self) -> TestNode:
        if (self.model_path is not None) == (len(self.depends_on) > 0):
            raise ValueError(
                "a test node needs exactly one model source: either 'model_path' or an explicit "
                "'depends_on' naming a train node"
            )
        return self


class EvalNode(ActionNode):
    """Compares candidate models with the folds recorded on its dataset."""

    type: Literal["eval"]
    candidates: list[ModelSpec] = Field(min_length=1)
    preprocessing: list[StepSpec] = Field(default_factory=list)
    metrics: list[Scoring] | None = None
    recipe_output: EvalOutput | None = None

    @model_validator(mode="after")
    def _scalar_candidates(self) -> EvalNode:
        for i, cand in enumerate(self.candidates):
            lists = cand.grid_params()
            if lists:
                raise ValueError(
                    f"candidates[{i}].params.{next(iter(lists))}: eval candidates take scalar hyperparameters; "
                    "use a train node with a list value to search"
                )
        return self


Node = Annotated[DatasetNode | TrainNode | TestNode | EvalNode, Field(discriminator="type")]
AnyNode = DatasetNode | TrainNode | TestNode | EvalNode
ActionNodeTypes = (TrainNode, TestNode, EvalNode)

_node_adapter = TypeAdapter(Node)
_dataset_adapter = TypeAdapter(DatasetSpec)
_runtime_adapter = TypeAdapter(RuntimeSpec)


# ------------------------------------------------------------------------- recipe


class Recipe(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = "<memory>"
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    datasets: dict[str, AnyDataset] = Field(default_factory=dict)
    nodes: dict[str, AnyNode] = Field(default_factory=dict)

    @property
    def order(self) -> list[str]:
        """Node names in listing order."""
        return list(self.nodes)

    @property
    def dataset_nodes(self) -> dict[str, DatasetNode]:
        return {n: node for n, node in self.nodes.items() if isinstance(node, DatasetNode)}

    @property
    def action_nodes(self) -> dict[str, TrainNode | TestNode | EvalNode]:
        return {n: node for n, node in self.nodes.items() if isinstance(node, ActionNodeTypes)}

    def referenceable(self) -> list[str]:
        return list(self.datasets) + list(self.dataset_nodes)

    def lookup(self, name: str, path: str) -> AnyDataset | DatasetNode:
        """The registry spec or dataset node a ``dataset_name`` refers to."""
        if name in self.datasets:
            return self.datasets[name]
        if name in self.dataset_nodes:
            return self.dataset_nodes[name]
        if name in self.nodes:
            raise RecipeError(path, f"'{name}' is a {self.nodes[name].type} node; only datasets can be referenced")
        raise RecipeError(
            path,
            f"unknown dataset '{name}'; known: {', '.join(self.referenceable()) or '(none)'}",
            did_you_mean(name, self.referenceable()),
        )


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
        raise RecipeError("", "recipe must be a non-empty mapping of node names to blocks")

    errors: list[RecipeError] = []

    runtime = RuntimeSpec()
    if "runtime" in data:
        try:
            runtime = _runtime_adapter.validate_python(data["runtime"] or {})
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix="runtime", root=RuntimeSpec))

    datasets: dict[str, Any] = {}
    raw_datasets = data.get("datasets") or {}
    if not isinstance(raw_datasets, dict):
        errors.append(RecipeError("datasets", "must be a mapping of dataset name to spec"))
        raw_datasets = {}
    for name, raw in raw_datasets.items():
        if not NAME_RE.match(str(name)):
            errors.append(
                RecipeError(f"datasets.{name}", "dataset names must be identifiers matching [A-Za-z_][A-Za-z0-9_-]*")
            )
            continue
        try:
            datasets[name] = _dataset_adapter.validate_python(raw)
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix=f"datasets.{name}", root=DatasetSpec))

    nodes: dict[str, Any] = {}
    node_items = [(k, v) for k, v in data.items() if k not in RESERVED_KEYS]
    if not node_items:
        errors.append(RecipeError("", "recipe has no nodes; add at least one block with a 'type'"))
    for name, raw in node_items:
        if not NAME_RE.match(str(name)):
            errors.append(
                RecipeError(
                    str(name),
                    "node names must be identifiers matching [A-Za-z_][A-Za-z0-9_-]* "
                    "(they become directory names); 'datasets' and 'runtime' are reserved keys",
                )
            )
            continue
        if not isinstance(raw, dict):
            errors.append(RecipeError(name, "node must be a mapping with a 'type' field"))
            continue
        if "type" not in raw:
            errors.append(RecipeError(f"{name}.type", f"missing; one of {', '.join(NODE_TYPES)}"))
            continue
        try:
            nodes[name] = _node_adapter.validate_python(raw)
        except ValidationError as exc:
            errors.extend(_convert(exc, prefix=name, root=Node))

    if errors:
        raise RecipeValidationError(errors)

    recipe = Recipe(source=source, runtime=runtime, datasets=datasets, nodes=nodes)
    _check_namespace(recipe, errors)
    _check_references(recipe, errors)
    if errors:
        raise RecipeValidationError(errors)
    return recipe


def _check_namespace(recipe: Recipe, errors: list[RecipeError]) -> None:
    """Registry keys, node names and inline names share one namespace."""
    seen: dict[str, str] = {name: f"datasets.{name}" for name in recipe.datasets}
    for name in recipe.nodes:
        if name in seen:
            errors.append(RecipeError(name, f"name already used by {seen[name]}"))
        seen[name] = name
    for node_name, node in recipe.nodes.items():
        for i, entry in enumerate(node.datasets):
            if isinstance(entry, InlineEntry):
                where = f"{node_name}.datasets[{i}].name"
                if not NAME_RE.match(entry.name):
                    errors.append(
                        RecipeError(where, "dataset names must be identifiers matching [A-Za-z_][A-Za-z0-9_-]*")
                    )
                elif entry.name in seen:
                    errors.append(RecipeError(where, f"name '{entry.name}' already used by {seen[entry.name]}"))
                seen[entry.name] = where


def _check_references(recipe: Recipe, errors: list[RecipeError]) -> None:
    for node_name, node in recipe.nodes.items():
        for i, entry in enumerate(node.datasets):
            if isinstance(entry, RefEntry):
                try:
                    recipe.lookup(entry.dataset_name, f"{node_name}.datasets[{i}].dataset_name")
                except RecipeError as exc:
                    errors.append(exc)


def parse_dataset_spec(data: dict[str, Any], path: str = "dataset") -> AnyDataset:
    """Validate a dataset spec given as a plain mapping."""
    try:
        return _dataset_adapter.validate_python(data)
    except ValidationError as exc:
        raise RecipeValidationError(_convert(exc, prefix=path, root=DatasetSpec)) from exc


# ------------------------------------------------------------- error conversion


_QUOTED = re.compile(r"'([^']*)'")
_HIDDEN_FIELDS = {"spec"}  # wrapper fields that do not appear in the user's YAML


def _unwrap(annotation: Any) -> list[Any]:
    """Model classes reachable from a type annotation (through Annotated,
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


def _tag_of(model_cls: Any) -> str | None:
    explicit = getattr(model_cls, "_tag", None)
    if isinstance(explicit, str):
        return explicit
    for disc in ("type", "source"):
        field = model_cls.model_fields.get(disc)
        if field is not None and get_origin(field.annotation) is Literal:
            return get_args(field.annotation)[0]
    return None


def _walk(root: Any, loc: tuple[Any, ...]) -> tuple[list[str], set[str] | None]:
    """Turn pydantic's loc into the user's config path and find the fields of the
    model at its end. Pydantic's loc interleaves field names, list indexes, dict keys
    and union tags; tags are recognised by position and dropped from the path."""
    candidates = _unwrap(root)
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
            continue
        item = str(item)
        fielded = [c for c in candidates if item in c.model_fields]
        tagged = [c for c in candidates if _tag_of(c) == item]
        if tagged and not fielded:
            candidates = tagged
            continue
        if fielded:
            if item not in _HIDDEN_FIELDS:
                parts.append(item)
            candidates = _unwrap(fielded[0].model_fields[item].annotation)
            continue
        parts.append(item)  # dict key (a dataset name); element type already unwrapped
    siblings = set(candidates[0].model_fields) - _HIDDEN_FIELDS if len(candidates) == 1 else None
    return parts, siblings


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
            m3 = re.match(r"(filter): (.*)", msg)
            if m3:
                loc = loc + ("filter",)
                msg = m3.group(2)
        elif etype == "enum":
            expected = _QUOTED.findall(str(ctx.get("expected", "")))
            suggestion = did_you_mean(str(value), expected)
            msg = f"'{value}' is not a known value; choose one of: {', '.join(expected)}"
        elif etype == "union_tag_invalid":
            tag = ctx.get("tag")
            expected = _QUOTED.findall(str(ctx.get("expected_tags", "")))
            loc = loc + (str(ctx.get("discriminator", "type")).strip("'"),)
            suggestion = did_you_mean(str(tag), expected)
            msg = f"'{tag}' is not a known value; choose one of: {', '.join(expected)}"
        elif etype == "union_tag_not_found":
            disc = str(ctx.get("discriminator", "'type'")).strip("'")
            loc = loc + (disc,)
            msg = f"missing required field '{disc}'"
        elif etype == "missing":
            msg = "missing required field"
        elif etype == "too_short":
            msg = f"needs at least {ctx.get('min_length', 1)} item(s)"

        parts, siblings = _walk(root, loc[:-1] if etype == "extra_forbidden" else loc)
        if etype == "extra_forbidden":
            parts.append(str(loc[-1]))
            msg = f"unknown field '{loc[-1]}'"
            if siblings:
                suggestion = did_you_mean(str(loc[-1]), siblings)
                msg += f"; valid fields: {', '.join(sorted(siblings))}"

        path = ".".join(([prefix] if prefix else []) + parts)
        out.append(RecipeError(path, msg, suggestion))
    return out
