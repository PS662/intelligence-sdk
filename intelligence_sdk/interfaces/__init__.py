"""Ports. One file per contract; this module re-exports them.

Dependency rule: ``interfaces`` imports only ``core.types``. ``core`` implements the
domain against these protocols; ``adapters`` implement them; ``factory`` is the only
module that names both.
"""
from intelligence_sdk.interfaces.artifact_store import ArtifactStore
from intelligence_sdk.interfaces.data_loader import DataLoader
from intelligence_sdk.interfaces.evaluator import Evaluator
from intelligence_sdk.interfaces.model_backend import ModelBackend

__all__ = ["ArtifactStore", "DataLoader", "Evaluator", "ModelBackend"]
