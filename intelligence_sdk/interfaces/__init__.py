"""Ports: abstract base classes every adapter inherits from.

Dependency rule: ``interfaces`` imports only ``core.types``. ``core`` codes against
these classes, ``adapters`` subclass them, and ``runtime.providers`` is the only module that
names both sides.
"""

from intelligence_sdk.interfaces.artifact_store import ArtifactStore
from intelligence_sdk.interfaces.data_loader import DataLoader
from intelligence_sdk.interfaces.evaluator import Evaluator
from intelligence_sdk.interfaces.model_backend import ModelBackend

__all__ = ["ArtifactStore", "DataLoader", "Evaluator", "ModelBackend"]
