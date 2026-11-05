"""Schemas and exceptions for Weight-Space Programming.

Defines declarative, versioned, reversible patch structures and safety budgets.
Base models are strictly immutable; all interventions are declarative diffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence
import numpy as np


class BudgetViolationError(Exception):
    """Raised when a candidate patch exceeds parameter-delta or tensor count budgets."""


class AllowlistViolationError(Exception):
    """Raised when a candidate patch attempts to modify a tensor not on the allowlist."""


@dataclass
class PatchBudget:
    """Safety guardrail defining parameter bounds and tensor allowlists."""
    max_changed_tensors: int = 5
    max_l2_norm: float = 10.0
    max_relative_norm: float = 0.20
    allowed_tensor_names: Sequence[str] | None = None


@dataclass
class WeightPatch:
    """Declarative parameter delta patch for an open-weight model."""
    patch_id: str
    base_model_id: str
    base_model_revision: str
    deltas: dict[str, np.ndarray]  # tensor_name -> delta numpy array
    hypothesis: str
    proposed_by: str = "specialist_optimizer"
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def sha256(self) -> str:
        """Compute deterministic SHA256 of the patch contents."""
        hasher = hashlib.sha256()
        hasher.update(self.patch_id.encode("utf-8"))
        hasher.update(self.base_model_id.encode("utf-8"))
        hasher.update(self.base_model_revision.encode("utf-8"))
        for name in sorted(self.deltas.keys()):
            hasher.update(name.encode("utf-8"))
            hasher.update(self.deltas[name].tobytes())
        return hasher.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Summarize changed tensors and delta norms."""
        tensor_summaries = {}
        total_l2 = 0.0
        for name, delta in self.deltas.items():
            l2 = float(np.linalg.norm(delta))
            total_l2 += l2 ** 2
            tensor_summaries[name] = {
                "shape": list(delta.shape),
                "numel": int(delta.size),
                "l2_norm": l2,
            }
        return {
            "patch_id": self.patch_id,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "patch_sha256": self.sha256(),
            "num_tensors_modified": len(self.deltas),
            "total_l2_norm": float(np.sqrt(total_l2)),
            "tensors": tensor_summaries,
            "hypothesis": self.hypothesis,
            "proposed_by": self.proposed_by,
        }
