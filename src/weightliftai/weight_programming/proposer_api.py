"""Proposer API Environment for Blind Weight-Repair Benchmark.

Provides a clean, narrow capability interface to the candidate repair agent.
The proposer receives:
1. A cloned corrupted model
2. High-level architecture metadata
3. Permitted diagnostic failure examples and labels ($D_\\text{diag}$)
4. Pre-registered PatchBudget constraints

The proposer NEVER receives:
1. Clean ground-truth model weights
2. Defect creation mask or coordinate metadata
3. Sealed held-out test splits, OOD splits, or regression suites
4. Evaluator object instances or memory references

All evaluations execute across a subprocess boundary with a strict single-submission budget.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch
from weightliftai.weight_programming.toy_defect import ToyMLP, generate_synthetic_data


class EvaluationBudgetExceededError(RuntimeError):
    """Raised when a proposer attempts to query the hidden evaluator multiple times."""


class ProposerIsolationViolationError(AttributeError):
    """Raised when an agent attempts to access sealed benchmark secrets directly."""


class BlindProposerEnvironment:
    """Narrow proposer environment enforcing process-separated blind evaluation."""

    def __init__(self, seed: int = 42, max_queries: int = 1):
        self.seed = seed
        self._max_queries = max_queries
        self._queries_used = 0

        # Generate ONLY diagnostic data in the proposer process
        torch.manual_seed(seed)
        data = generate_synthetic_data(seed=seed)
        self.diagnostic_data = {
            "x": data["in_domain"]["x"].clone(),
            "y": data["in_domain"]["y"].clone(),
            "target_col": 0,
            "sample_count": len(data["in_domain"]["x"]),
        }

        # Train baseline and apply corruption
        clean_model = ToyMLP()
        opt = torch.optim.Adam(clean_model.parameters(), lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                clean_model(data["train_full"]["x"]), data["train_full"]["y"]
            )
            loss.backward()
            opt.step()

        corrupted = copy.deepcopy(clean_model)
        # Apply corruption to Output 0
        defect_w = clean_model.fc2.weight.data.clone().numpy()
        defect_w[0, :4] = 0.0
        corrupted.fc2.weight.data.copy_(torch.from_numpy(defect_w))

        # Explicitly delete clean model and training data from proposer memory
        del clean_model
        del defect_w
        del data

        self.corrupted_model: nn.Module = corrupted
        self.corrupted_fingerprint: str = compute_model_fingerprint(corrupted)

        self.budget: PatchBudget = PatchBudget(
            max_changed_tensors=1,
            max_l2_norm=5.0,
            max_relative_norm=1.0,
            allowed_tensor_names=["fc2.weight"],
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_sealed") or "clean" in name:
            raise ProposerIsolationViolationError(
                f"Access denied: Proposer environment strictly isolates secrets. Attribute '{name}' does not exist."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_corrupted_model(self) -> nn.Module:
        """Return a disposable clone of the corrupted model."""
        return copy.deepcopy(self.corrupted_model)

    def get_architecture_metadata(self) -> dict[str, Any]:
        """Return parameter names, shapes, and layer information."""
        params = {}
        for name, param in self.corrupted_model.named_parameters():
            params[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "numel": param.numel(),
            }
        return {
            "architecture": "ToyMLP(4 -> 8 -> 2)",
            "base_fingerprint": self.corrupted_fingerprint,
            "parameters": params,
            "budget": {
                "max_changed_tensors": self.budget.max_changed_tensors,
                "max_l2_norm": self.budget.max_l2_norm,
                "allowed_tensor_names": self.budget.allowed_tensor_names,
            },
        }

    def get_diagnostic_failure_report(self) -> dict[str, Any]:
        """Compute empirical error on the diagnostic task without revealing clean weights."""
        with torch.no_grad():
            preds = self.corrupted_model(self.diagnostic_data["x"])
            target_preds = preds[:, :1]
            mse = torch.nn.functional.mse_loss(target_preds, self.diagnostic_data["y"]).item()

        return {
            "target_task": "Output column 0 (function approximation)",
            "control_task": "Output column 1 (untouched capability)",
            "observed_diagnostic_mse": float(mse),
            "diagnostic_sample_count": self.diagnostic_data["sample_count"],
            "inputs": self.diagnostic_data["x"].clone(),
            "expected_outputs": self.diagnostic_data["y"].clone(),
        }

    def submit_candidate_patch(self, patch: WeightPatch) -> dict[str, Any]:
        """Submit candidate patch to the isolated evaluator process.
        
        Enforces:
        - Query budget: at most 1 evaluation per candidate run.
        - Frozen serialization: patch is serialized to disk before calling evaluator.
        - Subprocess boundary: evaluator runs in a separate OS process.
        - Aggregated metrics only: returns macro gains, never per-sample predictions.
        """
        if self._queries_used >= self._max_queries:
            raise EvaluationBudgetExceededError(
                f"Evaluation query budget exceeded ({self._queries_used}/{self._max_queries}). "
                "Proposers are restricted to a single evaluation per run to prevent adaptive hill-climbing."
            )

        # Pre-submission sanity checks
        if patch.base_model_fingerprint != self.corrupted_fingerprint:
            raise ValueError(
                f"Candidate patch fingerprint mismatch: patch declared '{patch.base_model_fingerprint}' "
                f"but active baseline has '{self.corrupted_fingerprint}'."
            )

        # Serialize patch to a temp JSON file with unique UUID
        patch_dict = {
            "patch_id": patch.patch_id,
            "base_model_id": patch.base_model_id,
            "base_model_revision": patch.base_model_revision,
            "base_model_fingerprint": patch.base_model_fingerprint,
            "deltas": {k: v.tolist() for k, v in patch.deltas.items()},
            "hypothesis": patch.hypothesis,
            "proposed_by": patch.proposed_by,
            "rationale": getattr(patch, "rationale", ""),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            patch_file = Path(temp_dir) / f"candidate_{uuid.uuid4().hex}.json"
            patch_file.write_text(json.dumps(patch_dict), encoding="utf-8")

            # Call the isolated evaluator subprocess
            cmd = [
                sys.executable,
                "-m",
                "weightliftai.weight_programming.isolated_evaluator",
                "--seed",
                str(self.seed),
                "--patch-json",
                str(patch_file),
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if process.returncode != 0:
                try:
                    err_json = json.loads(process.stdout or process.stderr)
                    err_msg = err_json.get("error_message", process.stderr)
                except Exception:
                    err_msg = process.stderr or process.stdout
                raise RuntimeError(f"Isolated evaluator process failed: {err_msg}")

            eval_output = json.loads(process.stdout)

        self._queries_used += 1
        return eval_output
