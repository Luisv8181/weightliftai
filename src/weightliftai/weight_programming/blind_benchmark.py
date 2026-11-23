"""Legacy Same-Process Baseline Harness & Sealed Blind Benchmark Runner.

NOTE ON ISOLATION BOUNDARY (Prompt 6 / WSP-001A):
`BlindRepairBenchmark` is a same-process reference/baseline harness (`IS_SEALED_BLIND_BOUNDARY = False`)
used to compute reference baselines (no-op, random norm-matched, gradient recovery, and oracle upper bound)
within a single process for unit and CLI baseline checks. Because Python closures and attributes in the
same OS process can be inspected via `gc` or reflection, `BlindRepairBenchmark` MUST NOT be treated as a
sealed security boundary for untrusted proposers.

For sealed, evaluator-authoritative blind evaluation across an OS process boundary with private
evaluator-controlled randomness, persistent SQLite query-budget enforcement, and Ed25519 signed
receipts, use `BlindProposerEnvironment` (`weightliftai.weight_programming.proposer_api`) or
`run_sealed_blind_evaluation()`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.proposer_api import BlindProposerEnvironment
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch
from weightliftai.weight_programming.toy_defect import ToyMLP, generate_synthetic_data, mse_evaluator


class BlindRepairBenchmark:
    """Legacy same-process baseline harness (non-sealed; use BlindProposerEnvironment for sealed runs)."""

    IS_SEALED_BLIND_BOUNDARY: bool = False
    BOUNDARY_CLASSIFICATION: str = "LEGACY_SAME_PROCESS_BASELINE_HARNESS"

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

        # 1. Generate full dataset splits
        data = generate_synthetic_data(seed=seed)
        self.train_full = data["train_full"]
        self.diagnostic = data["in_domain"]
        self._sealed_held_out = data["held_out"]
        self._sealed_ood = data["distribution_shift"]
        self._sealed_regression = data["regression"]

        # 2. Train clean ground truth model
        clean_model = ToyMLP()
        opt = torch.optim.Adam(clean_model.parameters(), lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                clean_model(self.train_full["x"]), self.train_full["y"]
            )
            loss.backward()
            opt.step()

        # Seal clean weights in attribute (same-process baseline only; not a hard process boundary)
        self._sealed_clean_fc2 = clean_model.fc2.weight.data.clone().numpy()

        # 3. Create corrupted model with defect in Output 0 (Target Task)
        corrupted_model = copy.deepcopy(clean_model)
        defect_w = self._sealed_clean_fc2.copy()
        defect_w[0, :4] = 0.0  # Zero out first 4 incoming connections to Output 0
        corrupted_model.fc2.weight.data.copy_(torch.from_numpy(defect_w))

        self.corrupted_model = corrupted_model
        self.corrupted_fingerprint = compute_model_fingerprint(corrupted_model)

        self.budget = PatchBudget(
            max_changed_tensors=1,
            max_l2_norm=5.0,
            max_relative_norm=1.0,
            allowed_tensor_names=["fc2.weight"],
        )

        self.evaluator = IndependentEvaluator(
            eval_fn=mse_evaluator,
            target_in_domain=self.diagnostic,
            target_held_out=self._sealed_held_out,
            distribution_shift=self._sealed_ood,
            regression_suite=self._sealed_regression,
            min_target_gain=0.10,
            max_regression_tolerance=0.02,
            higher_is_better=True,
        )

    def get_corrupted_model(self) -> nn.Module:
        """Provide a cloned copy of the corrupted model to the repair agent."""
        return copy.deepcopy(self.corrupted_model)

    def get_failure_report(self) -> dict[str, Any]:
        """Provide empirical failure diagnostic examples without revealing clean weights or defect mask."""
        with torch.no_grad():
            preds = self.corrupted_model(self.diagnostic["x"])
            target_preds = preds[:, :1]
            mse = torch.nn.functional.mse_loss(target_preds, self.diagnostic["y"]).item()

        return {
            "boundary_classification": self.BOUNDARY_CLASSIFICATION,
            "is_sealed_blind_boundary": self.IS_SEALED_BLIND_BOUNDARY,
            "model_architecture": "ToyMLP(4 -> 8 -> 2)",
            "base_fingerprint": self.corrupted_fingerprint,
            "target_task": "Output column 0 (function approximation)",
            "control_task": "Output column 1 (untouched capability)",
            "diagnostic_sample_count": len(self.diagnostic["x"]),
            "observed_diagnostic_mse": float(mse),
            "diagnostic_inputs": self.diagnostic["x"].clone(),
            "expected_outputs": self.diagnostic["y"].clone(),
            "target_column": 0,
        }

    def compute_no_op_baseline(self) -> WeightPatch:
        """Baseline 1: Zero delta (no change)."""
        delta = np.zeros_like(self.corrupted_model.fc2.weight.data.numpy())
        return WeightPatch(
            patch_id="baseline-no-op",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=self.corrupted_fingerprint,
            deltas={"fc2.weight": delta},
            hypothesis="No intervention baseline.",
            proposed_by="benchmark_null_baseline",
        )

    def compute_random_norm_matched_baseline(self, target_l2: float = 1.0) -> WeightPatch:
        """Baseline 2: Random Gaussian delta scaled to match target L2 norm."""
        np.random.seed(self.seed)
        shape = self.corrupted_model.fc2.weight.data.shape
        raw_r = np.random.randn(*shape).astype(np.float32)
        norm_r = float(np.linalg.norm(raw_r))
        scaled_delta = (raw_r / max(norm_r, 1e-8)) * target_l2
        scaled_delta[1, :] = 0.0

        return WeightPatch(
            patch_id="baseline-random-norm-matched",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=self.corrupted_fingerprint,
            deltas={"fc2.weight": scaled_delta},
            hypothesis="Random norm-matched perturbation baseline.",
            proposed_by="benchmark_random_baseline",
        )

    def compute_gradient_recovery_baseline(self, steps: int = 100, lr: float = 0.05) -> WeightPatch:
        """Baseline 3: Conventional gradient-based recovery using diagnostic failure data only."""
        temp_model = copy.deepcopy(self.corrupted_model)
        temp_model.fc1.weight.requires_grad = False
        temp_model.fc1.bias.requires_grad = False
        temp_model.fc2.bias.requires_grad = False
        temp_model.fc2.weight.requires_grad = True

        opt = torch.optim.Adam([temp_model.fc2.weight], lr=lr)
        init_w = temp_model.fc2.weight.data.clone()

        for _ in range(steps):
            opt.zero_grad()
            preds = temp_model(self.diagnostic["x"])
            loss = torch.nn.functional.mse_loss(preds[:, :1], self.diagnostic["y"])
            loss.backward()
            if temp_model.fc2.weight.grad is not None:
                temp_model.fc2.weight.grad[1, :] = 0.0
            opt.step()

        delta_arr = (temp_model.fc2.weight.data - init_w).cpu().numpy()

        return WeightPatch(
            patch_id="baseline-empirical-gradient",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=self.corrupted_fingerprint,
            deltas={"fc2.weight": delta_arr},
            hypothesis="Empirical gradient-based repair from diagnostic failure samples alone.",
            proposed_by="empirical_gradient_optimizer",
        )

    def compute_oracle_upper_bound(self) -> WeightPatch:
        """Baseline 5: Oracle upper bound computed directly from sealed clean weights."""
        curr_w = self.corrupted_model.fc2.weight.data.numpy()
        oracle_delta = self._sealed_clean_fc2 - curr_w

        return WeightPatch(
            patch_id="reference-oracle-upper-bound",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=self.corrupted_fingerprint,
            deltas={"fc2.weight": oracle_delta},
            hypothesis="Oracle upper-bound repair computed directly from ground-truth clean weights.",
            proposed_by="sealed_oracle_reference",
        )

    def run_full_benchmark(self, candidate_patch: WeightPatch | None = None) -> dict[str, Any]:
        """Run all standard baselines and optional candidate patch (same-process baseline harness)."""
        results = {}

        no_op = self.compute_no_op_baseline()
        results["no_op"] = self.evaluator.evaluate_patch(self.corrupted_model, no_op, budget=self.budget)

        grad_patch = self.compute_gradient_recovery_baseline()
        results["gradient_baseline"] = self.evaluator.evaluate_patch(self.corrupted_model, grad_patch, budget=self.budget)

        target_norm = float(np.linalg.norm(grad_patch.deltas["fc2.weight"]))
        rand_patch = self.compute_random_norm_matched_baseline(target_l2=target_norm)
        results["random_norm_matched"] = self.evaluator.evaluate_patch(self.corrupted_model, rand_patch, budget=self.budget)

        if candidate_patch is not None:
            results["candidate_patch"] = self.evaluator.evaluate_patch(self.corrupted_model, candidate_patch, budget=self.budget)

        oracle = self.compute_oracle_upper_bound()
        results["oracle_upper_bound"] = self.evaluator.evaluate_patch(self.corrupted_model, oracle, budget=self.budget)

        return {
            "boundary_classification": self.BOUNDARY_CLASSIFICATION,
            "is_sealed_blind_boundary": self.IS_SEALED_BLIND_BOUNDARY,
            "benchmark_seed": self.seed,
            "base_fingerprint": self.corrupted_fingerprint,
            "baselines": results,
            "summary": {
                "no_op_promotable": results["no_op"]["passed_gates"],
                "random_norm_promotable": results["random_norm_matched"]["passed_gates"],
                "gradient_promotable": results["gradient_baseline"]["passed_gates"],
                "oracle_promotable": results["oracle_upper_bound"]["passed_gates"],
                "candidate_promotable": results["candidate_patch"]["passed_gates"] if candidate_patch else None,
            },
        }


def run_sealed_blind_evaluation(
    proposer_fn: Callable[[BlindProposerEnvironment], WeightPatch],
    defect_family: str = "fc2_row0_zero_mask",
    diagnostic_condition: str = "full_diagnostic_toolkit",
    ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute a truly blind, process-isolated, evaluator-authoritative repair evaluation."""
    env = BlindProposerEnvironment(
        defect_family=defect_family,
        diagnostic_condition=diagnostic_condition,
        ledger_path=ledger_path,
    )
    patch = proposer_fn(env)
    return env.submit_candidate_patch(patch)
