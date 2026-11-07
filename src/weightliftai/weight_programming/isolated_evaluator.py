"""Isolated Evaluator Service for Blind Weight-Repair Benchmark.

Executes in a separated operating system process boundary from the candidate proposer.
Privately holds clean weights, defect creation masks, and sealed test/OOD/regression splits.
Evaluates submitted candidate patches and outputs coarse aggregate metrics ONLY.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.inspector import compute_model_fingerprint, validate_patch_against_budget
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch
from weightliftai.weight_programming.toy_defect import (
    ToyMLP,
    generate_synthetic_data,
    mse_evaluator,
)


def run_isolated_evaluation(
    patch: WeightPatch,
    seed: int = 42,
    budget: PatchBudget | None = None,
) -> dict[str, Any]:
    """Execute evaluation inside the isolated evaluator process address space."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Generate full dataset splits privately
    data = generate_synthetic_data(seed=seed)
    train_full = data["train_full"]
    in_domain = data["in_domain"]
    held_out = data["held_out"]
    ood = data["distribution_shift"]
    regression = data["regression"]

    # 2. Train clean ground truth model in private memory
    clean_model = ToyMLP()
    opt = torch.optim.Adam(clean_model.parameters(), lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(
            clean_model(train_full["x"]), train_full["y"]
        )
        loss.backward()
        opt.step()

    clean_fc2 = clean_model.fc2.weight.data.clone().numpy()

    # 3. Create corrupted model
    corrupted_model = copy.deepcopy(clean_model)
    defect_w = clean_fc2.copy()
    defect_w[0, :4] = 0.0  # Zero out first 4 incoming connections to Output 0
    corrupted_model.fc2.weight.data.copy_(torch.from_numpy(defect_w))

    corrupted_fingerprint = compute_model_fingerprint(corrupted_model)

    # 4. Enforce base checkpoint identity match
    if patch.base_model_fingerprint != corrupted_fingerprint:
        raise ValueError(
            f"Base model fingerprint mismatch: patch declared '{patch.base_model_fingerprint}' "
            f"but corrupted baseline has '{corrupted_fingerprint}'."
        )

    # 5. Enforce budget
    effective_budget = budget or PatchBudget(
        max_changed_tensors=1,
        max_l2_norm=5.0,
        max_relative_norm=1.0,
        allowed_tensor_names=["fc2.weight"],
    )
    validate_patch_against_budget(corrupted_model, patch, effective_budget)

    # 6. Evaluate via IndependentEvaluator
    evaluator = IndependentEvaluator(
        eval_fn=mse_evaluator,
        target_in_domain=in_domain,
        target_held_out=held_out,
        distribution_shift=ood,
        regression_suite=regression,
        min_target_gain=0.10,
        max_regression_tolerance=0.02,
        higher_is_better=True,
    )

    eval_result = evaluator.evaluate_patch(corrupted_model, patch, budget=effective_budget)

    # 7. Return ONLY aggregate scalar metrics (no sample-level data or hidden predictions)
    return {
        "success": True,
        "patch_id": eval_result["patch_id"],
        "patch_sha256": eval_result["patch_sha256"],
        "passed_gates": eval_result["passed_gates"],
        "recommendation": eval_result["recommendation"],
        "gains": eval_result["gains"],
        "gate_thresholds": eval_result["gate_thresholds"],
        "aggregate_scores": {
            "held_out_mse": -float(eval_result["metrics"]["patched"]["held_out"]),
            "ood_mse": -float(eval_result["metrics"]["patched"]["distribution_shift"]),
            "regression_mse": -float(eval_result["metrics"]["patched"]["regression_suite"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Isolated Evaluator Service CLI")
    parser.add_argument("--seed", type=int, default=42, help="Benchmark random seed")
    parser.add_argument("--patch-json", type=str, required=True, help="Path to serialized WeightPatch JSON")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    try:
        patch_path = Path(args.patch_json)
        raw_patch_data = json.loads(patch_path.read_text(encoding="utf-8"))

        # Deserializing deltas from lists to numpy arrays
        deltas = {}
        for name, arr_list in raw_patch_data.get("deltas", {}).items():
            deltas[name] = np.array(arr_list, dtype=np.float32)

        patch = WeightPatch(
            patch_id=raw_patch_data["patch_id"],
            base_model_id=raw_patch_data["base_model_id"],
            base_model_revision=raw_patch_data["base_model_revision"],
            base_model_fingerprint=raw_patch_data.get("base_model_fingerprint"),
            deltas=deltas,
            hypothesis=raw_patch_data.get("hypothesis", ""),
            proposed_by=raw_patch_data.get("proposed_by", "unspecified"),
            rationale=raw_patch_data.get("rationale", ""),
        )

        result = run_isolated_evaluation(patch, seed=args.seed)
    except Exception as exc:
        result = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    out_str = json.dumps(result, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(out_str, encoding="utf-8")
    else:
        print(out_str)

    if not result.get("success", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
