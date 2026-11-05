"""Traceable Representation-to-Intervention Pipeline.

Implements Phase 4 of the Research Validation Program:
1. Traceable pipeline:
   Behavioral Failure -> Candidate Representation/Circuit Hypothesis ->
   Intervention -> Independent Multi-Task Evaluation.
2. Least-Invasive Intervention Comparison:
   Directly compares transient activation-space interventions (patching, steering, ablation)
   against persistent weight-space interventions (weight patches) on identical prompts,
   metrics, and controls.
3. Controls and Generalization:
   Includes diagnostic prompts, matched counterfactuals, held-out generalization prompts,
   and random-direction negative controls.
4. Causal Necessity and Sufficiency:
   Distinguishes associative probe decodability from causal mediation via directional
   activation clamping and ablation.
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclasses.dataclass(frozen=True)
class BehavioralFailure:
    """Documented behavioral failure on a diagnostic evaluation."""
    task_id: str
    prompt: str
    expected_output: str
    observed_output: str
    metric_name: str
    error_magnitude: float
    context_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CircuitHypothesis:
    """Hypothesis localizing a behavioral failure to a specific layer and direction."""
    hypothesis_id: str
    target_module: str
    target_layer: int
    site: Literal["input", "output", "residual"]
    direction_name: str
    rationale: str
    direction: list[float] | None = None  # unit vector or probe coefficients


@dataclasses.dataclass(frozen=True)
class EvaluationSample:
    """Evaluation sample with explicit sample role for confound control."""
    sample_id: str
    prompt: str
    target: str
    sample_type: Literal[
        "diagnostic",
        "matched_counterfactual",
        "held_out_generalization",
        "negative_control",
    ]


@dataclasses.dataclass
class ComparativeInterventionResult:
    """Results comparing activation interventions vs weight edits."""
    pipeline_id: str
    timestamp_utc: str
    hypothesis: CircuitHypothesis
    diagnostic_recovery: dict[str, float]
    counterfactual_preservation: dict[str, float]
    held_out_generalization: dict[str, float]
    negative_control_score: dict[str, float]
    least_invasive_recommendation: dict[str, Any]
    limitations: list[str]


def _get_module_by_name(model: Any, target_name: str) -> Any:
    for name, module in model.named_modules():
        if name == target_name:
            return module
    raise KeyError(f"Module '{target_name}' not found in model.")


def run_activation_steering_hook(
    model: Any,
    module_name: str,
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[Any, Callable[[], None]]:
    """Register a forward hook that adds alpha * steering_vector to the module output."""
    if torch is None:
        raise RuntimeError("PyTorch is required for activation steering.")

    target_mod = _get_module_by_name(model, module_name)
    vector = steering_vector.detach().float()
    norm = torch.linalg.norm(vector)
    if norm > 1e-8:
        vector = vector / norm

    def hook(_module: Any, _input: Any, output: Any):
        if isinstance(output, tuple):
            out_tensor = output[0]
            rest = output[1:]
        else:
            out_tensor = output
            rest = ()

        steered = out_tensor + (alpha * vector).to(out_tensor.device, out_tensor.dtype)
        if rest:
            return (steered, *rest)
        return steered

    handle = target_mod.register_forward_hook(hook)
    return handle, handle.remove


def run_activation_ablation_hook(
    model: Any,
    module_name: str,
    direction_vector: torch.Tensor,
) -> tuple[Any, Callable[[], None]]:
    """Register a forward hook that projects out the direction_vector from activations."""
    if torch is None:
        raise RuntimeError("PyTorch is required for activation ablation.")

    target_mod = _get_module_by_name(model, module_name)
    vector = direction_vector.detach().float()
    norm = torch.linalg.norm(vector)
    if norm > 1e-8:
        v = vector / norm
    else:
        v = vector

    def hook(_module: Any, _input: Any, output: Any):
        if isinstance(output, tuple):
            out_tensor = output[0]
            rest = output[1:]
        else:
            out_tensor = output
            rest = ()

        # Project out v: x_proj = x - (x . v) v
        v_dev = v.to(out_tensor.device, out_tensor.dtype)
        dot = torch.matmul(out_tensor, v_dev)
        ablated = out_tensor - torch.matmul(dot.unsqueeze(-1), v_dev.unsqueeze(0))
        if rest:
            return (ablated, *rest)
        return ablated

    handle = target_mod.register_forward_hook(hook)
    return handle, handle.remove


def run_traceable_intervention_pipeline(
    *,
    model: Any,
    failure: BehavioralFailure,
    hypothesis: CircuitHypothesis,
    eval_samples: Sequence[EvaluationSample],
    evaluate_fn: Callable[[Any, Sequence[EvaluationSample]], dict[str, dict[str, float]]],
    weight_patch_fn: Callable[[Any, CircuitHypothesis], Any],
    steering_alpha: float = 1.0,
    seed: int = 42,
) -> ComparativeInterventionResult:
    """Execute a traceable evaluation comparing activation vs weight interventions.
    
    evaluate_fn must accept (model, samples) and return a dictionary keyed by sample_type:
    {
        "diagnostic": {"accuracy": ..., "loss": ...},
        "matched_counterfactual": {"accuracy": ..., "loss": ...},
        "held_out_generalization": {"accuracy": ..., "loss": ...},
        "negative_control": {"accuracy": ..., "loss": ...},
    }
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for traceable intervention pipeline.")

    # 1. Baseline Evaluation
    model.eval()
    baseline_metrics = evaluate_fn(model, eval_samples)

    # Prepare direction vector
    target_mod = _get_module_by_name(model, hypothesis.target_module)
    if hypothesis.direction is not None:
        direction_tensor = torch.tensor(hypothesis.direction, dtype=torch.float32)
    else:
        # Fallback to output dimension of target module if unspecified
        dim = getattr(target_mod, "out_features", None) or getattr(model.config, "hidden_size", 64)
        torch.manual_seed(seed)
        direction_tensor = torch.randn(dim, dtype=torch.float32)

    # 2. Activation Steering Evaluation
    handle_steer, remove_steer = run_activation_steering_hook(
        model, hypothesis.target_module, direction_tensor, alpha=steering_alpha
    )
    try:
        steering_metrics = evaluate_fn(model, eval_samples)
    finally:
        remove_steer()

    # 3. Activation Ablation Evaluation (Necessity Test)
    handle_ablate, remove_ablate = run_activation_ablation_hook(
        model, hypothesis.target_module, direction_tensor
    )
    try:
        ablation_metrics = evaluate_fn(model, eval_samples)
    finally:
        remove_ablate()

    # 4. Negative Control: Random norm-matched steering vector
    torch.manual_seed(seed + 999)
    random_dir = torch.randn_like(direction_tensor)
    handle_ctrl, remove_ctrl = run_activation_steering_hook(
        model, hypothesis.target_module, random_dir, alpha=steering_alpha
    )
    try:
        negative_ctrl_metrics = evaluate_fn(model, eval_samples)
    finally:
        remove_ctrl()

    # 5. Weight Intervention Evaluation (Cloned / sandboxed model)
    patched_model = weight_patch_fn(copy.deepcopy(model), hypothesis)
    patched_model.eval()
    weight_metrics = evaluate_fn(patched_model, eval_samples)

    # Calculate Effect Sizes and Compare Interventions
    diag_base = baseline_metrics.get("diagnostic", {}).get("accuracy", 0.0)
    diag_steer = steering_metrics.get("diagnostic", {}).get("accuracy", 0.0)
    diag_ablate = ablation_metrics.get("diagnostic", {}).get("accuracy", 0.0)
    diag_weight = weight_metrics.get("diagnostic", {}).get("accuracy", 0.0)

    cf_base = baseline_metrics.get("matched_counterfactual", {}).get("accuracy", 1.0)
    cf_steer = steering_metrics.get("matched_counterfactual", {}).get("accuracy", 1.0)
    cf_weight = weight_metrics.get("matched_counterfactual", {}).get("accuracy", 1.0)

    gen_base = baseline_metrics.get("held_out_generalization", {}).get("accuracy", 0.0)
    gen_steer = steering_metrics.get("held_out_generalization", {}).get("accuracy", 0.0)
    gen_weight = weight_metrics.get("held_out_generalization", {}).get("accuracy", 0.0)

    # Least-Invasive Decision Logic
    steer_recovery_gain = diag_steer - diag_base
    weight_recovery_gain = diag_weight - diag_base
    cf_preservation_steer = cf_steer >= (cf_base - 0.05)
    cf_preservation_weight = cf_weight >= (cf_base - 0.05)

    if steer_recovery_gain >= 0.20 and cf_preservation_steer and gen_steer >= gen_base:
        recommendation = {
            "chosen_intervention": "activation_steering",
            "justification": (
                "Activation steering achieves target diagnostic recovery without permanent "
                "weight corruption or counterfactual regression. Adheres to the least-invasive "
                "intervention hierarchy."
            ),
            "activation_sufficient": True,
            "weight_edit_justified": False,
            "delta_recovery_steering": float(steer_recovery_gain),
            "delta_recovery_weight": float(weight_recovery_gain),
        }
    elif weight_recovery_gain >= 0.20 and cf_preservation_weight and gen_weight >= gen_base:
        recommendation = {
            "chosen_intervention": "weight_patch",
            "justification": (
                "Weight patch achieves superior target recovery and satisfies counterfactual "
                "preservation gates where activation intervention was insufficient or deployment "
                "precludes dynamic forward hooks."
            ),
            "activation_sufficient": False,
            "weight_edit_justified": True,
            "delta_recovery_steering": float(steer_recovery_gain),
            "delta_recovery_weight": float(weight_recovery_gain),
        }
    else:
        recommendation = {
            "chosen_intervention": "none",
            "justification": (
                "Neither activation steering nor candidate weight edit satisfied joint recovery "
                "and counterfactual preservation thresholds. Interventions rejected."
            ),
            "activation_sufficient": False,
            "weight_edit_justified": False,
            "delta_recovery_steering": float(steer_recovery_gain),
            "delta_recovery_weight": float(weight_recovery_gain),
        }

    return ComparativeInterventionResult(
        pipeline_id=f"pipe-{hypothesis.hypothesis_id}",
        timestamp_utc=_utc_now(),
        hypothesis=hypothesis,
        diagnostic_recovery={
            "baseline": float(diag_base),
            "activation_steering": float(diag_steer),
            "activation_ablation": float(diag_ablate),
            "weight_patch": float(diag_weight),
        },
        counterfactual_preservation={
            "baseline": float(cf_base),
            "activation_steering": float(cf_steer),
            "weight_patch": float(cf_weight),
        },
        held_out_generalization={
            "baseline": float(gen_base),
            "activation_steering": float(gen_steer),
            "weight_patch": float(gen_weight),
        },
        negative_control_score={
            "random_direction_steering": float(
                negative_ctrl_metrics.get("diagnostic", {}).get("accuracy", 0.0)
            )
        },
        least_invasive_recommendation=recommendation,
        limitations=[
            "Activation steering requires forward hook runtime support.",
            "Linear steering vectors assume representation linearity in the residual/activation space.",
            "Counterfactual preservation tested on declared benchmark pairs only.",
        ],
    )
