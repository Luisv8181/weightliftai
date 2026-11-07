"""Defect families, private benchmark instance generation, and diagnostic conditions.

Implements:
- Prompt 2 (Private evaluator-controlled instance generation)
- Prompt 7 (Bounded weight/activation serialization for LLM inspection)
- Prompt 8 (Five controlled diagnostic-information conditions)
- Prompt 9 (Multiple representative defect families across seeds)
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.schemas import PatchBudget
from weightliftai.weight_programming.toy_defect import ToyMLP, generate_synthetic_data


class DiagnosticCondition(str, Enum):
    """Five controlled diagnostic information conditions (Prompt 8)."""

    BEHAVIORAL_ONLY = "behavioral_only"
    WEIGHT_SUMMARIES_ONLY = "weights_only"
    WEIGHTS_AND_FAILURES = "weights_and_failures"
    WEIGHTS_ACTIVATIONS_FAILURES = "weights_activations_failures"
    FULL_DIAGNOSTIC_TOOLKIT = "full_diagnostic_toolkit"


class DefectFamily(str, Enum):
    """Representative defect families for ToyMLP(4 -> 8 -> 2) (Prompt 9)."""

    FC2_ROW0_ZERO_MASK = "fc2_row0_zero_mask"
    FC2_ROW0_SIGN_INVERSION = "fc2_row0_sign_inversion"
    FC2_ROW0_SCALE_ATTENUATION = "fc2_row0_scale_attenuation"
    FC1_HIDDEN_CORRUPT_CHANNEL = "fc1_hidden_corrupt_channel"


DEFECT_FAMILY_SPECS: dict[str, dict[str, Any]] = {
    DefectFamily.FC2_ROW0_ZERO_MASK.value: {
        "description": "Zeroes out the first 4 incoming hidden connections to Output 0 (fc2.weight[0, :4] = 0).",
        "allowed_tensor_names": ["fc2.weight"],
        "max_changed_tensors": 1,
        "max_l2_norm": 5.0,
        "max_relative_norm": 1.5,
    },
    DefectFamily.FC2_ROW0_SIGN_INVERSION.value: {
        "description": "Inverts the sign of the first 4 incoming hidden connections to Output 0 (fc2.weight[0, :4] *= -1).",
        "allowed_tensor_names": ["fc2.weight"],
        "max_changed_tensors": 1,
        "max_l2_norm": 6.0,
        "max_relative_norm": 2.0,
    },
    DefectFamily.FC2_ROW0_SCALE_ATTENUATION.value: {
        "description": "Attenuates all incoming connections to Output 0 by 85% (fc2.weight[0, :] *= 0.15).",
        "allowed_tensor_names": ["fc2.weight"],
        "max_changed_tensors": 1,
        "max_l2_norm": 5.0,
        "max_relative_norm": 5.0,
    },
    DefectFamily.FC1_HIDDEN_CORRUPT_CHANNEL.value: {
        "description": "Zeroes out the last 4 incoming connections to Output 0 (fc2.weight[0, 4:] = 0).",
        "allowed_tensor_names": ["fc2.weight"],
        "max_changed_tensors": 1,
        "max_l2_norm": 5.0,
        "max_relative_norm": 1.5,
    },
}


def serialize_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Serialize a PyTorch model's state_dict into JSON-safe nested float lists."""
    out: dict[str, Any] = {}
    for k, v in model.state_dict().items():
        out[k] = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "values": v.detach().cpu().float().numpy().tolist(),
        }
    return out


def deserialize_model_state_dict(serialized: dict[str, Any]) -> ToyMLP:
    """Reconstruct a ToyMLP from a JSON-serialized state_dict."""
    model = ToyMLP()
    state = {}
    for k, info in serialized.items():
        arr = np.array(info["values"], dtype=np.float32)
        state[k] = torch.from_numpy(arr)
    model.load_state_dict(state)
    return model


def build_bounded_weight_summary(model: nn.Module) -> dict[str, Any]:
    """Construct a bounded, LLM-inspectable serialization of model weights (Prompt 7)."""
    summaries: dict[str, Any] = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().double().numpy()
        row_norms = np.round(np.linalg.norm(arr, axis=-1), 4).tolist() if arr.ndim >= 2 else []
        col_norms = np.round(np.linalg.norm(arr, axis=0), 4).tolist() if arr.ndim >= 2 else []
        zero_positions = np.argwhere(np.abs(arr) < 1e-6).tolist()
        summaries[name] = {
            "shape": list(arr.shape),
            "dtype": str(param.dtype),
            "numel": int(arr.size),
            "l2_norm": round(float(np.linalg.norm(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "row_l2_norms": row_norms,
            "col_l2_norms": col_norms,
            "near_zero_indices": zero_positions,
            "rounded_slice": np.round(arr, 4).tolist(),
        }
    return summaries


def build_activation_summary(model: ToyMLP, x_diag: torch.Tensor, y_diag: torch.Tensor) -> dict[str, Any]:
    """Compute permitted hidden-layer activation statistics on diagnostic inputs only."""
    with torch.no_grad():
        pre_act = model.fc1(x_diag)
        hidden = model.relu(pre_act)
        preds = model.fc2(hidden)
        residual_col0 = (y_diag[:, 0] - preds[:, 0]).cpu().double().numpy()
        h_np = hidden.cpu().double().numpy()  # (N, 8)

        unit_mean = np.round(np.mean(h_np, axis=0), 4).tolist()
        unit_std = np.round(np.std(h_np, axis=0), 4).tolist()
        active_rate = np.round(np.mean(h_np > 0.0, axis=0), 4).tolist()

        # Correlation / covariance between hidden unit activations and Output 0 residual error
        cov_with_residual: list[float] = []
        least_squares_delta_hint: list[float] = []
        # Solve regularised least-squares on diagnostic hidden activations for Output 0 residual
        # (strictly using diagnostic inputs & outputs only, never clean weights!)
        reg = 1e-3 * np.eye(h_np.shape[1], dtype=np.float64)
        ls_vec = np.linalg.solve(h_np.T @ h_np + reg, h_np.T @ residual_col0)
        for j in range(h_np.shape[1]):
            cov = float(np.mean(h_np[:, j] * residual_col0))
            cov_with_residual.append(round(cov, 4))
            least_squares_delta_hint.append(round(float(ls_vec[j]), 4))

    return {
        "layer": "relu(fc1(x))",
        "hidden_dim": int(h_np.shape[1]),
        "sample_count": int(h_np.shape[0]),
        "unit_mean": unit_mean,
        "unit_std": unit_std,
        "unit_active_fraction": active_rate,
        "hidden_to_target_residual_covariance": cov_with_residual,
        "diagnostic_hidden_residual_projection": least_squares_delta_hint,
    }


def build_full_toolkit_sensitivity(
    model: ToyMLP,
    x_diag: torch.Tensor,
    y_diag: torch.Tensor,
) -> dict[str, Any]:
    """Compute first-order diagnostic sensitivity of fc2.weight on diagnostic data only."""
    temp = copy.deepcopy(model)
    temp.zero_grad()
    preds = temp(x_diag)
    loss = torch.nn.functional.mse_loss(preds[:, :1], y_diag)
    loss.backward()
    grad = temp.fc2.weight.grad.detach().cpu().double().numpy() if temp.fc2.weight.grad is not None else np.zeros((2, 8), dtype=np.float64)
    return {
        "loss_gradient_fc2_weight": np.round(grad, 4).tolist(),
        "negative_gradient_direction_row0": np.round(-grad[0, :], 4).tolist(),
        "grad_row_l2_norms": np.round(np.linalg.norm(grad, axis=1), 4).tolist(),
    }


def generate_private_benchmark_instance(
    private_seed: int,
    defect_family: str = DefectFamily.FC2_ROW0_ZERO_MASK.value,
    diagnostic_condition: str = DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Privately generate a benchmark instance and construct the scoped proposer payload.

    Returns:
        (private_vault_record, proposer_authorized_bundle)
    """
    if defect_family not in DEFECT_FAMILY_SPECS:
        raise ValueError(f"Unsupported defect_family '{defect_family}'. Valid: {sorted(DEFECT_FAMILY_SPECS)}")
    valid_conditions = {c.value for c in DiagnosticCondition}
    if diagnostic_condition not in valid_conditions:
        raise ValueError(
            f"Unsupported diagnostic_condition '{diagnostic_condition}'. Valid: {sorted(valid_conditions)}"
        )

    # Save & restore RNG states so evaluator randomness does not leak into caller RNG state
    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    try:
        effective_seed = int(private_seed) % (2**31 - 1)
        torch.manual_seed(effective_seed)
        np.random.seed(effective_seed)

        data = generate_synthetic_data(seed=effective_seed)

        clean_model = ToyMLP()
        opt = torch.optim.Adam(clean_model.parameters(), lr=0.05)
        for _ in range(250):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                clean_model(data["train_full"]["x"]),
                data["train_full"]["y"],
            )
            loss.backward()
            opt.step()

        clean_fc2 = clean_model.fc2.weight.data.clone().numpy()
        corrupted_model = copy.deepcopy(clean_model)
        defect_w = clean_fc2.copy()
        defect_mask = np.zeros_like(defect_w, dtype=np.float32)

        if defect_family == DefectFamily.FC2_ROW0_ZERO_MASK.value:
            defect_w[0, :4] = 0.0
            defect_mask[0, :4] = 1.0
        elif defect_family == DefectFamily.FC2_ROW0_SIGN_INVERSION.value:
            defect_w[0, :4] = -clean_fc2[0, :4]
            defect_mask[0, :4] = 1.0
        elif defect_family == DefectFamily.FC2_ROW0_SCALE_ATTENUATION.value:
            defect_w[0, :] = 0.15 * clean_fc2[0, :]
            defect_mask[0, :] = 1.0
        elif defect_family == DefectFamily.FC1_HIDDEN_CORRUPT_CHANNEL.value:
            defect_w[0, 4:] = 0.0
            defect_mask[0, 4:] = 1.0

        corrupted_model.fc2.weight.data.copy_(torch.from_numpy(defect_w))
        corrupted_fingerprint = compute_model_fingerprint(corrupted_model)

        spec = DEFECT_FAMILY_SPECS[defect_family]
        budget = PatchBudget(
            max_changed_tensors=spec["max_changed_tensors"],
            max_l2_norm=spec["max_l2_norm"],
            max_relative_norm=spec["max_relative_norm"],
            allowed_tensor_names=list(spec["allowed_tensor_names"]),
        )

        # Compute diagnostic behavioral observations on in_domain only
        x_diag = data["in_domain"]["x"]
        y_diag = data["in_domain"]["y"]
        with torch.no_grad():
            preds = corrupted_model(x_diag)
            target_preds = preds[:, :1]
            diag_mse = float(torch.nn.functional.mse_loss(target_preds, y_diag).item())

        # Construct scoped diagnostic payload strictly according to diagnostic_condition
        include_behavior = diagnostic_condition in {
            DiagnosticCondition.BEHAVIORAL_ONLY.value,
            DiagnosticCondition.WEIGHTS_AND_FAILURES.value,
            DiagnosticCondition.WEIGHTS_ACTIVATIONS_FAILURES.value,
            DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        }
        include_weights = diagnostic_condition in {
            DiagnosticCondition.WEIGHT_SUMMARIES_ONLY.value,
            DiagnosticCondition.WEIGHTS_AND_FAILURES.value,
            DiagnosticCondition.WEIGHTS_ACTIVATIONS_FAILURES.value,
            DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        }
        include_activations = diagnostic_condition in {
            DiagnosticCondition.WEIGHTS_ACTIVATIONS_FAILURES.value,
            DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        }
        include_toolkit = diagnostic_condition == DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value

        param_shapes = {
            name: {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "numel": int(p.numel()),
            }
            for name, p in corrupted_model.named_parameters()
        }

        scoped_diagnostic: dict[str, Any] = {
            "diagnostic_condition": diagnostic_condition,
            "target_task": "Output column 0 (function approximation)",
            "control_task": "Output column 1 (untouched capability)",
            "parameter_shapes": param_shapes,
        }

        if include_behavior:
            scoped_diagnostic["behavioral_failures"] = {
                "observed_diagnostic_mse": round(diag_mse, 6),
                "diagnostic_sample_count": int(len(x_diag)),
                "inputs": x_diag.detach().cpu().float().numpy().tolist(),
                "expected_outputs": y_diag.detach().cpu().float().numpy().tolist(),
                "corrupted_predictions": target_preds.detach().cpu().float().numpy().tolist(),
                "target_col": 0,
            }

        if include_weights:
            scoped_diagnostic["weight_summaries"] = build_bounded_weight_summary(corrupted_model)
            scoped_diagnostic["corrupted_state_dict"] = serialize_model_state_dict(corrupted_model)

        if include_activations:
            scoped_diagnostic["activation_summaries"] = build_activation_summary(
                corrupted_model, x_diag, y_diag
            )

        if include_toolkit:
            scoped_diagnostic["toolkit_sensitivity"] = build_full_toolkit_sensitivity(
                corrupted_model, x_diag, y_diag
            )

        private_vault_record = {
            "private_seed": effective_seed,
            "defect_family": defect_family,
            "diagnostic_condition": diagnostic_condition,
            "corrupted_fingerprint": corrupted_fingerprint,
            "clean_state_dict": serialize_model_state_dict(clean_model),
            "corrupted_state_dict": serialize_model_state_dict(corrupted_model),
            "defect_mask": defect_mask.tolist(),
            "oracle_delta_fc2": (clean_fc2 - defect_w).tolist(),
            "budget": {
                "max_changed_tensors": budget.max_changed_tensors,
                "max_l2_norm": budget.max_l2_norm,
                "max_relative_norm": budget.max_relative_norm,
                "allowed_tensor_names": list(budget.allowed_tensor_names or []),
            },
            "splits": {
                k: {
                    "x": v["x"].detach().cpu().float().numpy().tolist(),
                    "y": v["y"].detach().cpu().float().numpy().tolist(),
                    "target_col": v.get("target_col"),
                }
                for k, v in data.items()
            },
        }

        proposer_bundle = {
            "architecture": "ToyMLP(4 -> 8 -> 2)",
            "base_fingerprint": corrupted_fingerprint,
            "diagnostic_condition": diagnostic_condition,
            "parameters": param_shapes,
            "budget": {
                "max_changed_tensors": budget.max_changed_tensors,
                "max_l2_norm": budget.max_l2_norm,
                "max_relative_norm": budget.max_relative_norm,
                "allowed_tensor_names": list(budget.allowed_tensor_names or []),
            },
            "diagnostic_payload": scoped_diagnostic,
        }

        return private_vault_record, proposer_bundle
    finally:
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
