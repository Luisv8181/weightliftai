"""Jacobian Lens and J-Space Output-Influence Analysis.

Status: Implemented & Tested Prototype with Multi-Context / Cross-Position Aggregation
and 4-Dimension Evidence Separation (see docs/RESEARCH_VALIDATION_PROTOCOL.md & Prompts 28–31).

Implements the mathematical formulation from Gurnee et al. (2026) and Ostrow et al. (2026):
  J_l = E_{context c, source t, target t' >= t} [ d h_c[L, t'] / d h_c[l, t] ]
and strictly separates four distinct evidence dimensions (Prompt 31):
  1. Cosine Alignment (directional similarity vs. random/shuffled negative controls)
  2. Sensitivity (derivative magnitude & response to candidate/input variation)
  3. Stability (persistence across contexts, source/target positions, seeds, and layers)
  4. Intervention Effects (controlled causal activation interventions vs. norm-matched sham/random controls)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn

from weightliftai.cartography.layer_atlas import find_decoder_layers


JACOBIAN_LENS_METHOD_MAPPING: dict[str, dict[str, str]] = {
    "cross_position_jacobian_dh_L_tprime_dh_l_t": {
        "requirement": "Evaluate d h[L, t'] / d h[l, t] for distinct source position t and target/future position t' >= t.",
        "implementation": "compute_vjp(..., source_position=t, target_position=t_prime, future_offsets=..., enforce_causal_horizon=True)",
        "verification": "check_finite_differences(..., source_position=t, target_position=t_prime)",
    },
    "multi_context_and_position_expectation": {
        "requirement": "Average Jacobian directions across multiple contexts c, source positions t, and future offsets t' - t >= 0.",
        "implementation": "aggregate_jacobian_corpus(..., source_positions=..., future_offsets=..., aggregation_mode=..., normalization_mode=...)",
        "verification": "test_multi_context_and_future_position_aggregation_distinguishes_single_prompt",
    },
    "explicit_normalization_and_aggregation": {
        "requirement": "Support explicit vector normalization ('none', 'l2', 'rms') and robust context aggregation ('mean', 'median', 'trimmed_mean').",
        "implementation": "_normalize_vector(...) and _aggregate_direction_stack(...) in jacobian_lens.py",
        "verification": "test_jacobian_normalization_and_aggregation_modes",
    },
    "four_dimension_evidence_separation": {
        "requirement": "Report Cosine Alignment, Sensitivity, Stability, and Intervention Effects separately with negative controls.",
        "implementation": "evaluate_four_dimension_jacobian_evidence(...)",
        "verification": "test_four_dimension_jacobian_evidence_separation",
    },
}


def get_unembedding_weight(model: nn.Module) -> torch.Tensor:
    """Extract the unembedding (lm_head) weight matrix."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head.weight
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    raise AttributeError("Could not locate unembedding / lm_head weight matrix on model.")


def _normalize_vector(
    vec: torch.Tensor,
    mode: Literal["none", "l2", "rms"] = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a 1D tensor according to the specified mode."""
    if mode == "none":
        return vec
    if mode == "l2":
        norm = torch.norm(vec.float()).clamp_min(eps)
        return (vec.float() / norm).to(vec.dtype)
    if mode == "rms":
        rms = torch.sqrt(torch.mean(vec.float() ** 2)).clamp_min(eps)
        return (vec.float() / rms).to(vec.dtype)
    raise ValueError(f"Unsupported normalization mode '{mode}'. Choose 'none', 'l2', or 'rms'.")


def _aggregate_direction_stack(
    stack: np.ndarray,
    mode: Literal["mean", "median", "trimmed_mean"] = "mean",
    trim_fraction: float = 0.1,
) -> np.ndarray:
    """Aggregate a [N, dim] stack of Jacobian directions across contexts and positions."""
    if stack.ndim != 2 or len(stack) == 0:
        raise ValueError("Direction stack must have shape [N, dim] with N >= 1.")
    if mode == "mean":
        return np.mean(stack, axis=0)
    if mode == "median":
        return np.median(stack, axis=0)
    if mode == "trimmed_mean":
        n = len(stack)
        k = int(np.floor(n * trim_fraction))
        if 2 * k >= n or k == 0:
            return np.mean(stack, axis=0)
        sorted_stack = np.sort(stack, axis=0)
        return np.mean(sorted_stack[k : n - k], axis=0)
    raise ValueError(f"Unsupported aggregation_mode '{mode}'. Choose 'mean', 'median', or 'trimmed_mean'.")


def _resolve_positions(
    seq_len: int,
    position: int,
    source_position: int | None,
    target_position: int | None,
    future_offsets: Sequence[int] | None,
    enforce_causal_horizon: bool,
) -> tuple[int, list[int]]:
    """Resolve source token index t and target token indices t' >= t."""
    raw_src = source_position if source_position is not None else position
    src_idx = raw_src if raw_src >= 0 else seq_len + raw_src
    if src_idx < 0 or src_idx >= seq_len:
        raise ValueError(f"Resolved source_position {src_idx} is out of bounds for seq_len={seq_len}.")

    if future_offsets is not None:
        if len(future_offsets) == 0:
            raise ValueError("future_offsets cannot be empty when provided.")
        tgt_indices: list[int] = []
        for offset in future_offsets:
            if enforce_causal_horizon and offset < 0:
                raise ValueError(f"Negative future offset {offset} violates causal horizon (t' >= t).")
            candidate_t = src_idx + int(offset)
            if 0 <= candidate_t < seq_len:
                tgt_indices.append(candidate_t)
        if not tgt_indices:
            tgt_indices = [seq_len - 1]
    else:
        raw_tgt = target_position if target_position is not None else (
            source_position if source_position is not None and position == -1 and target_position is None else position
        )
        # Note: if caller explicitly passed source_position=t and left target_position=None,
        # use target_position=position (default -1 -> last token) if position != -1 or source_position
        if source_position is not None and target_position is None:
            tgt_idx = src_idx if position == -1 else (position if position >= 0 else seq_len + position)
        else:
            tgt_idx = raw_tgt if raw_tgt >= 0 else seq_len + raw_tgt

        if tgt_idx < 0 or tgt_idx >= seq_len:
            raise ValueError(f"Resolved target_position {tgt_idx} is out of bounds for seq_len={seq_len}.")
        if enforce_causal_horizon and tgt_idx < src_idx:
            raise ValueError(
                f"Causal horizon violation: target_position ({tgt_idx}) < source_position ({src_idx}). "
                "Published Jacobian Lens requires t' >= t."
            )
        tgt_indices = [tgt_idx]

    return src_idx, tgt_indices


def compute_vjp(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    target_vector: torch.Tensor,
    *,
    position: int = -1,
    source_position: int | None = None,
    target_position: int | None = None,
    future_offsets: Sequence[int] | None = None,
    enforce_causal_horizon: bool = True,
    normalize_vjp: Literal["none", "l2", "rms"] = "none",
    site: Literal["residual", "attention", "mlp"] = "residual",
    target_site: Literal["final_residual", "logits"] = "final_residual",
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the cross-position Vector-Jacobian Product v^T (d h[L, t'] / d h[l, t]).

    Supports:
    - Distinct `source_position` (t) and `target_position` (t' >= t).
    - Multi-step `future_offsets` (e.g., [0, 1, 2]) averaging sensitivity over future tokens t' = t + delta.
    - Explicit post-VJP normalization ('none', 'l2', 'rms').
    """
    decoder_layers = find_decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(decoder_layers):
        raise ValueError(f"layer_idx {layer_idx} out of range (0 to {len(decoder_layers) - 1})")

    source_layer = decoder_layers[layer_idx]
    if site == "residual":
        target_mod = source_layer
    elif site == "attention":
        target_mod = getattr(source_layer, "self_attn", getattr(source_layer, "attention", None))
        if target_mod is None:
            raise ValueError(f"Attention module not found in layer {layer_idx}")
    elif site == "mlp":
        target_mod = getattr(source_layer, "mlp", getattr(source_layer, "feed_forward", None))
        if target_mod is None:
            raise ValueError(f"MLP module not found in layer {layer_idx}")
    else:
        raise ValueError(f"Unknown site {site}")

    intermediate_act: list[torch.Tensor] = []

    def hook_fn(_m: nn.Module, _inputs: Any, output: Any) -> Any:
        act = output[0] if isinstance(output, tuple) else output
        act = act.clone()
        act.requires_grad_(True)
        act.retain_grad()
        intermediate_act.append(act)
        if isinstance(output, tuple):
            return (act,) + output[1:]
        return act

    hook_handle = target_mod.register_forward_hook(hook_fn)

    try:
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        with torch.enable_grad():
            outputs = model(input_ids, **kwargs)
            if target_site == "logits":
                out_tensor = outputs.logits
            else:
                if hasattr(outputs, "last_hidden_state"):
                    out_tensor = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    out_tensor = outputs.hidden_states[-1]
                else:
                    out_tensor = outputs.logits

            if len(intermediate_act) == 0:
                raise RuntimeError("Failed to capture intermediate activation during hook forward pass.")

            act = intermediate_act[0]
            seq_len = out_tensor.shape[1]
            src_pos_idx, tgt_pos_indices = _resolve_positions(
                seq_len=seq_len,
                position=position,
                source_position=source_position,
                target_position=target_position,
                future_offsets=future_offsets,
                enforce_causal_horizon=enforce_causal_horizon,
            )

            # Average target slices across selected future positions t' in tgt_pos_indices
            target_slices = out_tensor[0, tgt_pos_indices]  # [n_targets, dim]
            mean_target_slice = target_slices.mean(dim=0)  # [dim]

            tv = target_vector.to(mean_target_slice.device, dtype=mean_target_slice.dtype)
            if tv.ndim == 1:
                scalar_loss = torch.dot(tv, mean_target_slice)
            else:
                scalar_loss = (tv * mean_target_slice).sum()

            grads = torch.autograd.grad(scalar_loss, act, retain_graph=False, create_graph=False)
            grad_tensor = grads[0]
            vjp = grad_tensor[0, src_pos_idx].detach()
            return _normalize_vector(vjp, mode=normalize_vjp)
    finally:
        hook_handle.remove()


def check_finite_differences(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    target_vector: torch.Tensor,
    *,
    position: int = -1,
    source_position: int | None = None,
    target_position: int | None = None,
    future_offsets: Sequence[int] | None = None,
    site: Literal["residual", "attention", "mlp"] = "residual",
    target_site: Literal["final_residual", "logits"] = "final_residual",
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-4,
    dims_to_check: int = 5,
) -> dict[str, Any]:
    """Verify analytical autograd VJP against central finite differences across source t and target t' >= t."""
    import copy

    try:
        eval_model = copy.deepcopy(model).double()
        eval_target = target_vector.double()
    except Exception:
        eval_model = model
        eval_target = target_vector

    analytical_vjp = compute_vjp(
        eval_model,
        input_ids,
        layer_idx,
        eval_target,
        position=position,
        source_position=source_position,
        target_position=target_position,
        future_offsets=future_offsets,
        site=site,
        target_site=target_site,
        attention_mask=attention_mask,
    )

    seq_len = input_ids.shape[1]
    src_pos_idx, tgt_pos_indices = _resolve_positions(
        seq_len=seq_len,
        position=position,
        source_position=source_position,
        target_position=target_position,
        future_offsets=future_offsets,
        enforce_causal_horizon=True,
    )

    hidden_dim = analytical_vjp.shape[0]
    sample_indices = np.linspace(0, hidden_dim - 1, min(dims_to_check, hidden_dim), dtype=int)

    decoder_layers = find_decoder_layers(eval_model)
    source_layer = decoder_layers[layer_idx]

    numerical_grads = []
    analytical_grads = []

    for dim_idx in sample_indices:
        def run_perturbed(delta: float) -> float:
            perturb_hook = None
            try:
                def perturb_fn(_m: nn.Module, _inp: Any, output: Any) -> Any:
                    act = output[0] if isinstance(output, tuple) else output
                    act = act.clone()
                    act[0, src_pos_idx, dim_idx] += delta
                    if isinstance(output, tuple):
                        return (act,) + output[1:]
                    return act

                perturb_hook = source_layer.register_forward_hook(perturb_fn)
                with torch.no_grad():
                    kwargs: dict[str, Any] = {}
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask
                    out = eval_model(input_ids, **kwargs)
                    if target_site == "logits":
                        val = out.logits
                    else:
                        val = getattr(out, "last_hidden_state", out.logits)
                    slice_val = val[0, tgt_pos_indices].mean(dim=0)
                    scalar = torch.dot(
                        eval_target.to(slice_val.device, dtype=slice_val.dtype),
                        slice_val,
                    ).item()
                    return scalar
            finally:
                if perturb_hook is not None:
                    perturb_hook.remove()

        f_plus = run_perturbed(+epsilon)
        f_minus = run_perturbed(-epsilon)
        num_grad = (f_plus - f_minus) / (2.0 * epsilon)
        ana_grad = analytical_vjp[dim_idx].item()

        numerical_grads.append(num_grad)
        analytical_grads.append(ana_grad)

    ana_arr = np.array(analytical_grads)
    num_arr = np.array(numerical_grads)
    abs_errors = np.abs(ana_arr - num_arr)
    denom = np.maximum(np.abs(ana_arr) + np.abs(num_arr), 1e-7)
    rel_errors = 2.0 * abs_errors / denom

    max_rel = float(np.max(rel_errors))
    max_abs = float(np.max(abs_errors))
    passed = bool(max_rel < 0.05 or max_abs < 1e-3)

    return {
        "passed": passed,
        "source_position": src_pos_idx,
        "target_positions": tgt_pos_indices,
        "max_relative_error": max_rel,
        "max_absolute_error": max_abs,
        "checked_dims": sample_indices.tolist(),
        "analytical": ana_arr.tolist(),
        "numerical": num_arr.tolist(),
    }


def compare_logit_lens_and_jacobian_lens(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_idx: int,
    *,
    top_k: int = 10,
    position: int = -1,
    source_position: int | None = None,
    target_position: int | None = None,
    future_offsets: Sequence[int] | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Compare direct Logit Lens (W_U * h_l[t]) vs Jacobian Lens influence from source t to target t' >= t."""
    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    src_pos_idx, tgt_pos_indices = _resolve_positions(
        seq_len=seq_len,
        position=position,
        source_position=source_position,
        target_position=target_position,
        future_offsets=future_offsets,
        enforce_causal_horizon=True,
    )
    primary_tgt_idx = tgt_pos_indices[-1]
    token_str = tokenizer.decode(input_ids[0, src_pos_idx])

    decoder_layers = find_decoder_layers(model)
    intermediate_h: list[torch.Tensor] = []

    def capture_fn(_m: nn.Module, _i: Any, out: Any) -> None:
        act = out[0] if isinstance(out, tuple) else out
        intermediate_h.append(act[0, src_pos_idx].detach().clone())

    handle = decoder_layers[layer_idx].register_forward_hook(capture_fn)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()

    if not intermediate_h:
        raise RuntimeError("Failed to capture layer activation.")
    h_l = intermediate_h[0]

    W_U = get_unembedding_weight(model).to(device)
    logit_lens_scores = torch.matmul(W_U, h_l.to(W_U.dtype))

    top_logit_vals, top_logit_ids = torch.topk(logit_lens_scores.detach(), k=min(top_k, W_U.shape[0]))
    logit_top_list = [
        {"token_id": int(tid), "token": tokenizer.decode([tid]), "score": float(val)}
        for tid, val in zip(top_logit_ids.cpu(), top_logit_vals.cpu())
    ]

    final_logits = outputs.logits[0, primary_tgt_idx].detach()
    _, top_output_ids = torch.topk(final_logits, k=min(top_k, final_logits.shape[-1]))

    candidate_pool = torch.unique(torch.cat([top_logit_ids, top_output_ids]))

    jacobian_scores = []
    jacobian_vectors = []
    for tid in candidate_pool:
        v_k = W_U[tid]
        vjp_dir = compute_vjp(
            model,
            input_ids,
            layer_idx,
            v_k,
            source_position=src_pos_idx,
            target_position=target_position,
            future_offsets=future_offsets,
            position=primary_tgt_idx if (target_position is None and future_offsets is None) else position,
            target_site="final_residual",
        )
        align_score = torch.dot(h_l.float(), vjp_dir.float()).item()
        jacobian_scores.append(align_score)
        jacobian_vectors.append(vjp_dir)

    sorted_indices = np.argsort(jacobian_scores)[::-1]
    top_j_indices = sorted_indices[:top_k]
    jacobian_top_list = [
        {
            "token_id": int(candidate_pool[idx].cpu()),
            "token": tokenizer.decode([candidate_pool[idx].cpu()]),
            "score": float(jacobian_scores[idx]),
        }
        for idx in top_j_indices
    ]

    avg_vjp = torch.stack(jacobian_vectors).mean(dim=0).float()
    avg_wu = W_U[candidate_pool].mean(dim=0).float()

    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            avg_wu.unsqueeze(0), avg_vjp.unsqueeze(0)
        ).item()
    )

    torch.manual_seed(42)
    rand_dir = torch.randn_like(avg_vjp)
    rand_dir = rand_dir / torch.norm(rand_dir).clamp_min(1e-8) * torch.norm(avg_vjp)
    random_cos_sim = float(
        torch.nn.functional.cosine_similarity(
            avg_wu.unsqueeze(0), rand_dir.unsqueeze(0)
        ).item()
    )

    logit_scores_pool = [float(logit_lens_scores[tid].item()) for tid in candidate_pool]
    spearman_corr, _ = spearmanr(logit_scores_pool, jacobian_scores)

    return {
        "layer": layer_idx,
        "position": src_pos_idx,
        "source_position": src_pos_idx,
        "target_positions": tgt_pos_indices,
        "token": token_str,
        "candidate_pool_size": len(candidate_pool),
        "shortlist_mode": "union_logit_and_final_prediction",
        "logit_lens_top_k": logit_top_list,
        "jacobian_lens_top_k": jacobian_top_list,
        "cosine_similarity": cos_sim,
        "norm_matched_random_cosine_similarity": random_cos_sim,
        "rank_correlation_spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 1.0,
    }


def aggregate_jacobian_corpus(
    model: nn.Module,
    tokenizer: Any,
    corpus: Sequence[str],
    layer_idx: int,
    target_token_id: int,
    *,
    position: int = -1,
    source_positions: Sequence[int] | Literal["all", "last"] | None = None,
    future_offsets: Sequence[int] | None = None,
    aggregation_mode: Literal["mean", "median", "trimmed_mean"] = "mean",
    normalization_mode: Literal["none", "l2", "rms"] = "none",
    device: Any = None,
) -> dict[str, Any]:
    """Aggregate context- and position-averaged Jacobian Lens directions across a frozen prompt corpus (Prompt 28).

    Estimates:
        J_l(target) = E_{c in corpus, t in source_positions, delta in future_offsets} [ v_target^T (dh_c[L, t+delta] / dh_c[l, t]) ]
    """
    if not corpus:
        raise ValueError("Corpus cannot be empty.")
    if device is None:
        device = next(model.parameters()).device

    corpus_sha256 = hashlib.sha256(
        json.dumps(list(corpus), sort_keys=True).encode("utf-8")
    ).hexdigest()

    W_U = get_unembedding_weight(model).to(device)
    target_v = W_U[target_token_id]

    effective_offsets = list(future_offsets) if future_offsets is not None else [0, 1]

    vjp_list: list[np.ndarray] = []
    per_context_means: list[np.ndarray] = []
    pair_records: list[dict[str, Any]] = []

    for ctx_idx, text in enumerate(corpus):
        encoded = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])

        if source_positions is None:
            resolved_sources = [position if position >= 0 else seq_len + position]
        elif source_positions == "last":
            resolved_sources = [seq_len - 1]
        elif source_positions == "all":
            resolved_sources = list(range(seq_len))
        else:
            resolved_sources = [
                (p if p >= 0 else seq_len + p)
                for p in source_positions
                if 0 <= (p if p >= 0 else seq_len + p) < seq_len
            ]
            if not resolved_sources:
                resolved_sources = [seq_len - 1]

        ctx_vjps: list[np.ndarray] = []
        for src_t in resolved_sources:
            valid_targets = [
                src_t + int(off)
                for off in effective_offsets
                if 0 <= src_t + int(off) < seq_len and int(off) >= 0
            ]
            if not valid_targets:
                valid_targets = [seq_len - 1]

            for tgt_t in valid_targets:
                vjp = compute_vjp(
                    model,
                    input_ids,
                    layer_idx,
                    target_v,
                    source_position=src_t,
                    target_position=tgt_t,
                    normalize_vjp=normalization_mode,
                    target_site="final_residual",
                )
                vjp_np = vjp.cpu().float().numpy()
                vjp_list.append(vjp_np)
                ctx_vjps.append(vjp_np)
                pair_records.append(
                    {
                        "context_index": ctx_idx,
                        "source_position": int(src_t),
                        "target_position": int(tgt_t),
                        "horizon_offset": int(tgt_t - src_t),
                        "vjp_l2_norm": float(np.linalg.norm(vjp_np)),
                    }
                )

        per_context_means.append(np.mean(np.stack(ctx_vjps, axis=0), axis=0))

    vjp_stack = np.stack(vjp_list, axis=0)  # [n_total_pairs, dim]
    mean_dir = _aggregate_direction_stack(vjp_stack, mode=aggregation_mode)
    std_err = np.std(vjp_stack, axis=0) / np.sqrt(len(vjp_stack))

    # Compute cross-context directional stability (average pairwise cosine across contexts)
    ctx_stack = np.stack(per_context_means, axis=0)
    if len(ctx_stack) > 1:
        norms = np.linalg.norm(ctx_stack, axis=1, keepdims=True) + 1e-8
        unit_ctx = ctx_stack / norms
        cos_mat = unit_ctx @ unit_ctx.T
        tri_u = cos_mat[np.triu_indices(len(ctx_stack), k=1)]
        cross_context_stability = float(np.mean(tri_u))
    else:
        cross_context_stability = 1.0

    target_token_str = tokenizer.decode([target_token_id])

    return {
        "corpus_sha256": corpus_sha256,
        "sample_count": len(corpus),
        "evaluated_pair_count": len(vjp_stack),
        "layer_idx": layer_idx,
        "target_token_id": target_token_id,
        "target_token": target_token_str,
        "aggregation_mode": aggregation_mode,
        "normalization_mode": normalization_mode,
        "future_offsets": effective_offsets,
        "mean_direction_norm": float(np.linalg.norm(mean_dir)),
        "mean_direction": mean_dir.tolist(),
        "std_error_mean": float(np.mean(std_err)),
        "cross_context_cosine_stability": cross_context_stability,
        "position_pairs": pair_records,
    }


def evaluate_four_dimension_jacobian_evidence(
    model: nn.Module,
    tokenizer: Any,
    corpus: Sequence[str],
    layer_idx: int,
    target_token_id: int,
    *,
    control_token_id: int = 0,
    source_positions: Sequence[int] | Literal["all", "last"] = "all",
    future_offsets: Sequence[int] = (0, 1),
    intervention_alpha: float = 1.0,
    seed: int = 42,
    device: Any = None,
) -> dict[str, Any]:
    """Evaluate and strictly separate the 4 Jacobian Lens evidence dimensions (Prompt 31).

    Dimensions:
      1. `cosine_alignment`: Observational directional similarity vs. random and shuffled controls.
      2. `sensitivity`: Output response magnitude (||VJP||_2) across contexts and positions.
      3. `stability`: Persistence across contexts, source/target positions, and bootstrap resamples.
      4. `intervention_effects`: Controlled activation steering at (layer_idx, t) measuring
         causal target logit shift at t' >= t against norm-matched random and orthogonal sham controls.
    """
    if len(corpus) < 2:
        raise ValueError("At least 2 contexts are required to evaluate cross-context stability.")
    if device is None:
        device = next(model.parameters()).device

    agg = aggregate_jacobian_corpus(
        model,
        tokenizer,
        corpus,
        layer_idx=layer_idx,
        target_token_id=target_token_id,
        source_positions=source_positions,
        future_offsets=future_offsets,
        aggregation_mode="mean",
        normalization_mode="none",
        device=device,
    )

    W_U = get_unembedding_weight(model).to(device)
    target_u = W_U[target_token_id].detach().cpu().float().numpy()
    control_u = W_U[control_token_id].detach().cpu().float().numpy()
    mean_j = np.array(agg["mean_direction"], dtype=np.float32)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # 1. Dimension 1: Cosine Alignment (with random & shuffled-token negative controls)
    rng = np.random.default_rng(seed)
    rand_vec = rng.standard_normal(size=mean_j.shape).astype(np.float32)
    observed_cos = _cos(target_u, mean_j)
    random_cos = _cos(target_u, rand_vec)
    shuffled_token_cos = _cos(control_u, mean_j)

    dim1_alignment = {
        "operational_definition": "Directional cosine similarity between unembedding vector W_U[target] and context-averaged Jacobian direction J_l.",
        "observed_cosine_alignment": round(observed_cos, 4),
        "norm_matched_random_control_cosine": round(random_cos, 4),
        "unrelated_token_control_cosine": round(shuffled_token_cos, 4),
        "excess_alignment_over_random": round(observed_cos - random_cos, 4),
        "non_claim_guardrail": (
            "Observational directional similarity only. High cosine alignment does NOT prove semantic concept "
            "representation, cross-context stability, or causal influence."
        ),
    }

    # 2. Dimension 2: Sensitivity (magnitude of gradient response across contexts & positions)
    pair_norms = [float(p["vjp_l2_norm"]) for p in agg["position_pairs"]]
    dim2_sensitivity = {
        "operational_definition": "L2 norm of cross-position VJP response ||v_target^T (dh[L, t'] / dh[l, t])||_2 across contexts and positions.",
        "mean_vjp_l2_norm": round(float(np.mean(pair_norms)), 4),
        "std_vjp_l2_norm": round(float(np.std(pair_norms)), 4),
        "min_vjp_l2_norm": round(float(np.min(pair_norms)), 4),
        "max_vjp_l2_norm": round(float(np.max(pair_norms)), 4),
        "aggregated_direction_norm": round(float(agg["mean_direction_norm"]), 4),
    }

    # 3. Dimension 3: Stability (persistence across contexts, positions, and bootstrap splits)
    same_pos_norms = [p["vjp_l2_norm"] for p in agg["position_pairs"] if p["horizon_offset"] == 0]
    future_pos_norms = [p["vjp_l2_norm"] for p in agg["position_pairs"] if p["horizon_offset"] > 0]
    dim3_stability = {
        "operational_definition": "Directional persistence across distinct contexts (pairwise cosine) and across current vs. future token horizons.",
        "cross_context_pairwise_cosine_mean": round(float(agg["cross_context_cosine_stability"]), 4),
        "std_error_across_pairs": round(float(agg["std_error_mean"]), 4),
        "same_position_mean_norm": round(float(np.mean(same_pos_norms)) if same_pos_norms else 0.0, 4),
        "future_horizon_mean_norm": round(float(np.mean(future_pos_norms)) if future_pos_norms else 0.0, 4),
        "evaluated_context_count": int(agg["sample_count"]),
        "evaluated_pair_count": int(agg["evaluated_pair_count"]),
    }

    # 4. Dimension 4: Controlled Intervention Effects (causal test vs. norm-matched random & orthogonal sham)
    j_norm = float(np.linalg.norm(mean_j))
    unit_j = mean_j / max(j_norm, 1e-8)

    # Construct norm-matched random direction and Gram-Schmidt orthogonal sham direction
    unit_rand = rand_vec / max(float(np.linalg.norm(rand_vec)), 1e-8)
    sham_raw = rng.standard_normal(size=mean_j.shape).astype(np.float32)
    sham_orth = sham_raw - np.dot(sham_raw, unit_j) * unit_j
    unit_sham = sham_orth / max(float(np.linalg.norm(sham_orth)), 1e-8)

    decoder_layers = find_decoder_layers(model)
    source_layer = decoder_layers[layer_idx]

    target_shifts = {"candidate": [], "random_control": [], "orthogonal_sham": []}
    control_regression_shifts = {"candidate": [], "random_control": [], "orthogonal_sham": []}

    for text in corpus:
        encoded = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])
        src_pos = 0
        tgt_pos = seq_len - 1

        with torch.no_grad():
            base_logits = model(**inputs).logits[0, tgt_pos].detach().cpu().float().numpy()

        for arm_name, vec_np in [
            ("candidate", unit_j),
            ("random_control", unit_rand),
            ("orthogonal_sham", unit_sham),
        ]:
            delta_t = torch.tensor(vec_np * intervention_alpha, device=device)

            def _steer_hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
                act = output[0] if isinstance(output, tuple) else output
                act = act.clone()
                act[0, src_pos] = act[0, src_pos] + delta_t.to(act.dtype)
                if isinstance(output, tuple):
                    return (act,) + output[1:]
                return act

            h = source_layer.register_forward_hook(_steer_hook)
            try:
                with torch.no_grad():
                    steered_logits = model(**inputs).logits[0, tgt_pos].detach().cpu().float().numpy()
            finally:
                h.remove()

            target_shifts[arm_name].append(float(steered_logits[target_token_id] - base_logits[target_token_id]))
            control_regression_shifts[arm_name].append(
                float( abs(steered_logits[control_token_id] - base_logits[control_token_id]) )
            )

    cand_mean_shift = float(np.mean(target_shifts["candidate"]))
    rand_mean_shift = float(np.mean(target_shifts["random_control"]))
    sham_mean_shift = float(np.mean(target_shifts["orthogonal_sham"]))
    selectivity_margin = cand_mean_shift - max(rand_mean_shift, sham_mean_shift)

    dim4_intervention = {
        "operational_definition": (
            "Causal change in target token logit at future position t' under reversible activation addition "
            "at (layer_idx, t) compared against norm-matched random and orthogonal sham controls."
        ),
        "intervention_alpha": float(intervention_alpha),
        "candidate_mean_target_logit_shift": round(cand_mean_shift, 4),
        "random_control_mean_target_logit_shift": round(rand_mean_shift, 4),
        "orthogonal_sham_mean_target_logit_shift": round(sham_mean_shift, 4),
        "candidate_mean_unrelated_token_drift": round(float(np.mean(control_regression_shifts["candidate"])), 4),
        "selectivity_margin_over_controls": round(selectivity_margin, 4),
        "causal_claim_supported": bool(cand_mean_shift > 0.0 and selectivity_margin > 0.0),
    }

    return {
        "layer_idx": layer_idx,
        "target_token_id": target_token_id,
        "control_token_id": control_token_id,
        "method_mapping": JACOBIAN_LENS_METHOD_MAPPING,
        "dimension_1_cosine_alignment": dim1_alignment,
        "dimension_2_sensitivity": dim2_sensitivity,
        "dimension_3_stability": dim3_stability,
        "dimension_4_intervention_effects": dim4_intervention,
    }
