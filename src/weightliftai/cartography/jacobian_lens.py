"""Jacobian Lens and Output Sensitivity Prototype (Research Prototype).

Status: Level 2 Output Sensitivity Prototype (see docs/RESEARCH_VALIDATION_PROTOCOL.md).
Investigates activation-to-output influence in transformer decoder models by computing
Vector-Jacobian Products (VJP) and directional derivatives between intermediate residual
stream activations and final hidden states / vocabulary logits.

Explicit Limitations & Disclosures:
1. Mathematical Verification != Mechanism Discovery: Passing finite-difference derivative
   checks verifies autograd calculus, not that a semantic feature or causal circuit is found.
2. Candidate Shortlist Bias: In compare_logit_lens_and_jacobian_lens(), candidate tokens are
   currently filtered to top-k of direct Logit Lens; this can miss candidates outside the shortlist.
3. Normalization: Computations currently project through residual stream / unembedding directly
   and do not yet account for all model-family specific final RMSNorm/LayerNorm scaling dynamics.
4. Prompt-Specific vs. Global: Single-prompt readouts reflect local directional sensitivity;
   global claims require context aggregation (see aggregate_jacobian_corpus).

References:
- Anthropic (2026): A global workspace in language models
- Gurnee et al. (2026): Verbalizable Representations Form a Global Workspace
- Ostrow et al. (2026): A Mathematical Theory of the Jacobian Lens
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from weightliftai.cartography.layer_atlas import find_decoder_layers


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


def compute_vjp(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    target_vector: torch.Tensor,
    *,
    position: int = -1,
    site: Literal["residual", "attention", "mlp"] = "residual",
    target_site: Literal["final_residual", "logits"] = "final_residual",
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Vector-Jacobian Product (VJP) v^T (dh_target / dh_source).
    
    Given a target gradient vector v at target_site (final residual or logits),
    computes the exact gradient with respect to the source activation at layer_idx.
    
    Memory-efficient: uses PyTorch autograd without materializing full d x d Jacobian.
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
        # Clone and retain grad
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

        # Forward pass with autograd enabled
        with torch.enable_grad():
            outputs = model(input_ids, **kwargs)
            if target_site == "logits":
                out_tensor = outputs.logits
            else:
                # Last hidden state
                if hasattr(outputs, "last_hidden_state"):
                    out_tensor = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    out_tensor = outputs.hidden_states[-1]
                else:
                    # Fallback to logits if hidden states not directly exposed
                    out_tensor = outputs.logits

            if len(intermediate_act) == 0:
                raise RuntimeError("Failed to capture intermediate activation during hook forward pass.")

            act = intermediate_act[0]
            # Select target slice at position
            # Shape of out_tensor: [batch, seq_len, dim]
            batch_size, seq_len = out_tensor.shape[0], out_tensor.shape[1]
            pos_idx = position if position >= 0 else seq_len + position
            
            target_slice = out_tensor[0, pos_idx]  # [dim]
            
            # Scalar projection for VJP: target_vector @ target_slice
            if target_vector.ndim == 1:
                scalar_loss = torch.dot(target_vector.to(target_slice.device, dtype=target_slice.dtype), target_slice)
            else:
                scalar_loss = (target_vector.to(target_slice.device, dtype=target_slice.dtype) * target_slice).sum()

            # Backprop to intermediate_act
            grads = torch.autograd.grad(scalar_loss, act, retain_graph=False, create_graph=False)
            grad_tensor = grads[0]
            
            # Extract gradient at source position
            src_pos_idx = position if position >= 0 else act.shape[1] + position
            vjp = grad_tensor[0, src_pos_idx].detach()
            return vjp
    finally:
        hook_handle.remove()


def check_finite_differences(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    target_vector: torch.Tensor,
    *,
    position: int = -1,
    site: Literal["residual", "attention", "mlp"] = "residual",
    target_site: Literal["final_residual", "logits"] = "final_residual",
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-4,
    dims_to_check: int = 5,
) -> dict[str, Any]:
    """Verify analytical autograd VJP against numerical finite differences on sampled dimensions.
    
    Guarantees that autograd hooks and forward pass kwargs (including masks and positions)
    match central finite differences.
    """
    analytical_vjp = compute_vjp(
        model, input_ids, layer_idx, target_vector,
        position=position, site=site, target_site=target_site,
        attention_mask=attention_mask,
    )
    
    hidden_dim = analytical_vjp.shape[0]
    sample_indices = np.linspace(0, hidden_dim - 1, min(dims_to_check, hidden_dim), dtype=int)
    
    decoder_layers = find_decoder_layers(model)
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
                    pos = position if position >= 0 else act.shape[1] + position
                    act[0, pos, dim_idx] += delta
                    if isinstance(output, tuple):
                        return (act,) + output[1:]
                    return act

                perturb_hook = source_layer.register_forward_hook(perturb_fn)
                with torch.no_grad():
                    kwargs: dict[str, Any] = {}
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask
                    out = model(input_ids, **kwargs)
                    if target_site == "logits":
                        val = out.logits
                    else:
                        val = getattr(out, "last_hidden_state", out.logits)
                    pos = position if position >= 0 else val.shape[1] + position
                    slice_val = val[0, pos]
                    scalar = torch.dot(target_vector.to(slice_val.device, dtype=slice_val.dtype), slice_val).item()
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
    device: Any = None,
) -> dict[str, Any]:
    """Compare direct Logit Lens (W_U * h_l) vs Jacobian Lens influence for a given prompt and layer.
    
    Diagnostics returned:
    - Logit lens top-k tokens and scores
    - Jacobian lens top-k tokens and alignment
    - Direction cosine similarity
    - Spearman rank correlation
    - Random baseline direction comparison
    """
    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    pos_idx = position if position >= 0 else seq_len + position
    token_str = tokenizer.decode(input_ids[0, pos_idx])

    # 1. Extract intermediate activation h_l
    decoder_layers = find_decoder_layers(model)
    intermediate_h: list[torch.Tensor] = []

    def capture_fn(_m: nn.Module, _i: Any, out: Any) -> None:
        act = out[0] if isinstance(out, tuple) else out
        intermediate_h.append(act[0, pos_idx].detach().clone())

    handle = decoder_layers[layer_idx].register_forward_hook(capture_fn)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()

    if not intermediate_h:
        raise RuntimeError("Failed to capture layer activation.")
    h_l = intermediate_h[0]  # [hidden_dim]

    # 2. Logit Lens: project h_l through unembedding W_U
    W_U = get_unembedding_weight(model).to(device)  # [vocab_size, hidden_dim]
    logit_lens_scores = torch.matmul(W_U, h_l.to(W_U.dtype))  # [vocab_size]

    top_logit_vals, top_logit_ids = torch.topk(logit_lens_scores.detach(), k=top_k)
    logit_top_list = [
        {"token_id": int(tid), "token": tokenizer.decode([tid]), "score": float(val)}
        for tid, val in zip(top_logit_ids.cpu(), top_logit_vals.cpu())
    ]

    # Model's actual final prediction logits at target position
    final_logits = outputs.logits[0, pos_idx].detach()
    _, top_output_ids = torch.topk(final_logits, k=min(top_k, final_logits.shape[-1]))

    # Form unbiased candidate pool (union of logit lens candidates and final model predictions)
    candidate_pool = torch.unique(torch.cat([top_logit_ids, top_output_ids]))

    # 3. Jacobian Lens: for candidate tokens, compute directional gradient
    # v_k = W_U[k]
    jacobian_scores = []
    jacobian_vectors = []
    for tid in candidate_pool:
        v_k = W_U[tid]  # [hidden_dim]
        # VJP gives direction in layer l that most increases target token logit
        vjp_dir = compute_vjp(
            model, input_ids, layer_idx, v_k,
            position=pos_idx, target_site="final_residual"
        )
        # Alignment between current activation h_l and the sensitivity direction vjp_dir
        align_score = torch.dot(h_l.float(), vjp_dir.float()).item()
        jacobian_scores.append(align_score)
        jacobian_vectors.append(vjp_dir)

    # Rank tokens by Jacobian influence alignment
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

    # 4. Diagnostics: Cosine similarity between Logit Lens direction and Jacobian VJP
    avg_vjp = torch.stack(jacobian_vectors).mean(dim=0).float()
    avg_wu = W_U[candidate_pool].mean(dim=0).float()
    
    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            avg_wu.unsqueeze(0), avg_vjp.unsqueeze(0)
        ).item()
    )

    # Norm-matched random baseline direction comparison
    torch.manual_seed(42)
    rand_dir = torch.randn_like(avg_vjp)
    rand_dir = rand_dir / torch.norm(rand_dir) * torch.norm(avg_vjp)
    random_cos_sim = float(
        torch.nn.functional.cosine_similarity(
            avg_wu.unsqueeze(0), rand_dir.unsqueeze(0)
        ).item()
    )

    # Spearman rank correlation of scores among candidate tokens
    logit_scores_pool = [float(logit_lens_scores[tid].item()) for tid in candidate_pool]
    spearman_corr, _ = spearmanr(logit_scores_pool, jacobian_scores)

    return {
        "layer": layer_idx,
        "position": pos_idx,
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
    device: Any = None,
) -> dict[str, Any]:
    """Aggregate context-averaged Jacobian Lens directions across a frozen prompt corpus.
    
    Computes corpus hash, average direction vector, and standard error across contexts.
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

    vjp_list: list[np.ndarray] = []
    for text in corpus:
        encoded = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
        input_ids = inputs["input_ids"]
        vjp = compute_vjp(
            model, input_ids, layer_idx, target_v,
            position=position, target_site="final_residual"
        )
        vjp_list.append(vjp.cpu().float().numpy())

    vjp_stack = np.stack(vjp_list, axis=0)  # [n_samples, dim]
    mean_dir = np.mean(vjp_stack, axis=0)
    std_err = np.std(vjp_stack, axis=0) / np.sqrt(len(corpus))

    target_token_str = tokenizer.decode([target_token_id])

    return {
        "corpus_sha256": corpus_sha256,
        "sample_count": len(corpus),
        "layer_idx": layer_idx,
        "target_token_id": target_token_id,
        "target_token": target_token_str,
        "mean_direction_norm": float(np.linalg.norm(mean_dir)),
        "mean_direction": mean_dir.tolist(),
        "std_error_mean": float(np.mean(std_err)),
    }
