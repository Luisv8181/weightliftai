"""Layer Atlas: Multi-depth representation capture, trajectory analysis, and reporting.

Extracts representation artifacts across layers in one controlled forward pass,
computes original-space paired metrics with bootstrap confidence intervals,
and generates self-contained visual trajectory reports.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from weightliftai.cartography.schemas import (
    CaptureSpec,
    DatasetRef,
    ModelRef,
    RepresentationArtifact,
)
from weightliftai.cartography.validators import validate_matched_dataset


def find_decoder_layers(model: Any):
    """Locate the transformer decoder layers in a model."""
    import torch

    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    # Common direct attributes
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers") and isinstance(model.layers, torch.nn.ModuleList):
        return model.layers

    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and (n is None or len(mod) == n):
            return mod
    raise RuntimeError("Could not locate decoder layers in model.")


def content_token_mask(attention_mask: Any):
    """Zero out padding tokens and the first non-padding token (BOS).
    
    Handles left padding, right padding, and preserves single-token sequences
    so representation vectors are never entirely zeroed out.
    """
    import torch

    mask = attention_mask.clone()
    for b in range(mask.shape[0]):
        valid = (mask[b] == 1).nonzero(as_tuple=True)[0]
        if len(valid) > 1:
            # Zero out the first valid content position (BOS)
            mask[b, valid[0]] = 0
    return mask


def extract_layer_representations(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layers: Sequence[int],
    *,
    site: Literal["residual", "attention", "mlp"] = "residual",
    token_mask: Literal["content", "all", "last"] = "content",
    pooling: Literal["mean", "last", "max"] = "mean",
    batch_size: int = 16,
    device: Any = None,
) -> dict[int, np.ndarray]:
    """Capture layer activations across requested depths in a single controlled pass.
    
    Site definitions:
    - 'residual': output of the transformer block (post-block residual stream x_{L+1}).
    - 'attention': output of self-attention submodule prior to residual add.
    - 'mlp': output of feed-forward submodule prior to residual add.
    
    Token masking and pooling properly handle left/right padding and non-padding
    token selection across variable-length batches.
    """
    import torch

    decoder_blocks = find_decoder_layers(model)
    max_layer = len(decoder_blocks) - 1
    for L in layers:
        if L < 0 or L > max_layer:
            raise ValueError(f"Requested layer {L} is out of bounds (0 to {max_layer}).")

    if device is None:
        device = next(model.parameters()).device

    # Resolve target module per layer based on site
    target_modules = {}
    for L in layers:
        block = decoder_blocks[L]
        if site == "residual":
            target_modules[L] = block
        elif site == "attention":
            mod = getattr(block, "self_attn", getattr(block, "attention", None))
            if mod is None:
                raise ValueError(f"Could not locate attention submodule in layer {L}")
            target_modules[L] = mod
        elif site == "mlp":
            mod = getattr(block, "mlp", getattr(block, "feed_forward", None))
            if mod is None:
                raise ValueError(f"Could not locate MLP submodule in layer {L}")
            target_modules[L] = mod

    cache: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(_mod: Any, _inp: Any, output: Any):
            val = output[0] if isinstance(output, (tuple, list)) else output
            if val is not None and hasattr(val, "detach"):
                cache[layer_idx] = val.detach().float()
        return _hook

    out_by_layer: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    model.eval()

    try:
        for L in layers:
            hooks.append(target_modules[L].register_forward_hook(make_hook(L)))

        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                enc = tokenizer(batch_texts, return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
                model(**enc)

                # Token masking with robust left/right padding handling
                attn_mask = enc.get("attention_mask")
                if attn_mask is not None:
                    if token_mask == "content":
                        mask = content_token_mask(attn_mask)[..., None].float()
                    elif token_mask == "all":
                        mask = attn_mask[..., None].float()
                    else:  # "last" non-padding token
                        last_mask = torch.zeros_like(attn_mask).float()
                        for b in range(attn_mask.shape[0]):
                            valid = (attn_mask[b] == 1).nonzero(as_tuple=True)[0]
                            if len(valid) > 0:
                                last_mask[b, valid[-1]] = 1.0
                            else:
                                last_mask[b, -1] = 1.0
                        mask = last_mask[..., None]
                else:
                    mask = torch.ones_like(cache[layers[0]][..., :1]).float()

                denom = mask.sum(1).clamp(min=1.0)

                # Precompute last valid token indices per row for pooling == "last"
                last_indices = []
                if attn_mask is not None:
                    for b in range(attn_mask.shape[0]):
                        valid = (attn_mask[b] == 1).nonzero(as_tuple=True)[0]
                        last_indices.append(valid[-1].item() if len(valid) > 0 else -1)
                else:
                    last_indices = [-1] * len(batch_texts)

                for L in layers:
                    h = cache[L]
                    if pooling == "mean":
                        pooled = ((h * mask).sum(1) / denom).cpu().numpy()
                    elif pooling == "last":
                        # Index at the true last non-padding token position
                        last_vecs = [h[b, last_indices[b]] for b in range(h.shape[0])]
                        pooled = torch.stack(last_vecs).cpu().numpy()
                    elif pooling == "max":
                        # Set masked positions to -inf before max
                        h_masked = h.clone()
                        h_masked[mask.squeeze(-1) == 0] = -1e9
                        pooled = h_masked.max(1).values.cpu().numpy()
                    else:
                        raise ValueError(f"Unknown pooling method: {pooling}")
                    out_by_layer[L].append(pooled)

    finally:
        for h_hook in hooks:
            h_hook.remove()

    return {L: np.vstack(v) for L, v in out_by_layer.items()}


def compute_layer_metrics(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    *,
    item_id_col: str = "item_id",
    variant_col: str = "variant",
    variants: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute original-space pairwise cosines and paired bootstrap confidence intervals."""
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norm

    df_clean = df.reset_index(drop=True)
    all_variants = variants or sorted(list(df_clean[variant_col].unique()))
    item_ids = df_clean[item_id_col].unique()

    # Pre-index item rows per variant
    var_indices = {v: {} for v in all_variants}
    for idx, row in df_clean.iterrows():
        v = row[variant_col]
        it = row[item_id_col]
        if v in var_indices:
            var_indices[v][it] = idx

    # Calculate item-level pairwise cosine similarities
    pair_cosines: dict[str, list[float]] = {}
    for i, v1 in enumerate(all_variants):
        for v2 in all_variants[i + 1 :]:
            pair_key = f"{v1}~{v2}"
            cos_list = []
            for it in item_ids:
                if it in var_indices[v1] and it in var_indices[v2]:
                    idx1 = var_indices[v1][it]
                    idx2 = var_indices[v2][it]
                    dot = float(emb_norm[idx1] @ emb_norm[idx2])
                    cos_list.append(dot)
            pair_cosines[pair_key] = cos_list

    # Calculate unrelated-concept negative control baseline (null distribution)
    item_arr = np.array(df_clean[item_id_col])
    n_samples = len(embeddings)
    cos_matrix = emb_norm @ emb_norm.T

    # Distinct concept pairs (upper triangle)
    diff_mask = (item_arr[:, None] != item_arr[None, :]) & np.triu(np.ones((n_samples, n_samples), dtype=bool), k=1)
    unrelated_cos = cos_matrix[diff_mask]

    rng = np.random.default_rng(seed)
    unrelated_stats: dict[str, Any] = {}
    unrel_mean = 0.0

    if len(unrelated_cos) > 0:
        unrel_mean = float(unrelated_cos.mean())
        if n_bootstrap > 0 and len(unrelated_cos) > 1:
            sample_unrel = unrelated_cos if len(unrelated_cos) <= 5000 else rng.choice(unrelated_cos, size=5000, replace=False)
            u_boot_idx = rng.integers(0, len(sample_unrel), size=(n_bootstrap, len(sample_unrel)))
            u_boot_means = sample_unrel[u_boot_idx].mean(axis=1)
            u_ci_low = float(np.percentile(u_boot_means, 2.5))
            u_ci_high = float(np.percentile(u_boot_means, 97.5))
        else:
            u_ci_low, u_ci_high = unrel_mean, unrel_mean
        unrelated_stats = {
            "mean": round(unrel_mean, 4),
            "mean_raw": unrel_mean,
            "ci_95": [round(u_ci_low, 4), round(u_ci_high, 4)],
            "ci_95_raw": [u_ci_low, u_ci_high],
            "n_pairs": int(len(unrelated_cos)),
        }

    # Paired concept-level bootstrap 95% confidence intervals per pair
    bootstrap_results = {}
    for pair_key, cos_vals in pair_cosines.items():
        if not cos_vals:
            continue
        arr = np.array(cos_vals)
        mean_val = float(arr.mean())
        if n_bootstrap > 0 and len(arr) > 1:
            boot_idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
            boot_means = arr[boot_idx].mean(axis=1)
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
        else:
            ci_low, ci_high = mean_val, mean_val

        delta_over_unrelated = float(mean_val - unrel_mean) if unrelated_stats else None

        bootstrap_results[pair_key] = {
            "mean": round(mean_val, 4),
            "mean_raw": mean_val,
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "ci_95_raw": [ci_low, ci_high],
            "n_items": len(arr),
            "delta_over_unrelated_baseline": round(delta_over_unrelated, 4) if delta_over_unrelated is not None else None,
        }

    return {
        "n_samples": len(embeddings),
        "feature_dim": int(embeddings.shape[1]),
        "evidence_type": "dimension_1_cosine_alignment",
        "guardrail": (
            "Observational directional similarity under defined pairwise comparisons. "
            "Does NOT establish semantic representation, stability across contexts, "
            "or causal necessity/intervention effects (see Prompt 31 & docs/RESEARCH_VALIDATION_PROTOCOL.md)."
        ),
        "pairwise_cosines": bootstrap_results,
        "unrelated_negative_control": unrelated_stats,
    }


def generate_atlas_html(
    model_name: str,
    dataset_name: str,
    layer_metrics: dict[int, dict[str, Any]],
    *,
    title: str = "WeightLiftAI Layer Atlas",
) -> str:
    """Generate a self-contained, publication-grade HTML report of the layer trajectory."""
    sorted_layers = sorted(layer_metrics.keys())
    layers_data = []
    pair_names = set()

    for L in sorted_layers:
        m = layer_metrics[L]
        d_pct = m.get("depth_pct", round(L / max(1, sorted_layers[-1]) * 100, 1))
        pairs = m.get("pairwise_cosines", {})
        pair_names.update(pairs.keys())
        layers_data.append({
            "layer": L,
            "depth_pct": d_pct,
            "pairs": pairs,
        })

    pair_list = sorted(list(pair_names))
    palette = ["#3fe0c5", "#ffb35c", "#ff6f61", "#b69cff", "#9be36b"]

    table_rows = []
    for row in layers_data:
        L = row["layer"]
        dp = row["depth_pct"]
        unrel = layer_metrics[L].get("unrelated_negative_control", {})
        unrel_col = f"<td style='padding:8px 12px;color:#ffb35c'><b>{unrel.get('mean', '—')}</b> <span style='color:#8ea3ad;font-size:11px'>[{unrel.get('ci_95', ['—', '—'])[0]}, {unrel.get('ci_95', ['—', '—'])[1]}]</span></td>" if unrel else "<td style='padding:8px 12px;color:#8ea3ad'>—</td>"

        cols = [f"<td style='padding:8px 12px;font-weight:600'>{L}</td><td style='padding:8px 12px;color:#8ea3ad'>{dp:.1f}%</td>"]
        for p in pair_list:
            p_data = row["pairs"].get(p)
            if p_data:
                mean = p_data["mean"]
                ci = p_data.get("ci_95", [mean, mean])
                delta_str = f" <span style='font-size:10px;color:#3fe0c5'>(&Delta;+{p_data['delta_over_unrelated_baseline']:.3f})</span>" if p_data.get("delta_over_unrelated_baseline") is not None else ""
                cols.append(f"<td style='padding:8px 12px'><b>{mean:.3f}</b> <span style='color:#8ea3ad;font-size:11px'>[{ci[0]:.2f}, {ci[1]:.2f}]</span>{delta_str}</td>")
            else:
                cols.append("<td style='padding:8px 12px;color:#8ea3ad'>—</td>")
        cols.append(unrel_col)
        table_rows.append(f"<tr style='border-bottom:1px solid #23324a'>{''.join(cols)}</tr>")

    headers = [f"<th style='padding:8px 12px;text-align:left;color:#8ea3ad'>{html.escape(p)}</th>" for p in pair_list]
    headers.append("<th style='padding:8px 12px;text-align:left;color:#ffb35c'>Unrelated Control (Null)</th>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — {html.escape(model_name)}</title>
<style>
:root {{
  --bg: #0a0e1c; --panel: #111a2e; --ink: #ecebe4; --muted: #8ea3ad; --line: #23324a;
  --teal: #3fe0c5; --amber: #ffb35c; --coral: #ff6f61; --lilac: #b69cff;
}}
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ background: var(--bg); color: var(--ink); font-family: Inter, -apple-system, system-ui, sans-serif; line-height: 1.5; padding: 32px 24px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
.kicker {{ color: var(--teal); letter-spacing: .15em; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }}
h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }}
.meta {{ color: var(--muted); font-size: 13.5px; margin-bottom: 24px; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
th {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; }}
.guardrail {{ border-left: 3px solid var(--amber); background: #261f14; padding: 12px 16px; border-radius: 6px; font-size: 12.5px; color: #ffd36b; margin-top: 16px; }}
</style>
</head>
<body>
<div class="container">
  <div class="kicker">WeightLiftAI · Representation Cartography</div>
  <h1>Layer Atlas: Trajectory & Representation Alignment</h1>
  <div class="meta">Model: <b>{html.escape(model_name)}</b> &nbsp;|&nbsp; Dataset: <b>{html.escape(dataset_name)}</b> &nbsp;|&nbsp; Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>

  <div class="panel">
    <h3 style="font-size:15px;margin-bottom:14px;color:var(--teal)">Dimension 1: Observational Directional Alignment (Original Vector Space Cosines)</h3>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr style="border-bottom:2px solid var(--line)">
            <th style="padding:8px 12px;text-align:left;color:#8ea3ad">Layer</th>
            <th style="padding:8px 12px;text-align:left;color:#8ea3ad">Depth %</th>
            {''.join(headers)}
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
    <div class="guardrail">
      <b>Scientific Guardrail (Prompt 31):</b> Values represent Dimension 1 (observational pairwise cosine similarities) in the original high-dimensional representation space with paired bootstrap 95% confidence intervals (2.5%–97.5%). Observational directional alignment does not establish semantic concept representation, sensitivity, stability across contexts, or causal intervention effects. Screen-space projections (e.g. 2D PCA) are visual guides only and must not be substituted for original-space metrics.
    </div>
  </div>
</div>
</body>
</html>"""
