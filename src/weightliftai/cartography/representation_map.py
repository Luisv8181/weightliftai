"""Representation Map: 2D/3D Dimensionality reduction with mandatory geometry guardrails.

Projects high-dimensional representation artifacts into exploratory 2D/3D coordinate space
while strictly preserving high-D metric traceability and reporting projection distortion:
1. Deterministic baseline PCA with explained-variance ratio.
2. Optional non-linear exploratory projections (t-SNE, UMAP).
3. Distortion Diagnostics:
   - Trustworthiness T(k): penalizes false 2D neighbors.
   - Continuity C(k): penalizes false 2D tears.
   - k-NN Overlap R@k: intersection between original-space and projected-space neighborhoods.
4. Point Inspection & Matched Triad Alignment:
   - Original-space cosine nearest neighbors.
   - True high-D distance vs. 2D screen distance to matched triad partners.
5. Self-contained publication-grade interactive HTML visualization.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

from weightliftai.cartography.schemas import ProjectionArtifact


def compute_distortion_metrics(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    k_neighbors: int = 5,
) -> dict[str, Any]:
    """Calculate trustworthiness, continuity, and neighborhood preservation.
    
    Adheres to the Representation Cartography guardrail: Projections are not geometry.
    Never substitute screen-space Euclidean distance for original-space metrics.
    """
    n_samples = len(high_dim)
    k = min(k_neighbors, max(1, n_samples - 2))

    if n_samples < 4:
        return {
            "k_neighbors": k,
            "trustworthiness": 1.0,
            "continuity": 1.0,
            "knn_overlap": 1.0,
            "distortion_warning": False,
            "interpretive_note": "Sample size too small for statistical distortion diagnostics.",
        }

    # Trustworthiness: penalizes points close in low-D that were NOT close in high-D
    t_val = float(trustworthiness(high_dim, low_dim, n_neighbors=k))

    # Continuity: dual of trustworthiness (swap high-D and low-D)
    c_val = float(trustworthiness(low_dim, high_dim, n_neighbors=k))

    # k-NN Neighborhood Overlap (Jaccard / intersection ratio)
    norm_high = np.linalg.norm(high_dim, axis=1, keepdims=True) + 1e-9
    high_norm = high_dim / norm_high
    cos_sim = high_norm @ high_norm.T

    # Low-D Euclidean distance
    diff = low_dim[:, None, :] - low_dim[None, :, :]
    dist_low = np.linalg.norm(diff, axis=-1)

    overlaps = []
    for i in range(n_samples):
        # Exclude self
        high_order = np.argsort(-cos_sim[i])
        high_nn = set([idx for idx in high_order if idx != i][:k])

        low_order = np.argsort(dist_low[i])
        low_nn = set([idx for idx in low_order if idx != i][:k])

        overlap = len(high_nn & low_nn) / float(k)
        overlaps.append(overlap)

    mean_overlap = float(np.mean(overlaps))
    warning = bool(t_val < 0.80 or c_val < 0.80 or mean_overlap < 0.50)

    return {
        "k_neighbors": k,
        "trustworthiness": round(t_val, 4),
        "continuity": round(c_val, 4),
        "knn_overlap": round(mean_overlap, 4),
        "distortion_warning": warning,
        "interpretive_note": (
            "Screen-space distance in 2D is a lossy visual approximation. "
            "Never substitute 2D cluster proximity for true high-dimensional similarity or causal identity."
        ),
    }


def project_representations(
    embeddings: np.ndarray,
    item_ids: Sequence[str | int],
    *,
    algorithm: Literal["pca", "umap", "tsne"] = "pca",
    n_components: int = 2,
    random_seed: int = 42,
    source_sha256: str = "placeholder_sha256",
    k_neighbors: int = 5,
    **kwargs: Any,
) -> tuple[ProjectionArtifact, np.ndarray]:
    """Fit dimensionality reduction with deterministic parameters and compute distortion diagnostics."""
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")
    if len(embeddings) != len(item_ids):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(item_ids)} item_ids.")

    explained_var = None
    if algorithm == "pca":
        pca = PCA(n_components=n_components, random_state=random_seed)
        coords = pca.fit_transform(embeddings)
        explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    elif algorithm == "tsne":
        perp = kwargs.get("perplexity", min(30, max(5, (len(embeddings) - 1) // 3)))
        tsne = TSNE(
            n_components=n_components,
            random_state=random_seed,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(embeddings)
    elif algorithm == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_seed,
                n_neighbors=kwargs.get("n_neighbors", min(15, len(embeddings) - 1)),
                min_dist=kwargs.get("min_dist", 0.1),
            )
            coords = reducer.fit_transform(embeddings)
        except ImportError as err:
            raise ImportError(
                "UMAP is not installed. Install with 'pip install umap-learn' or use algorithm='pca'."
            ) from err
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'pca', 'umap', or 'tsne'.")

    # Center and scale coordinates into normalized box [-10, 10] for consistent rendering
    coords_arr = np.asarray(coords, dtype=float)
    c_min = coords_arr.min(axis=0)
    c_max = coords_arr.max(axis=0)
    c_span = np.where((c_max - c_min) == 0, 1.0, (c_max - c_min))
    norm_coords = ((coords_arr - c_min) / c_span * 20.0 - 10.0)

    # Compute distortion metrics
    dist_metrics = compute_distortion_metrics(embeddings, norm_coords, k_neighbors=k_neighbors)

    artifact = ProjectionArtifact(
        artifact_version="v1",
        source_artifact_sha256=source_sha256,
        algorithm=algorithm,
        n_components=n_components,
        random_seed=random_seed,
        explained_variance_ratio=explained_var,
        coordinates=[[round(float(c), 4) for c in row] for row in norm_coords],
        item_ids=list(item_ids),
        distortion_metrics=dist_metrics,
    )

    return artifact, norm_coords


def build_point_inspection_data(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    *,
    k_neighbors: int = 5,
    item_id_col: str = "item_id",
    concept_col: str = "concept",
    variant_col: str = "variant",
    text_col: str = "text",
    lang_col: str = "lang",
    split_col: str = "split",
) -> list[dict[str, Any]]:
    """Build enriched inspection data with true high-D nearest neighbors and matched triad alignment."""
    n_samples = len(embeddings)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norm
    cos_matrix = emb_norm @ emb_norm.T

    # Index by item_id to find triad partners
    item_map: dict[Any, dict[str, int]] = {}
    for idx, row in df.iterrows():
        it = row.get(item_id_col)
        var = row.get(variant_col, "default")
        if it not in item_map:
            item_map[it] = {}
        item_map[it][var] = int(idx)

    points = []
    for i in range(n_samples):
        row = df.iloc[i]
        it_id = row.get(item_id_col, f"item_{i}")
        var = row.get(variant_col, "unknown")
        concept_lbl = row.get(concept_col, str(it_id))
        txt = str(row.get(text_col, ""))
        lang = str(row.get(lang_col, "en"))
        split = str(row.get(split_col, "all"))

        # True high-D cosine nearest neighbors (excluding self)
        ranked = np.argsort(-cos_matrix[i])
        nn_list = []
        for r_idx in ranked:
            if r_idx == i:
                continue
            r_row = df.iloc[r_idx]
            nn_list.append({
                "item_id": str(r_row.get(item_id_col, "")),
                "concept": str(r_row.get(concept_col, "")),
                "variant": str(r_row.get(variant_col, "")),
                "cosine": round(float(cos_matrix[i, r_idx]), 3),
                "text": str(r_row.get(text_col, ""))[:60] + "...",
            })
            if len(nn_list) >= k_neighbors:
                break

        # Matched triad counterparts
        partners = []
        if it_id in item_map:
            for p_var, p_idx in item_map[it_id].items():
                if p_idx == i:
                    continue
                true_cos = float(cos_matrix[i, p_idx])
                low_dist = float(np.linalg.norm(coordinates[i] - coordinates[p_idx]))
                partners.append({
                    "variant": p_var,
                    "target_concept": str(df.iloc[p_idx].get(concept_col, "")),
                    "high_dim_cosine": round(true_cos, 3),
                    "low_dim_euclidean": round(low_dist, 3),
                })

        points.append({
            "point_idx": i,
            "item_id": str(it_id),
            "concept": str(concept_lbl),
            "variant": str(var),
            "lang": lang,
            "split": split,
            "text": txt,
            "x": round(float(coordinates[i, 0]), 4),
            "y": round(float(coordinates[i, 1]), 4),
            "nearest_neighbors": nn_list,
            "matched_partners": partners,
        })

    return points


def generate_map_html(
    model_name: str,
    dataset_name: str,
    layer: int,
    projection_artifact: ProjectionArtifact,
    points_data: list[dict[str, Any]],
    *,
    depth_pct: float | None = None,
    title: str = "WeightLiftAI Representation Map",
) -> str:
    """Generate self-contained publication-grade interactive HTML representation map."""
    dist = projection_artifact.distortion_metrics or {}
    t_val = dist.get("trustworthiness", "N/A")
    c_val = dist.get("continuity", "N/A")
    knn_val = dist.get("knn_overlap", "N/A")
    warning = dist.get("distortion_warning", False)
    algo_name = projection_artifact.algorithm.upper()
    exp_var = projection_artifact.explained_variance_ratio

    points_json = json.dumps(points_data)
    dist_json = json.dumps(dist)

    var_stats = exp_var and len(exp_var) >= 2
    pc1_str = f"{exp_var[0] * 100:.1f}%" if var_stats else "N/A"
    pc2_str = f"{exp_var[1] * 100:.1f}%" if var_stats else "N/A"
    pc_total_str = f"{(exp_var[0] + exp_var[1]) * 100:.1f}%" if var_stats else "N/A"

    depth_label = f" (Depth: {depth_pct:.1f}%)" if depth_pct is not None else ""

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg-page: #090d16;
    --bg-panel: #111827;
    --bg-card: #1f2937;
    --border: #374151;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --accent-teal: #06b6d4;
    --accent-purple: #a855f7;
    --accent-amber: #f59e0b;
    --alert-bg: #451a03;
    --alert-border: #b45309;
    --alert-text: #fef3c7;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-page);
    color: var(--text-primary);
    font-family: var(--font);
    line-height: 1.5;
    padding: 24px;
  }}
  .container {{
    max-width: 1400px;
    margin: 0 auto;
  }}
  header {{
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
  }}
  h1 {{
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
  }}
  .subtitle {{
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 4px;
  }}
  .guardrail-banner {{
    background: var(--alert-bg);
    border: 1px solid var(--alert-border);
    color: var(--alert-text);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 13px;
  }}
  .guardrail-icon {{
    font-size: 18px;
    line-height: 1;
  }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .metric-card {{
    background: var(--bg-panel);
    border: 1px solid var(--border);
    padding: 14px;
    border-radius: 8px;
  }}
  .metric-label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }}
  .metric-val {{
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
    color: #ffffff;
  }}
  .metric-note {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }}
  .main-layout {{
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 20px;
  }}
  @media (max-width: 1024px) {{
    .main-layout {{ grid-template-columns: 1fr; }}
  }}
  .map-wrapper {{
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    position: relative;
  }}
  .controls-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .filters {{
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 13px;
  }}
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}
  .dot-clinical {{ background-color: var(--accent-teal); }}
  .dot-lay {{ background-color: var(--accent-purple); }}
  .dot-cultural {{ background-color: var(--accent-amber); }}
  svg.map-canvas {{
    width: 100%;
    height: 620px;
    background: #0d121f;
    border-radius: 6px;
    cursor: crosshair;
  }}
  .point {{
    transition: r 0.15s ease, stroke-width 0.15s ease;
    cursor: pointer;
  }}
  .point:hover {{
    r: 8;
    stroke: #ffffff;
    stroke-width: 2px;
  }}
  .point.selected {{
    r: 9;
    stroke: #ffffff;
    stroke-width: 2.5px;
    filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.8));
  }}
  .triad-line {{
    stroke: rgba(255, 255, 255, 0.15);
    stroke-width: 1;
    stroke-dasharray: 2, 2;
    pointer-events: none;
  }}
  .triad-line.active {{
    stroke: rgba(255, 255, 255, 0.8);
    stroke-width: 2;
    stroke-dasharray: none;
  }}
  .inspector-panel {{
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    max-height: 720px;
    overflow-y: auto;
  }}
  .inspector-title {{
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .empty-state {{
    color: var(--text-secondary);
    font-size: 13px;
    text-align: center;
    padding: 40px 10px;
  }}
  .inspect-section {{
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }}
  .inspect-label {{
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}
  .inspect-text {{
    font-size: 14px;
    color: #ffffff;
    font-weight: 500;
  }}
  .inspect-quote {{
    background: var(--bg-card);
    border-left: 3px solid var(--accent-teal);
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-top: 6px;
    font-style: italic;
  }}
  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 6px;
  }}
  table.data-table th, table.data-table td {{
    padding: 6px 8px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }}
  table.data-table th {{
    color: var(--text-secondary);
    font-weight: 600;
  }}
  .sim-tag {{
    font-family: monospace;
    font-weight: 700;
    color: var(--accent-teal);
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="subtitle">
      Model: <strong>{html.escape(model_name)}</strong> | Layer: <strong>{layer}{depth_label}</strong> | Dataset: <strong>{html.escape(dataset_name)}</strong>
    </div>
  </header>

  <div class="guardrail-banner">
    <div class="guardrail-icon">⚠️</div>
    <div>
      <strong>Scientific Cartography Guardrail:</strong> Projections are exploratory visuals, not metric geometry.
      Screen-space distances between clusters are non-linear approximations.
      Always verify semantic proximity using the original high-dimensional cosine neighbors reported in the Inspector.
    </div>
  </div>

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Projection Algorithm</div>
      <div class="metric-val">{algo_name}</div>
      <div class="metric-note">Random Seed: {projection_artifact.random_seed}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Trustworthiness T(k)</div>
      <div class="metric-val">{t_val}</div>
      <div class="metric-note">Penalizes false 2D neighbors (Target: &ge; 0.85)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Continuity C(k)</div>
      <div class="metric-val">{c_val}</div>
      <div class="metric-note">Penalizes false 2D tears (Target: &ge; 0.85)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">k-NN Overlap R@k</div>
      <div class="metric-val">{knn_val}</div>
      <div class="metric-note">Neighborhood preservation (k={dist.get("k_neighbors", 5)})</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">PCA Variance (PC1+PC2)</div>
      <div class="metric-val">{pc_total_str}</div>
      <div class="metric-note">PC1: {pc1_str} | PC2: {pc2_str}</div>
    </div>
  </div>

  <div class="main-layout">
    <div class="map-wrapper">
      <div class="controls-bar">
        <div class="filters">
          <label class="badge">
            <input type="checkbox" id="showTriads" checked onchange="toggleTriads()"/>
            <span>Connect Matched Triads</span>
          </label>
          <label class="badge">
            <input type="checkbox" id="filterClinical" checked onchange="updateVisibility()"/>
            <span class="dot dot-clinical"></span>
            <span>Clinical</span>
          </label>
          <label class="badge">
            <input type="checkbox" id="filterLay" checked onchange="updateVisibility()"/>
            <span class="dot dot-lay"></span>
            <span>Lay</span>
          </label>
          <label class="badge">
            <input type="checkbox" id="filterCultural" checked onchange="updateVisibility()"/>
            <span class="dot dot-cultural"></span>
            <span>Cultural Idiom</span>
          </label>
        </div>
        <button onclick="resetSelection()" style="background:#374151;color:#fff;border:none;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:12px;">Reset Selection</button>
      </div>

      <svg id="scatterCanvas" class="map-canvas" viewBox="-11 -11 22 22">
        <g id="gridGroup"></g>
        <g id="triadLinesGroup"></g>
        <g id="pointsGroup"></g>
      </svg>
    </div>

    <div class="inspector-panel" id="inspectorPanel">
      <div class="inspector-title">
        <span>Point Inspector</span>
        <span id="inspectBadge" style="font-size:11px;font-weight:normal;color:var(--text-secondary);">Click any point</span>
      </div>

      <div id="emptyInspector" class="empty-state">
        Select a point on the map to inspect its concept, textual probe, true high-dimensional cosine nearest neighbors, and matched triad alignment.
      </div>

      <div id="inspectorContent" style="display:none;">
        <div class="inspect-section">
          <div class="inspect-label">Concept / Ontology Ref</div>
          <div class="inspect-text" id="inspectConcept">--</div>
          <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;" id="inspectId">--</div>
        </div>

        <div class="inspect-section">
          <div class="inspect-label">Variant & Speech Register</div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
            <span class="dot" id="inspectDot"></span>
            <span class="inspect-text" id="inspectVariant">--</span>
            <span style="font-size:11px;color:var(--text-secondary);" id="inspectLang"></span>
          </div>
          <div class="inspect-quote" id="inspectText">--</div>
        </div>

        <div class="inspect-section">
          <div class="inspect-label">Matched Triad Alignment (True vs 2D)</div>
          <table class="data-table">
            <thead>
              <tr><th>Variant</th><th>High-D Cosine</th><th>2D Dist</th></tr>
            </thead>
            <tbody id="inspectTriadTable"></tbody>
          </table>
        </div>

        <div class="inspect-section">
          <div class="inspect-label">True High-D Nearest Neighbors (k=5)</div>
          <table class="data-table">
            <thead>
              <tr><th>Neighbor Concept</th><th>Variant</th><th>Cosine</th></tr>
            </thead>
            <tbody id="inspectNnTable"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const points = {points_json};
  const distMetrics = {dist_json};
  let selectedIdx = null;

  const colorMap = {{
    'canonical_clinical': 'var(--accent-teal)',
    'plain_lay': 'var(--accent-purple)',
    'cultural_idiom': 'var(--accent-amber)'
  }};

  function initMap() {{
    const linesGroup = document.getElementById('triadLinesGroup');
    const ptsGroup = document.getElementById('pointsGroup');
    const gridGroup = document.getElementById('gridGroup');

    // Draw subtle grid axes
    gridGroup.innerHTML = `
      <line x1="-10" y1="0" x2="10" y2="0" stroke="rgba(255,255,255,0.08)" stroke-width="0.1"/>
      <line x1="0" y1="-10" x2="0" y2="10" stroke="rgba(255,255,255,0.08)" stroke-width="0.1"/>
      <rect x="-10" y="-10" width="20" height="20" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="0.1"/>
    `;

    // Group points by item_id to draw triad lines
    const itemMap = {{}};
    points.forEach((p, idx) => {{
      if (!itemMap[p.item_id]) itemMap[p.item_id] = [];
      itemMap[p.item_id].push(idx);
    }});

    // Create triad lines
    for (const it in itemMap) {{
      const idxs = itemMap[it];
      if (idxs.length >= 2) {{
        for (let i = 0; i < idxs.length; i++) {{
          for (let j = i + 1; j < idxs.length; j++) {{
            const p1 = points[idxs[i]];
            const p2 = points[idxs[j]];
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', p1.x);
            line.setAttribute('y1', p1.y);
            line.setAttribute('x2', p2.x);
            line.setAttribute('y2', p2.y);
            line.setAttribute('class', 'triad-line');
            line.setAttribute('id', `triad_${{idxs[i]}}_${{idxs[j]}}`);
            line.setAttribute('data-item', it);
            linesGroup.appendChild(line);
          }}
        }}
      }}
    }}

    // Render points
    points.forEach((p, idx) => {{
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', p.x);
      circle.setAttribute('cy', p.y);
      circle.setAttribute('r', '0.35');
      circle.setAttribute('fill', colorMap[p.variant] || '#ffffff');
      circle.setAttribute('class', `point pt-${{p.variant}}`);
      circle.setAttribute('id', `pt_${{idx}}`);
      circle.setAttribute('data-idx', idx);

      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${{p.concept}} (${{p.variant}})\n${{p.text}}`;
      circle.appendChild(title);

      circle.addEventListener('click', () => selectPoint(idx));
      ptsGroup.appendChild(circle);
    }});
  }}

  function selectPoint(idx) {{
    selectedIdx = idx;
    const p = points[idx];

    // Highlight selected circle
    document.querySelectorAll('.point').forEach(el => {{
      el.classList.remove('selected');
      el.setAttribute('r', '0.35');
    }});
    const activeCircle = document.getElementById(`pt_${{idx}}`);
    if (activeCircle) {{
      activeCircle.classList.add('selected');
      activeCircle.setAttribute('r', '0.55');
    }}

    // Highlight active triad lines
    document.querySelectorAll('.triad-line').forEach(l => {{
      if (l.getAttribute('data-item') === p.item_id) {{
        l.classList.add('active');
      }} else {{
        l.classList.remove('active');
      }}
    }});

    // Populate inspector
    document.getElementById('emptyInspector').style.display = 'none';
    document.getElementById('inspectorContent').style.display = 'block';

    document.getElementById('inspectConcept').textContent = p.concept;
    document.getElementById('inspectId').textContent = `ID: ${{p.item_id}} | Split: ${{p.split}}`;
    document.getElementById('inspectVariant').textContent = p.variant;
    document.getElementById('inspectLang').textContent = `[${{p.lang.toUpperCase()}}]`;
    document.getElementById('inspectText').textContent = `"${{p.text}}"`;

    const dot = document.getElementById('inspectDot');
    dot.className = 'dot ' + (p.variant === 'canonical_clinical' ? 'dot-clinical' : p.variant === 'plain_lay' ? 'dot-lay' : 'dot-cultural');

    // Triad partners
    const triadTbody = document.getElementById('inspectTriadTable');
    triadTbody.innerHTML = '';
    if (p.matched_partners && p.matched_partners.length > 0) {{
      p.matched_partners.forEach(mp => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${{mp.variant}}</strong></td>
          <td class="sim-tag">${{mp.high_dim_cosine.toFixed(3)}}</td>
          <td>${{mp.low_dim_euclidean.toFixed(2)}}</td>
        `;
        triadTbody.appendChild(tr);
      }});
    }} else {{
      triadTbody.innerHTML = '<tr><td colspan="3" style="color:var(--text-secondary);">No matched partners found.</td></tr>';
    }}

    // High-D Nearest Neighbors
    const nnTbody = document.getElementById('inspectNnTable');
    nnTbody.innerHTML = '';
    p.nearest_neighbors.forEach(nn => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{nn.concept}}</td>
        <td><span style="font-size:10px;opacity:0.8;">${{nn.variant}}</span></td>
        <td class="sim-tag">${{nn.cosine.toFixed(3)}}</td>
      `;
      nnTbody.appendChild(tr);
    }});
  }}

  function resetSelection() {{
    selectedIdx = null;
    document.querySelectorAll('.point').forEach(el => {{
      el.classList.remove('selected');
      el.setAttribute('r', '0.35');
    }});
    document.querySelectorAll('.triad-line').forEach(l => l.classList.remove('active'));
    document.getElementById('emptyInspector').style.display = 'block';
    document.getElementById('inspectorContent').style.display = 'none';
  }}

  function toggleTriads() {{
    const show = document.getElementById('showTriads').checked;
    document.getElementById('triadLinesGroup').style.display = show ? 'block' : 'none';
  }}

  function updateVisibility() {{
    const showClin = document.getElementById('filterClinical').checked;
    const showLay = document.getElementById('filterLay').checked;
    const showCult = document.getElementById('filterCultural').checked;

    document.querySelectorAll('.pt-canonical_clinical').forEach(el => el.style.display = showClin ? 'block' : 'none');
    document.querySelectorAll('.pt-plain_lay').forEach(el => el.style.display = showLay ? 'block' : 'none');
    document.querySelectorAll('.pt-cultural_idiom').forEach(el => el.style.display = showCult ? 'block' : 'none');
  }}

  window.addEventListener('DOMContentLoaded', initMap);
</script>
</body>
</html>
"""
    return html_page
