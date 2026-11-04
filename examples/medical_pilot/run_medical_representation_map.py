"""Execution pipeline for Medical Representation Maps.

Generates 2D dimensionality reduction maps with mandatory distortion guardrails
for the 50-concept Medical Pilot dataset across:
1. google/gemma-3-1b-it at Layer 13 (52% depth interlingua midpoint)
2. google/gemma-3-1b-it at Layer 25 (100% depth output layer)
3. google/gemma-4-E2B-it at Layer 17 (50% depth interlingua midpoint)

Computes Trustworthiness T(k), Continuity C(k), and k-NN overlap R@k.
Builds interactive point-inspection side panels and triad connection overlays.
Saves JSON projection artifacts and self-contained HTML maps in examples/medical_pilot/results/.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
)

from weightliftai.cartography.layer_atlas import (
    extract_layer_representations,
    find_decoder_layers,
)
from weightliftai.cartography.medical_pilot import MedicalPilotDataset
from weightliftai.cartography.representation_map import (
    build_point_inspection_data,
    generate_map_html,
    project_representations,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_map_pipeline(
    model_id: str,
    target_layer: int,
    df: pd.DataFrame,
    device: str,
    *,
    is_gemma4: bool = False,
    algorithm: str = "pca",
) -> dict:
    print(f"\n{'='*70}")
    print(f"Generating Representation Map: {model_id} | Layer {target_layer} ({algorithm.upper()})")
    print(f"{'='*70}\n")

    t0 = time.time()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_gemma4:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    decoder_blocks = find_decoder_layers(model)
    total_layers = len(decoder_blocks)
    depth_pct = round(target_layer / max(1, total_layers - 1) * 100, 1)

    print(f"Extracting layer {target_layer} representations ({depth_pct}% depth) for {len(df)} samples...")
    embeddings_by_layer = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=df["text"].tolist(),
        layers=[target_layer],
        site="residual",
        token_mask="content",
        pooling="mean",
        batch_size=8,
        device=device,
    )
    embeddings = embeddings_by_layer[target_layer]

    print(f"Projecting {len(embeddings)} vectors with {algorithm.upper()}...")
    item_ids = [str(x) for x in df["item_id"]]
    source_sha = hashlib.sha256(embeddings.tobytes()).hexdigest()
    proj_artifact, raw_coords, disp_coords = project_representations(
        embeddings=embeddings,
        item_ids=item_ids,
        source_sha256=source_sha,
        algorithm=algorithm,
        n_components=2,
        random_seed=42,
        k_neighbors=5,
    )

    dist = proj_artifact.distortion_metrics or {}
    print(f"Distortion Diagnostics (k=5):")
    print(f"  - Trustworthiness T(k): {dist.get('trustworthiness')}")
    print(f"  - Continuity C(k):      {dist.get('continuity')}")
    print(f"  - k-NN Inter/k:         {dist.get('knn_intersection_over_k')}")
    print(f"  - k-NN Jaccard:         {dist.get('knn_jaccard')}")
    if proj_artifact.explained_variance_ratio:
        pc1, pc2 = proj_artifact.explained_variance_ratio[:2]
        total_v = (pc1 + pc2) * 100
        print(f"  - PCA Variance:         PC1={pc1*100:.1f}%, PC2={pc2*100:.1f}% (Total: {total_v:.1f}%, Discarded: {100.0 - total_v:.1f}%)")

    points_data = build_point_inspection_data(
        df=df,
        embeddings=embeddings,
        coordinates=raw_coords,
        display_coordinates=disp_coords,
        k_neighbors=5,
    )

    safe_model = model_id.replace("/", "_").replace("-", "_")
    html_page = generate_map_html(
        model_name=model_id,
        dataset_name="NLM MeSH 50-Concept Medical Pilot (Balanced Triad)",
        layer=target_layer,
        projection_artifact=proj_artifact,
        points_data=points_data,
        depth_pct=depth_pct,
        title=f"Representation Map — {model_id} (Layer {target_layer}, {algorithm.upper()})",
    )

    json_path = RESULTS_DIR / f"{safe_model}_map_layer_{target_layer}_{algorithm}.json"
    html_path = RESULTS_DIR / f"{safe_model}_map_layer_{target_layer}_{algorithm}.html"

    json_path.write_text(proj_artifact.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(html_page, encoding="utf-8")

    print(f"Saved artifacts:")
    print(f"  - Coordinates & Metrics: {json_path}")
    print(f"  - Interactive Map HTML:  {html_path}\n")

    del model
    del tok
    del embeddings_by_layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "layer": target_layer,
        "depth_pct": depth_pct,
        "algorithm": algorithm,
        "distortion_metrics": dist,
        "html_path": str(html_path),
        "json_path": str(json_path),
    }


def main():
    print("Loading 50-concept Medical Pilot Dataset...")
    ds = MedicalPilotDataset()
    df = ds.to_dataframe()
    print(f"Dataset: {len(df)} rows across {df['item_id'].nunique()} concepts.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    runs = []

    # 1. Gemma 3 1B at Layer 13 (Interlingua Midpoint, PCA)
    runs.append(run_map_pipeline(
        model_id="google/gemma-3-1b-it",
        target_layer=13,
        df=df,
        device=device,
        is_gemma4=False,
        algorithm="pca",
    ))

    # 2. Gemma 3 1B at Layer 25 (Output Layer, PCA) - to demonstrate vocabulary divergence
    runs.append(run_map_pipeline(
        model_id="google/gemma-3-1b-it",
        target_layer=25,
        df=df,
        device=device,
        is_gemma4=False,
        algorithm="pca",
    ))

    # 3. Gemma 4 E2B at Layer 17 (Interlingua Midpoint, PCA)
    runs.append(run_map_pipeline(
        model_id="google/gemma-4-E2B-it",
        target_layer=17,
        df=df,
        device=device,
        is_gemma4=True,
        algorithm="pca",
    ))

    summary_file = RESULTS_DIR / "medical_representation_maps_summary.json"
    summary_file.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nAll maps successfully generated! Summary at: {summary_file}")


if __name__ == "__main__":
    main()
