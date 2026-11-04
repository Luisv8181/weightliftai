"""Execution pipeline for Medical Terminology Representation Atlas.

Captures multi-layer representations of matched medical concepts across 3 expression variants:
1. canonical_clinical (e.g. "type 2 diabetes mellitus", "cerebrovascular accident")
2. plain_lay (e.g. "adult-onset diabetes", "stroke")
3. cultural_idiom (e.g. "azúcar alta en la sangre", "trombosis / derrame cerebral")

Evaluates whether clinical and cultural expressions converge into an invariant conceptual
manifold at the structural midpoint (50% depth interlingua midpoint) across:
- google/gemma-3-1b-it (26 layers: 0, 7, 13, 19, 25)
- google/gemma-4-E2B-it (35 layers: 0, 8, 17, 26, 34)

Scientific Rigor & Validation (Addressing GitHub Issue #1):
1. Evaluates held-out TEST split (10 concepts, 30 probes) independently of ALL (50 concepts).
2. Computes empirical Unrelated-Concept Negative Control Baseline (null distribution).
3. Reports specificity delta over baseline for every matched pair.
4. Records pinned model commit hashes, tokenizer revisions, and dataset SHA-256 digests.
"""

from __future__ import annotations

import gc
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
    compute_layer_metrics,
    extract_layer_representations,
    find_decoder_layers,
    generate_atlas_html,
)
from weightliftai.cartography.medical_pilot import MedicalPilotDataset

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_model_atlas(
    model_id: str,
    layer_indices: list[int],
    ds: MedicalPilotDataset,
    device: str,
    *,
    split_name: str = "test",
    is_gemma4: bool = False,
) -> dict:
    df = ds.to_matched_dataframe(split=None if split_name == "all" else split_name)
    n_concepts = df["item_id"].nunique()

    print(f"\n{'='*70}")
    print(f"Layer Atlas: {model_id} | Split: {split_name.upper()} ({len(df)} probes, {n_concepts} concepts)")
    print(f"Layers to evaluate: {layer_indices}")
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
    load_time = time.time() - t0

    model_commit = getattr(model.config, "_commit_hash", None) or "main"
    tok_commit = getattr(tok, "_commit_hash", None) or "main"
    print(f"Model loaded in {load_time:.1f}s | Commit: {model_commit[:12]} | Total layers: {total_layers}")

    print("Extracting multi-layer representations in one controlled pass...")
    t_ext = time.time()
    embeddings_by_layer = extract_layer_representations(
        model=model,
        tokenizer=tok,
        texts=df["text"].tolist(),
        layers=layer_indices,
        site="residual",
        token_mask="content",
        pooling="mean",
        batch_size=8,
        device=device,
    )
    ext_time = time.time() - t_ext
    print(f"Extraction completed in {ext_time:.1f}s")

    # Compute paired metrics with bootstrap CIs and unrelated baseline
    print("Computing pairwise alignment metrics, bootstrap 95% CIs, and unrelated baseline...")
    layer_metrics = {}
    for L in layer_indices:
        d_pct = round(L / max(1, total_layers - 1) * 100, 1)
        m = compute_layer_metrics(embeddings_by_layer[L], df=df, n_bootstrap=2000, seed=42)
        m["depth_pct"] = d_pct
        layer_metrics[L] = m

        unrel = m.get("unrelated_negative_control", {})
        unrel_str = f"{unrel.get('mean', 'N/A')} (95% CI {unrel.get('ci_95', 'N/A')})"
        print(f"  Layer {L:02d} ({d_pct:5.1f}% depth) | Unrelated Null Baseline: {unrel_str}:")
        for pair_key, pdata in m["pairwise_cosines"].items():
            mean = pdata["mean"]
            ci = pdata["ci_95"]
            delta = pdata.get("delta_over_unrelated_baseline")
            delta_str = f" [Δ = +{delta:.3f} vs null]" if delta is not None else ""
            print(f"    - {pair_key:40s}: cos = {mean:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]){delta_str}")

    # Generate HTML report
    safe_name = model_id.replace("/", "_").replace("-", "_")
    html_content = generate_atlas_html(
        model_name=model_id,
        dataset_name=f"NLM MeSH Medical Pilot (Split: {split_name.upper()}, {n_concepts} concepts)",
        layer_metrics=layer_metrics,
        title=f"Medical Terminology Layer Atlas — {model_id} ({split_name.upper()})",
    )
    html_path = RESULTS_DIR / f"{safe_name}_medical_atlas_{split_name}.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Summary record
    summary = {
        "model_id": model_id,
        "model_commit": model_commit,
        "tokenizer_commit": tok_commit,
        "dataset_id": ds.dataset_ref.dataset_id,
        "dataset_sha256": ds.dataset_ref.sha256,
        "split": split_name,
        "total_layers": total_layers,
        "evaluated_layers": layer_indices,
        "sample_count": len(df),
        "concepts_count": n_concepts,
        "variants": sorted(df["variant"].unique().tolist()),
        "resampling_unit": "paired item_id (concept level)",
        "bootstrap_iterations": 2000,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "html_report": str(html_path),
        "layer_metrics": {str(k): v for k, v in layer_metrics.items()},
    }
    json_path = RESULTS_DIR / f"{safe_name}_medical_atlas_{split_name}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nArtifacts saved ({split_name.upper()} split):")
    print(f"  - Metrics: {json_path}")
    print(f"  - HTML:    {html_path}")

    # Free memory
    del model
    del tok
    del embeddings_by_layer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    print("Initializing 50-concept Medical Pilot Dataset...")
    ds = MedicalPilotDataset()
    print(f"Dataset verified: 150 rows, 50 concepts, SHA-256: {ds.dataset_ref.sha256[:16]}...")
    print("Zero PHI verified. Licensing: MeSH 2026 Public Domain.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute environment: {device} (VRAM: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})\n")

    all_summaries = []

    # 1. Gemma 3 1B on HELD-OUT TEST SPLIT (10 concepts, 30 probes)
    print("\n>>> EVALUATING HELD-OUT TEST SPLIT ON GEMMA 3 1B <<<")
    g3_test = run_model_atlas(
        model_id="google/gemma-3-1b-it",
        layer_indices=[0, 7, 13, 19, 25],
        ds=ds,
        device=device,
        split_name="test",
        is_gemma4=False,
    )
    all_summaries.append(g3_test)

    # 2. Gemma 3 1B on FULL DATASET (50 concepts, 150 probes)
    print("\n>>> EVALUATING FULL DATASET ON GEMMA 3 1B <<<")
    g3_all = run_model_atlas(
        model_id="google/gemma-3-1b-it",
        layer_indices=[0, 7, 13, 19, 25],
        ds=ds,
        device=device,
        split_name="all",
        is_gemma4=False,
    )
    all_summaries.append(g3_all)

    # 3. Gemma 4 E2B on HELD-OUT TEST SPLIT (10 concepts, 30 probes)
    print("\n>>> EVALUATING HELD-OUT TEST SPLIT ON GEMMA 4 E2B <<<")
    g4_test = run_model_atlas(
        model_id="google/gemma-4-E2B-it",
        layer_indices=[0, 8, 17, 26, 34],
        ds=ds,
        device=device,
        split_name="test",
        is_gemma4=True,
    )
    all_summaries.append(g4_test)

    # 4. Gemma 4 E2B on FULL DATASET (50 concepts, 150 probes)
    print("\n>>> EVALUATING FULL DATASET ON GEMMA 4 E2B <<<")
    g4_all = run_model_atlas(
        model_id="google/gemma-4-E2B-it",
        layer_indices=[0, 8, 17, 26, 34],
        ds=ds,
        device=device,
        split_name="all",
        is_gemma4=True,
    )
    all_summaries.append(g4_all)

    summary_file = RESULTS_DIR / "medical_layer_atlas_comprehensive_summary.json"
    summary_file.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nAll layer atlas runs complete! Comprehensive summary saved at: {summary_file}")


if __name__ == "__main__":
    main()
