"""Experiment 014: Direct Q1_0 GGUF 1-Bit Bonsai-27B FP16 Self-Gated Sidecar Lab.

Loads PrismML `Bonsai-27B-Q1_0.gguf` (27B parameters, 64 Qwen3.5 hybrid Gated-DeltaNet +
Full-Attention layers) directly into PyTorch CUDA in ~3.81 GB VRAM (~4.74 GB peak) with
zero FP16 unpacking, runs a 7-layer homograph ('tigre') Cartography scan, fits a 3-way
discriminative self-gate (SOCIAL_TRAIN vs. ANIMAL_TRAIN + GENERAL_SPANISH_TRAIN), and
attaches a tiny multi-layer FP16 residual sidecar (~61 KB, < 0.002% of base model VRAM)
synthesized via closed-form contrastive representation distillation (steered - unsteered).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "004-cartography-and-gated-patch"))
from homograph_data import (  # noqa: E402
    ANIMAL_TIGRE_TEST as ANIMAL_TEST,
    ANIMAL_TIGRE_TRAIN as ANIMAL_TRAIN,
    GENERAL_SPANISH_TEST,
    GENERAL_SPANISH_TRAIN,
    SOCIAL_TIGRE_TEST as SOCIAL_TEST,
    SOCIAL_TIGRE_TRAIN as SOCIAL_TRAIN,
)

from weightliftai.weight_programming.gguf_1bit_loader import (  # noqa: E402
    load_bonsai_27b_q1_0_gguf,
)
from weightliftai.weight_programming.onebit_sidecar import (  # noqa: E402
    SelfGatedFP16Sidecar,
)

GGUF_PATH = Path(
    os.environ.get("BONSAI_27B_GGUF", "C:/Users/luisv/models/bonsai/model/Bonsai-27B-Q1_0.gguf")
)
TOKENIZER_ID = os.environ.get("BONSAI_27B_TOKENIZER", "HAL-X-AI/THX-02")
CACHE_DIR = "D:/cache/huggingface/hub" if Path("D:/cache/huggingface/hub").exists() else None
SCAN_LAYERS = [20, 24, 28, 32, 36, 40, 44]


def _build_prompt(spanish_sentence: str, steered: bool = False) -> str:
    if steered:
        return (
            "Translate the following Spanish sentence into natural English. "
            "Note: In Dominican and Caribbean Spanish, 'tigre' can mean a street-smart guy, player, "
            "or hustler in social/interpersonal contexts, OR a literal tiger in zoological/animal contexts. "
            "Use the sentence context to choose the correct meaning. Output only the English translation.\n"
            f"Spanish: {spanish_sentence}\nEnglish:"
        )
    return (
        "Translate the following Spanish sentence into natural English. Output only the English translation.\n"
        f"Spanish: {spanish_sentence}\nEnglish:"
    )


def _score_completion_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: str,
) -> float:
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + " " + completion, return_tensors="pt").input_ids.to(device)
    c_len = full_ids.shape[1] - p_ids.shape[1]
    if c_len <= 0:
        return -100.0
    with torch.no_grad():
        logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target_ids = full_ids[:, 1:]
    token_lps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_lps[0, -c_len:].sum().item())


def _compute_text_nll(model: Any, tokenizer: Any, texts: list[str], device: str) -> float:
    losses = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        losses.append(float(out.loss.item()))
    return float(np.mean(losses))


def _extract_multilayer_hiddens(
    model: Any,
    tokenizer: Any,
    sentences: list[str],
    layer_indices: list[int],
    device: str,
    steered: bool = False,
) -> dict[int, np.ndarray]:
    buckets: dict[int, list[np.ndarray]] = {idx: [] for idx in layer_indices}
    for s in sentences:
        ids = tokenizer(_build_prompt(s, steered=steered), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for idx in layer_indices:
            buckets[idx].append(out.hidden_states[idx + 1][0, -1, :].float().cpu().numpy())
    return {idx: np.stack(vecs, axis=0) for idx, vecs in buckets.items()}


def _fit_three_way_ridge_gate(
    h_pos: np.ndarray,
    h_neg: np.ndarray,
    ridge_lambda: float = 500.0,
) -> tuple[np.ndarray, float, float, float, float]:
    """Fit a regularized linear discriminant separating SOCIAL_TRAIN (+1) from ANIMAL + GENERAL (-1)."""
    x = np.concatenate([h_pos, h_neg], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(len(h_pos)), -np.ones(len(h_neg))]).astype(np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    # Dual ridge (N << D): w = xc^T (xc xc^T + lambda I)^-1 y
    gram = xc @ xc.T + ridge_lambda * np.eye(len(x), dtype=np.float64)
    dual_alpha = np.linalg.solve(gram, y)
    w = xc.T @ dual_alpha
    w = w / (np.linalg.norm(w) + 1e-12)

    all_proj = x @ w
    proj_mean = float(all_proj.mean())
    proj_std = float(all_proj.std() + 1e-8)
    z_pos = (h_pos @ w - proj_mean) / proj_std
    z_neg = (h_neg @ w - proj_mean) / proj_std

    # Place threshold cleanly between negative train max and positive train min
    gap = float(z_pos.min() - z_neg.max())
    if gap > 0:
        tau = float(z_neg.max() + 0.50 * gap)
    else:
        tau = float(z_neg.max() + 0.25)
    cohen_d = float((z_pos.mean() - z_neg.mean()) / (np.sqrt(0.5 * (z_pos.var() + z_neg.var())) + 1e-8))
    return w.astype(np.float32), proj_mean, proj_std, tau, cohen_d


def _evaluate_condition(
    name: str,
    model: Any,
    tokenizer: Any,
    device: str,
    steered: bool = False,
    baseline_nll: float | None = None,
) -> dict[str, Any]:
    soc_margins: list[float] = []
    for item in SOCIAL_TEST:
        prompt = _build_prompt(item.spanish, steered=steered)
        lp_good = _score_completion_logprob(model, tokenizer, prompt, item.good_english, device)
        lp_bad = _score_completion_logprob(model, tokenizer, prompt, item.bad_english, device)
        soc_margins.append(lp_good - lp_bad)

    ani_margins: list[float] = []
    for item in ANIMAL_TEST:
        prompt = _build_prompt(item.spanish, steered=steered)
        lp_good = _score_completion_logprob(model, tokenizer, prompt, item.good_english, device)
        lp_bad = _score_completion_logprob(model, tokenizer, prompt, item.bad_english, device)
        ani_margins.append(lp_good - lp_bad)

    nll = _compute_text_nll(model, tokenizer, [x.spanish for x in GENERAL_SPANISH_TEST], device)
    drift_pct = 0.0 if baseline_nll is None else ((nll - baseline_nll) / baseline_nll) * 100.0

    soc_win = float(np.mean([m > 0 for m in soc_margins]))
    ani_win = float(np.mean([m > 0 for m in ani_margins]))
    res = {
        "condition": name,
        "social_win_rate": round(soc_win, 4),
        "social_mean_margin": round(float(np.mean(soc_margins)), 4),
        "animal_win_rate": round(ani_win, 4),
        "animal_mean_margin": round(float(np.mean(ani_margins)), 4),
        "general_spanish_nll": round(nll, 6),
        "general_nll_drift_pct": round(drift_pct, 4),
    }
    print(
        f"[{name}] soc_win={soc_win:.3f} (m={res['social_mean_margin']:+.3f}) | "
        f"ani_win={ani_win:.3f} (m={res['animal_mean_margin']:+.3f}) | "
        f"nll={nll:.5f} (drift={drift_pct:+.4f}%)",
        flush=True,
    )
    return res


def _make_layer_hook(sidecar: SelfGatedFP16Sidecar):
    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (sidecar.apply_to_hidden(output[0]),) + output[1:]
        return sidecar.apply_to_hidden(output)
    return _hook


def main() -> None:
    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Experiment 014: Direct Q1_0 GGUF Bonsai-27B 1-Bit Sidecar Lab ({device}) ===", flush=True)

    t0 = time.perf_counter()
    model, load_stats = load_bonsai_27b_q1_0_gguf(GGUF_PATH, device=device, scale_dtype=torch.bfloat16)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"Loaded Bonsai-27B in {load_s}s | stats={load_stats}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=CACHE_DIR, local_files_only=True)

    # 1. Single-pass Cartography scan on TRAIN splits (zero test leakage); also cache neighbors
    extract_layers = sorted(set(SCAN_LAYERS + [l - 1 for l in SCAN_LAYERS] + [l + 1 for l in SCAN_LAYERS]))
    print(f"\n[1/4] 7-Layer Cartography Scan on TRAIN splits (layers={SCAN_LAYERS})...", flush=True)
    soc_train_sents = [x.spanish for x in SOCIAL_TRAIN]
    ani_train_sents = [x.spanish for x in ANIMAL_TRAIN]

    h_soc_train = _extract_multilayer_hiddens(model, tokenizer, soc_train_sents, extract_layers, device, steered=False)
    h_ani_train = _extract_multilayer_hiddens(model, tokenizer, ani_train_sents, extract_layers, device, steered=False)
    h_gen_train = _extract_multilayer_hiddens(
        model, tokenizer, [x.spanish for x in GENERAL_SPANISH_TRAIN], extract_layers, device, steered=False
    )

    cartography: list[dict[str, Any]] = []
    gate_params: dict[int, tuple[np.ndarray, float, float, float, float]] = {}
    for idx in SCAN_LAYERS:
        h_neg = np.concatenate([h_ani_train[idx], h_gen_train[idx]], axis=0)
        w, pmean, pstd, tau, cohen_d = _fit_three_way_ridge_gate(h_soc_train[idx], h_neg)
        gate_params[idx] = (w, pmean, pstd, tau, cohen_d)
        cartography.append({
            "layer": idx,
            "cohen_d": round(cohen_d, 3),
            "tau": round(tau, 4),
            "hidden_norm_mean": round(float(np.linalg.norm(h_soc_train[idx], axis=-1).mean()), 2),
        })
        print(f"  L{idx:02d}: cohen_d={cohen_d:.3f} | tau={tau:+.4f} | norm={cartography[-1]['hidden_norm_mean']}", flush=True)

    best_entry = max(cartography, key=lambda r: r["cohen_d"])
    best_layer = int(best_entry["layer"])
    target_layers = [best_layer - 1, best_layer, best_layer + 1]
    print(f"Selected peak Cartography layer: L{best_layer} -> 3-layer sidecar bank {target_layers}", flush=True)

    # 2. Synthesize closed-form contrastive FP16 sidecar vectors (steered - unsteered) on SOCIAL_TRAIN
    print(f"\n[2/4] Extracting contrastive representation deltas on SOCIAL_TRAIN for {target_layers}...", flush=True)
    h_soc_steered_3 = _extract_multilayer_hiddens(model, tokenizer, soc_train_sents, target_layers, device, steered=True)

    sidecars: dict[int, SelfGatedFP16Sidecar] = {}
    total_sidecar_bytes = 0
    for idx in target_layers:
        h_neg = np.concatenate([h_ani_train[idx], h_gen_train[idx]], axis=0)
        w, pmean, pstd, tau, _ = _fit_three_way_ridge_gate(h_soc_train[idx], h_neg)
        delta = (h_soc_steered_3[idx] - h_soc_train[idx]).mean(axis=0).astype(np.float32)
        sc = SelfGatedFP16Sidecar(
            w_sense=torch.from_numpy(w),
            u_edit=torch.from_numpy(delta),
            proj_mean=pmean,
            proj_std=pstd,
            alpha=1.0,
            tau=tau,
            temperature=0.15,
            ungated=False,
            dtype=torch.bfloat16,
        ).to(device=device)
        sidecars[idx] = sc
        total_sidecar_bytes += sc.nbytes()
        print(f"  Sidecar L{idx:02d}: bytes={sc.nbytes()} | ||u_edit||={float(np.linalg.norm(delta)):.3f} | tau={tau:+.4f}", flush=True)

    # 3. Verify 3-Way Gate Generalization on UNSEEN test splits
    print(f"\n[3/4] Evaluating 3-way self-gate generalization on UNSEEN test splits (L{best_layer})...", flush=True)
    h_soc_test = _extract_multilayer_hiddens(model, tokenizer, [x.spanish for x in SOCIAL_TEST], [best_layer], device)[best_layer]
    h_ani_test = _extract_multilayer_hiddens(model, tokenizer, [x.spanish for x in ANIMAL_TEST], [best_layer], device)[best_layer]
    h_gen_test = _extract_multilayer_hiddens(
        model, tokenizer, [x.spanish for x in GENERAL_SPANISH_TEST], [best_layer], device
    )[best_layer]

    w_b, pm_b, ps_b, tau_b, _ = gate_params[best_layer]
    z_soc_t = (h_soc_test @ w_b - pm_b) / ps_b
    z_ani_t = (h_ani_test @ w_b - pm_b) / ps_b
    z_gen_t = (h_gen_test @ w_b - pm_b) / ps_b
    gate_eval = {
        "layer": best_layer,
        "tau": round(tau_b, 4),
        "unseen_social_recall": round(float(np.mean(z_soc_t > tau_b)), 4),
        "unseen_animal_fpr": round(float(np.mean(z_ani_t > tau_b)), 4),
        "unseen_general_fpr": round(float(np.mean(z_gen_t > tau_b)), 4),
        "z_social_mean": round(float(z_soc_t.mean()), 3),
        "z_animal_mean": round(float(z_ani_t.mean()), 3),
        "z_general_mean": round(float(z_gen_t.mean()), 3),
    }
    print(f"  Unseen Gate Eval: {gate_eval}", flush=True)

    # 4. Full 4-Condition Evaluation on Unseen Test Splits
    print("\n[4/4] Running 4-Condition Evaluation on Unseen Test Splits...", flush=True)
    res_base = _evaluate_condition("baseline_1bit_27b", model, tokenizer, device, steered=False, baseline_nll=None)
    base_nll = res_base["general_spanish_nll"]

    res_prompt = _evaluate_condition("prompt_steering_1bit_27b", model, tokenizer, device, steered=True, baseline_nll=base_nll)

    # Attach 3-layer sidecar bank (ungated)
    handles = []
    for idx, sc in sidecars.items():
        sc.ungated = True
        handles.append(model.model.layers[idx].register_forward_hook(_make_layer_hook(sc)))
    try:
        res_ungated = _evaluate_condition("ungated_fp16_sidecar_27b", model, tokenizer, device, steered=False, baseline_nll=base_nll)
        # Switch to self-gated (3-way discriminative gate)
        for sc in sidecars.values():
            sc.ungated = False
        res_gated = _evaluate_condition("self_gated_fp16_sidecar_27b", model, tokenizer, device, steered=False, baseline_nll=base_nll)
    finally:
        for h in handles:
            h.remove()

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 2) if device == "cuda" else 0.0
    elapsed_s = round(time.perf_counter() - t_start, 2)

    out_data = {
        "experiment": "014-bonsai-27b-1bit-sidecar-lab",
        "model_file": GGUF_PATH.name,
        "tokenizer_id": TOKENIZER_ID,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "load_time_s": load_s,
        "elapsed_total_s": elapsed_s,
        "load_stats": load_stats,
        "peak_vram_mb": peak_vram_mb,
        "cartography_scan": cartography,
        "best_layer": best_layer,
        "sidecar_layers": target_layers,
        "sidecar_total_bytes": total_sidecar_bytes,
        "sidecar_overhead_pct_of_1bit_model": round((total_sidecar_bytes / (load_stats["total_packed_mb"] * 1e6)) * 100.0, 6),
        "unseen_gate_eval": gate_eval,
        "results": {
            "baseline_1bit_27b": res_base,
            "prompt_steering_1bit_27b": res_prompt,
            "ungated_fp16_sidecar_27b": res_ungated,
            "self_gated_fp16_sidecar_27b": res_gated,
        },
    }

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nSaved Experiment 014 results to {out_path.name} ({elapsed_s}s total, peak VRAM {peak_vram_mb} MB).", flush=True)


if __name__ == "__main__":
    main()
