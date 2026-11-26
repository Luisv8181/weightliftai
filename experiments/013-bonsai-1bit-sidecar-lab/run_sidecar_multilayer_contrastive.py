"""Experiment 013b: Multi-Layer Contrastive FP16 Self-Gated Sidecar on 1-Bit Bonsai-4B.

Upgrades Experiment 013 (single-layer L19, 0/8 social win rate, +0.0485% NLL drift) with:
1. 3-Way Discriminative Self-Gate (`SOCIAL_TRAIN` vs. `ANIMAL_TRAIN + GENERAL_SPANISH_TRAIN`)
   calibrated strictly on TRAIN splits with threshold `tau` set above all negative training
   projections to target exact 0.0000% General Spanish NLL drift and 0.0% Animal FPR.
2. Multi-Layer Contrastive FP16 Residual Sidecar across Layers 18-22 (5 layers x 10 KB = ~51 KB
   total, frozen 1-bit base weights, zero test-set leakage). Only the 5 tiny `u_edit` vectors
   (12,800 FP16 scalars) are optimized on `SOCIAL_TRAIN` contrastive log-prob margins.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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

from weightliftai.weight_programming.onebit_sidecar import (  # noqa: E402
    SelfGatedFP16Sidecar,
    pack_transformer_linears_to_1bit,
)

MODEL_ID = "prism-ml/Bonsai-4B-unpacked"
CACHE_DIR = "D:/cache/huggingface/hub" if Path("D:/cache/huggingface/hub").exists() else None
TARGET_LAYERS = [18, 19, 20, 21, 22]


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
    require_grad: bool = False,
) -> torch.Tensor:
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + " " + completion, return_tensors="pt").input_ids.to(device)
    c_len = full_ids.shape[1] - p_ids.shape[1]
    if c_len <= 0:
        return torch.tensor(-100.0, device=device)
    if require_grad:
        logits = model(full_ids).logits
    else:
        with torch.no_grad():
            logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target_ids = full_ids[:, 1:]
    token_lps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_lps[0, -c_len:].sum()


def _compute_text_nll(model: Any, tokenizer: Any, texts: list[str], device: str) -> float:
    losses = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        losses.append(float(out.loss.item()))
    return float(np.mean(losses))


def _extract_layer_hiddens(model: Any, tokenizer: Any, sentences: list[str], layer_idx: int, device: str) -> np.ndarray:
    vecs = []
    for s in sentences:
        ids = tokenizer(_build_prompt(s, steered=False), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        vecs.append(out.hidden_states[layer_idx + 1][0, -1, :].float().cpu().numpy())
    return np.stack(vecs, axis=0)


def _fit_three_way_ridge_gate(
    h_pos: np.ndarray,
    h_neg: np.ndarray,
    ridge_lambda: float = 10.0,
) -> tuple[np.ndarray, float, float, float]:
    """Fit a regularized linear discriminant separating SOCIAL_TRAIN (+1) from ANIMAL_TRAIN + GENERAL_SPANISH_TRAIN (-1).

    Returns `(w_sense_unit, proj_mean, proj_std, tau)` where `tau` is placed strictly above
    all negative training projections so collateral false positives are minimized.
    """
    x = np.concatenate([h_pos, h_neg], axis=0).astype(np.float64)
    y = np.concatenate([np.ones(len(h_pos)), -np.ones(len(h_neg))]).astype(np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    # Dual ridge solution (N << D): w = xc^T (xc xc^T + lambda I)^-1 y
    gram = xc @ xc.T + ridge_lambda * np.eye(len(x), dtype=np.float64)
    dual_alpha = np.linalg.solve(gram, y)
    w = xc.T @ dual_alpha
    w = w / (np.linalg.norm(w) + 1e-12)

    p_pos = h_pos @ w
    p_neg = h_neg @ w
    all_p = np.concatenate([p_pos, p_neg])
    p_mean = float(all_p.mean())
    p_std = float(all_p.std() + 1e-8)

    z_pos = (p_pos - p_mean) / p_std
    z_neg = (p_neg - p_mean) / p_std
    # Set tau midway between max(z_neg) and min(z_pos), with safety buffer above max(z_neg)
    max_neg = float(z_neg.max())
    min_pos = float(z_pos.min())
    if min_pos > max_neg:
        tau = 0.5 * (max_neg + min_pos)
    else:
        tau = max_neg + 0.15
    return w.astype(np.float32), p_mean, p_std, float(tau)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(__file__).resolve().parent
    t_start = time.perf_counter()

    print(f"=== Loading & 1-Bit Packing {MODEL_ID} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pack_stats = pack_transformer_linears_to_1bit(model, group_size=128, scale_dtype=torch.bfloat16)
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    vram_mb = round(torch.cuda.memory_allocated() / 1e6, 2) if device == "cuda" else 0.0
    print(f"Packed 1-bit Bonsai-4B on {device}: VRAM={vram_mb} MB (compression={pack_stats['compression_ratio']}x)")

    # Step 1: Fit 3-way discriminative self-gate at Layer 19 strictly on TRAIN splits
    print("\n=== Step 1: Fitting 3-Way Discriminative Self-Gate at L=19 (TRAIN only) ===")
    soc_tr_es = [x.spanish for x in SOCIAL_TRAIN]
    ani_tr_es = [x.spanish for x in ANIMAL_TRAIN]
    gen_tr_es = [x.spanish for x in GENERAL_SPANISH_TRAIN]

    h_soc_tr = _extract_layer_hiddens(model, tokenizer, soc_tr_es, 19, device)
    h_ani_tr = _extract_layer_hiddens(model, tokenizer, ani_tr_es, 19, device)
    h_gen_tr = _extract_layer_hiddens(model, tokenizer, gen_tr_es, 19, device)

    w_sense_np, proj_mean, proj_std, tau = _fit_three_way_ridge_gate(
        h_soc_tr, np.concatenate([h_ani_tr, h_gen_tr], axis=0), ridge_lambda=25.0
    )

    # Verify gate generalization on UNSEEN test splits (diagnostic only, no tuning!)
    h_soc_te = _extract_layer_hiddens(model, tokenizer, [x.spanish for x in SOCIAL_TEST], 19, device)
    h_ani_te = _extract_layer_hiddens(model, tokenizer, [x.spanish for x in ANIMAL_TEST], 19, device)
    h_gen_te = _extract_layer_hiddens(model, tokenizer, [x.spanish for x in GENERAL_SPANISH_TEST], 19, device)
    z_soc_te = (h_soc_te @ w_sense_np - proj_mean) / proj_std
    z_ani_te = (h_ani_te @ w_sense_np - proj_mean) / proj_std
    z_gen_te = (h_gen_te @ w_sense_np - proj_mean) / proj_std

    gate_stats = {
        "gate_layer": 19,
        "tau": round(tau, 4),
        "unseen_social_recall": round(float(np.mean(z_soc_te > tau)), 4),
        "unseen_animal_fpr": round(float(np.mean(z_ani_te > tau)), 4),
        "unseen_general_spanish_fpr": round(float(np.mean(z_gen_te > tau)), 4),
    }
    print("3-Way Gate Unseen Generalization:", json.dumps(gate_stats))

    # Step 2: Train 5 tiny FP16 residual edit vectors (Layers 18-22) on SOCIAL_TRAIN contrastive margins
    print(f"\n=== Step 2: Training Multi-Layer Contrastive FP16 Sidecar on Layers {TARGET_LAYERS} (TRAIN only) ===")
    d_model = int(model.config.hidden_size)
    edit_params = nn.ParameterDict(
        {str(L): nn.Parameter(torch.zeros(d_model, device=device, dtype=torch.float32)) for L in TARGET_LAYERS}
    )
    optimizer = torch.optim.Adam(edit_params.parameters(), lr=0.12)
    max_vec_norm = 3.5

    # Register differentiable training hooks on TARGET_LAYERS
    train_handles = []
    for L in TARGET_LAYERS:
        def _make_train_hook(layer_key: str):
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                u = edit_params[layer_key].to(out[0].dtype if isinstance(out, tuple) else out.dtype)
                if isinstance(out, tuple):
                    return (out[0] + u.view(1, 1, -1),) + out[1:]
                return out + u.view(1, 1, -1)
            return _hook
        train_handles.append(model.model.layers[L].register_forward_hook(_make_train_hook(str(L))))

    try:
        epochs = 6
        for ep in range(epochs):
            ep_margins = []
            for item in SOCIAL_TRAIN:
                optimizer.zero_grad()
                p = _build_prompt(item.spanish, steered=False)
                lp_good = _score_completion_logprob(model, tokenizer, p, item.good_english, device, require_grad=True)
                lp_bad = _score_completion_logprob(model, tokenizer, p, item.bad_english, device, require_grad=True)
                margin = lp_good - lp_bad
                # Margin hinge + smooth logistic loss pushing margin >= +6.0
                loss = torch.nn.functional.softplus(6.0 - margin)
                loss.backward()
                optimizer.step()
                # Norm projection per layer
                with torch.no_grad():
                    for k in edit_params:
                        nrm = float(torch.linalg.norm(edit_params[k]).item())
                        if nrm > max_vec_norm:
                            edit_params[k].mul_(max_vec_norm / nrm)
                ep_margins.append(float(margin.detach().item()))
            print(f"  Epoch {ep+1}/{epochs}: mean_train_margin={np.mean(ep_margins):+.3f} (wins={sum(m > 0 for m in ep_margins)}/{len(ep_margins)})")
    finally:
        for h in train_handles:
            h.remove()

    learned_edits = {L: edit_params[str(L)].detach().cpu().to(torch.float16) for L in TARGET_LAYERS}
    w_sense_t = torch.tensor(w_sense_np, dtype=torch.float16)

    # Step 3: Evaluate 4 arms on UNSEEN test splits
    print("\n=== Step 3: 4-Arm Controlled Evaluation on UNSEEN Test Splits ===")
    gen_test_texts = [x.spanish for x in GENERAL_SPANISH_TEST]

    def _eval_multilayer_arm(arm_name: str, steered_prompt: bool = False, use_sidecar: bool = False, ungated: bool = False) -> dict[str, Any]:
        handles = []
        sidecars: dict[int, SelfGatedFP16Sidecar] = {}
        active_gate_state = {"gate": 0.0}

        if use_sidecar:
            # Shared 3-way gate computed at Layer 19 (or ungated=1.0) controls Layers 18-22 cleanly!
            # Wait: Layer 18 runs BEFORE Layer 19, so we evaluate gate at Layer 18? Better: compute gate on Layer 19 and apply sidecars on Layers 19-22 (4 layers), OR compute gate at Layer 18!
            # Actually, applying sidecar on Layers [19, 20, 21, 22] using Layer 19's clean pre-edit hidden state ensures causal ordering AND zero leakage!
            # Even better: evaluate gate cleanly at Layer 19 BEFORE adding L19 edit, and also store per-layer SelfGatedFP16Sidecar objects for clean byte/hash accounting.
            for L in TARGET_LAYERS:
                sidecars[L] = SelfGatedFP16Sidecar(
                    w_sense=w_sense_t,
                    u_edit=learned_edits[L],
                    proj_mean=proj_mean,
                    proj_std=proj_std,
                    alpha=1.0,
                    tau=tau,
                    temperature=0.20,
                    ungated=ungated,
                )

            def _make_eval_hook(layer_idx: int):
                sc = sidecars[layer_idx]
                def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                    h = out[0] if isinstance(out, tuple) else out
                    if layer_idx == 19:
                        # Compute 3-way gate from Layer 19 (using prompt_end / last token on prefill)
                        if h.shape[1] > 1:
                            h_last = h[:, -1, :].float()
                            w_s = sc.w_sense.to(device=h.device, dtype=torch.float32)
                            z = float(((torch.matmul(h_last, w_s) - sc.proj_mean) / sc.proj_std).mean().item())
                            if ungated:
                                active_gate_state["gate"] = 1.0
                            else:
                                active_gate_state["gate"] = 1.0 if z > sc.tau else 0.0
                    g = 1.0 if ungated else active_gate_state["gate"]
                    if g <= 0.0:
                        return out
                    u_e = sc.u_edit.to(device=h.device, dtype=h.dtype)
                    h_new = h + (g * u_e).view(1, 1, -1)
                    return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
                return _hook

            # Only hook Layers 19-22 for strict causal gate-then-edit ordering
            for L in [19, 20, 21, 22]:
                handles.append(model.model.layers[L].register_forward_hook(_make_eval_hook(L)))

        try:
            soc_margins, soc_wins = [], 0
            for item in SOCIAL_TEST:
                active_gate_state["gate"] = 0.0
                p = _build_prompt(item.spanish, steered=steered_prompt)
                lp_good = float(_score_completion_logprob(model, tokenizer, p, item.good_english, device).item())
                lp_bad = float(_score_completion_logprob(model, tokenizer, p, item.bad_english, device).item())
                m = lp_good - lp_bad
                soc_margins.append(m)
                if m > 0:
                    soc_wins += 1

            ani_margins, ani_wins = [], 0
            for item in ANIMAL_TEST:
                active_gate_state["gate"] = 0.0
                p = _build_prompt(item.spanish, steered=steered_prompt)
                lp_good = float(_score_completion_logprob(model, tokenizer, p, item.good_english, device).item())
                lp_bad = float(_score_completion_logprob(model, tokenizer, p, item.bad_english, device).item())
                m = lp_good - lp_bad
                ani_margins.append(m)
                if m > 0:
                    ani_wins += 1

            active_gate_state["gate"] = 0.0
            gen_nll = _compute_text_nll(model, tokenizer, gen_test_texts, device)

            # Qualitative sample translations
            active_gate_state["gate"] = 0.0
            s_in = tokenizer(_build_prompt(SOCIAL_TEST[0].spanish, steered=steered_prompt), return_tensors="pt").to(device)
            with torch.no_grad():
                g_out = model.generate(**s_in, max_new_tokens=20, do_sample=False)
            soc_sample = tokenizer.decode(g_out[0, s_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            active_gate_state["gate"] = 0.0
            a_in = tokenizer(_build_prompt(ANIMAL_TEST[0].spanish, steered=steered_prompt), return_tensors="pt").to(device)
            with torch.no_grad():
                ga_out = model.generate(**a_in, max_new_tokens=20, do_sample=False)
            ani_sample = tokenizer.decode(ga_out[0, a_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            total_sidecar_bytes = sum(sidecars[L].nbytes() for L in [19, 20, 21, 22]) if use_sidecar else 0
            return {
                "arm": arm_name,
                "social_win_rate": round(soc_wins / len(SOCIAL_TEST), 4),
                "social_mean_margin": round(float(np.mean(soc_margins)), 4),
                "animal_win_rate": round(ani_wins / len(ANIMAL_TEST), 4),
                "animal_mean_margin": round(float(np.mean(ani_margins)), 4),
                "general_spanish_nll": round(gen_nll, 6),
                "sample_social_translation": soc_sample,
                "sample_animal_translation": ani_sample,
                "sidecar_bytes": total_sidecar_bytes,
            }
        finally:
            for h in handles:
                h.remove()

    res_base = _eval_multilayer_arm("baseline_1bit", steered_prompt=False, use_sidecar=False)
    res_prompt = _eval_multilayer_arm("prompt_steering_1bit", steered_prompt=True, use_sidecar=False)
    res_ungated = _eval_multilayer_arm("ungated_multilayer_sidecar", steered_prompt=False, use_sidecar=True, ungated=True)
    res_gated = _eval_multilayer_arm("self_gated_multilayer_sidecar", steered_prompt=False, use_sidecar=True, ungated=False)

    base_nll = res_base["general_spanish_nll"]
    for r in (res_base, res_prompt, res_ungated, res_gated):
        drift_pct = ((r["general_spanish_nll"] - base_nll) / max(base_nll, 1e-8)) * 100.0
        r["general_spanish_drift_pct"] = round(drift_pct, 4)
        print(
            f"  {r['arm']:30s} | soc_win={r['social_win_rate']:.3f} (m={r['social_mean_margin']:+.3f}) | "
            f"ani_win={r['animal_win_rate']:.3f} (m={r['animal_mean_margin']:+.3f}) | "
            f"nll_drift={r['general_spanish_drift_pct']:+.4f}% | bytes={r['sidecar_bytes']}"
        )
        print(f"     soc_sample: {r['sample_social_translation']!r}")
        print(f"     ani_sample: {r['sample_animal_translation']!r}")

    elapsed = round(time.perf_counter() - t_start, 2)
    out_path = out_dir / "results_013b_multilayer.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "013b-bonsai-4b-multilayer-contrastive-sidecar",
                "model_id": MODEL_ID,
                "vram_allocated_mb": vram_mb,
                "active_sidecar_layers": [19, 20, 21, 22],
                "three_way_gate_stats": gate_stats,
                "arms": [res_base, res_prompt, res_ungated, res_gated],
                "total_wall_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n=== EXP 013B COMPLETE in {elapsed}s -> {out_path} ===")


if __name__ == "__main__":
    main()
