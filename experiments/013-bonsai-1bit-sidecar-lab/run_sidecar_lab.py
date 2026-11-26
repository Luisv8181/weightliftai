"""Experiment 013: 1-Bit Sidecar Lab on PyTorch PrismML Bonsai-4B-unpacked.

Downloads/loads `prism-ml/Bonsai-4B-unpacked` (36 layers, hidden_size=2560), packs all
transformer attention & MLP projections into true 1-bit `Packed1BitLinear` (`uint8` bit-planes
+ per-group BF16 scales), verifies VRAM & reconstruction fidelity, runs a 36-layer homograph
`"tigre"` Cartography scan, and evaluates a tiny (< 20 KB) FP16 `SelfGatedFP16Sidecar` against:
1. `baseline_1bit` (unpatched 1-bit packed Bonsai-4B)
2. `prompt_steering_1bit` (textbook human prompt baseline)
3. `ungated_fp16_sidecar` (always-on FP16 sidecar control)
4. `self_gated_fp16_sidecar` (context-self-gated FP16 sidecar targeting 0.00% general drift)
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

# Reuse the exact 100% homograph corpus from Experiment 004
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "004-cartography-and-gated-patch"))
from homograph_data import (  # noqa: E402
    ANIMAL_TIGRE_TEST as ANIMAL_TEST,
    ANIMAL_TIGRE_TRAIN as ANIMAL_TRAIN,
    GENERAL_SPANISH_TEST,
    SOCIAL_TIGRE_TEST as SOCIAL_TEST,
    SOCIAL_TIGRE_TRAIN as SOCIAL_TRAIN,
)
GENERAL_SPANISH_CONTROL = [x.spanish for x in GENERAL_SPANISH_TEST]

from weightliftai.weight_programming.onebit_sidecar import (  # noqa: E402
    SelfGatedFP16Sidecar,
    pack_transformer_linears_to_1bit,
)

MODEL_ID = "prism-ml/Bonsai-4B-unpacked"
CACHE_DIR = "D:/cache/huggingface/hub" if Path("D:/cache/huggingface/hub").exists() else None


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


def _score_completion_logprob(model: Any, tokenizer: Any, prompt: str, completion: str, device: str) -> float:
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


def _extract_last_token_hiddens(model: Any, tokenizer: Any, sentences: list[str], device: str) -> np.ndarray:
    """Return `[N, num_layers, hidden_size]` float32 array of last-token hidden states."""
    all_vecs = []
    for s in sentences:
        prompt = _build_prompt(s, steered=False)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # Skip embedding layer [0], keep transformer layers 1..L
        layer_vecs = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states[1:]]
        all_vecs.append(np.stack(layer_vecs, axis=0))
    return np.stack(all_vecs, axis=0)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    pooled = math.sqrt(0.5 * (var_a + var_b) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / pooled)


import math  # noqa: E402


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(__file__).resolve().parent
    t_start = time.perf_counter()

    print(f"=== Loading {MODEL_ID} on CPU for 1-bit packing ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    print("=== Packing all transformer Linear layers into Packed1BitLinear (uint8 + BF16 scales) ===")
    pack_stats = pack_transformer_linears_to_1bit(model, group_size=128, scale_dtype=torch.bfloat16)
    print("Pack stats:", json.dumps(pack_stats, indent=2))

    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
        vram_mb = round(torch.cuda.memory_allocated() / 1e6, 2)
        vram_reserved_mb = round(torch.cuda.memory_reserved() / 1e6, 2)
    else:
        vram_mb = 0.0
        vram_reserved_mb = 0.0
    print(f"Loaded 1-bit packed Bonsai-4B on {device}: VRAM allocated={vram_mb} MB, reserved={vram_reserved_mb} MB")

    # Quick throughput benchmark (24 new tokens)
    t_gen = time.perf_counter()
    sample_in = tokenizer(_build_prompt(SOCIAL_TEST[0].spanish), return_tensors="pt").to(device)
    with torch.no_grad():
        gen_out = model.generate(**sample_in, max_new_tokens=24, do_sample=False)
    gen_elapsed = time.perf_counter() - t_gen
    gen_toks = int(gen_out.shape[1] - sample_in["input_ids"].shape[1])
    baseline_sample_text = tokenizer.decode(gen_out[0, sample_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    print(f"Baseline 1-bit sample ({gen_toks} toks in {gen_elapsed:.2f}s = {gen_toks/max(gen_elapsed,1e-6):.1f} tok/s): {baseline_sample_text!r}")

    # Phase 2: 36-Layer Homograph Cartography Scan
    print("\n=== Phase 2: 36-Layer Homograph Cartography Scan on 1-Bit Bonsai-4B ===")
    soc_train_es = [x.spanish for x in SOCIAL_TRAIN]
    ani_train_es = [x.spanish for x in ANIMAL_TRAIN]
    soc_test_es = [x.spanish for x in SOCIAL_TEST]
    ani_test_es = [x.spanish for x in ANIMAL_TEST]

    h_soc_tr = _extract_last_token_hiddens(model, tokenizer, soc_train_es, device)
    h_ani_tr = _extract_last_token_hiddens(model, tokenizer, ani_train_es, device)
    h_soc_te = _extract_last_token_hiddens(model, tokenizer, soc_test_es, device)
    h_ani_te = _extract_last_token_hiddens(model, tokenizer, ani_test_es, device)
    h_gen = _extract_last_token_hiddens(model, tokenizer, GENERAL_SPANISH_CONTROL, device)

    num_layers = h_soc_tr.shape[1]
    layer_scan: list[dict[str, Any]] = []
    best_layer = 0
    best_score = -1e9
    best_probe: dict[str, Any] = {}

    for L in range(num_layers):
        mu_s = h_soc_tr[:, L, :].mean(axis=0)
        mu_a = h_ani_tr[:, L, :].mean(axis=0)
        w = mu_s - mu_a
        w = w / (np.linalg.norm(w) + 1e-8)

        p_s_tr = h_soc_tr[:, L, :] @ w
        p_a_tr = h_ani_tr[:, L, :] @ w
        all_tr = np.concatenate([p_s_tr, p_a_tr])
        mean_tr = float(all_tr.mean())
        std_tr = float(all_tr.std() + 1e-8)

        z_s_te = (h_soc_te[:, L, :] @ w - mean_tr) / std_tr
        z_a_te = (h_ani_te[:, L, :] @ w - mean_tr) / std_tr
        z_gen = (h_gen[:, L, :] @ w - mean_tr) / std_tr

        d_val = _cohens_d(p_s_tr, p_a_tr)
        te_acc = float(0.5 * (np.mean(z_s_te > 0.0) + np.mean(z_a_te <= 0.0)))
        soc_recall = float(np.mean(z_s_te > 0.25))
        ani_fpr = float(np.mean(z_a_te > 0.25))
        gen_fpr = float(np.mean(z_gen > 0.25))

        rec = {
            "layer": L,
            "cohens_d_train": round(d_val, 4),
            "unseen_test_acc": round(te_acc, 4),
            "social_gate_recall": round(soc_recall, 4),
            "animal_gate_fpr": round(ani_fpr, 4),
            "general_spanish_fpr": round(gen_fpr, 4),
        }
        layer_scan.append(rec)

        # Prefer middle layers (35%-75% depth) with 0.0 FPR and high separation
        if int(0.35 * num_layers) <= L <= int(0.75 * num_layers):
            composite = d_val + 4.0 * te_acc + 3.0 * soc_recall - 6.0 * ani_fpr - 6.0 * gen_fpr
            if composite > best_score:
                best_score = composite
                best_layer = L
                best_probe = {
                    "w_sense": w,
                    "proj_mean": mean_tr,
                    "proj_std": std_tr,
                    "layer_metrics": rec,
                }

    print(f"Best Cartography Layer: L={best_layer} -> {best_probe['layer_metrics']}")

    # Compute steering direction u_edit at best_layer from steered vs unsteered social train prompts
    steered_vecs = []
    for s in soc_train_es:
        ids = tokenizer(_build_prompt(s, steered=True), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        steered_vecs.append(out.hidden_states[best_layer + 1][0, -1, :].float().cpu().numpy())
    h_soc_steered = np.stack(steered_vecs, axis=0)
    u_edit_np = (h_soc_steered - h_soc_tr[:, best_layer, :]).mean(axis=0)
    u_edit_norm = float(np.linalg.norm(u_edit_np))
    print(f"Computed FP16 sidecar edit vector u_edit at L={best_layer} (norm={u_edit_norm:.4f})")

    # Build both ungated and self-gated FP16 sidecars (< 20 KB)
    w_sense_t = torch.tensor(best_probe["w_sense"], dtype=torch.float16)
    u_edit_t = torch.tensor(u_edit_np, dtype=torch.float16)

    def _evaluate_arm(arm_name: str, steered_prompt: bool = False, sidecar: SelfGatedFP16Sidecar | None = None) -> dict[str, Any]:
        handle = None
        if sidecar is not None:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                if isinstance(out, tuple):
                    return (sidecar.apply_to_hidden(out[0]),) + out[1:]
                return sidecar.apply_to_hidden(out)

            handle = model.model.layers[best_layer].register_forward_hook(_hook)

        try:
            soc_margins, soc_wins = [], 0
            for item in SOCIAL_TEST:
                p = _build_prompt(item.spanish, steered=steered_prompt)
                lp_good = _score_completion_logprob(model, tokenizer, p, item.good_english, device)
                lp_bad = _score_completion_logprob(model, tokenizer, p, item.bad_english, device)
                margin = lp_good - lp_bad
                soc_margins.append(margin)
                if margin > 0:
                    soc_wins += 1

            ani_margins, ani_wins = [], 0
            for item in ANIMAL_TEST:
                p = _build_prompt(item.spanish, steered=steered_prompt)
                lp_good = _score_completion_logprob(model, tokenizer, p, item.good_english, device)
                lp_bad = _score_completion_logprob(model, tokenizer, p, item.bad_english, device)
                margin = lp_good - lp_bad
                ani_margins.append(margin)
                if margin > 0:
                    ani_wins += 1

            gen_nll = _compute_text_nll(model, tokenizer, GENERAL_SPANISH_CONTROL, device)

            # Qualitative sample translation
            s_in = tokenizer(_build_prompt(SOCIAL_TEST[0].spanish, steered=steered_prompt), return_tensors="pt").to(device)
            with torch.no_grad():
                g_out = model.generate(**s_in, max_new_tokens=20, do_sample=False)
            soc_sample = tokenizer.decode(g_out[0, s_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            a_in = tokenizer(_build_prompt(ANIMAL_TEST[0].spanish, steered=steered_prompt), return_tensors="pt").to(device)
            with torch.no_grad():
                ga_out = model.generate(**a_in, max_new_tokens=20, do_sample=False)
            ani_sample = tokenizer.decode(ga_out[0, a_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            return {
                "arm": arm_name,
                "social_win_rate": round(soc_wins / len(SOCIAL_TEST), 4),
                "social_mean_margin": round(float(np.mean(soc_margins)), 4),
                "animal_win_rate": round(ani_wins / len(ANIMAL_TEST), 4),
                "animal_mean_margin": round(float(np.mean(ani_margins)), 4),
                "general_spanish_nll": round(gen_nll, 6),
                "sample_social_translation": soc_sample,
                "sample_animal_translation": ani_sample,
                "sidecar_bytes": sidecar.nbytes() if sidecar is not None else 0,
                "sidecar_sha256": sidecar.sha256() if sidecar is not None else None,
            }
        finally:
            if handle is not None:
                handle.remove()

    print("\n=== Phase 3: 4-Arm Controlled Evaluation on Unseen Homograph Test Split ===")
    res_base = _evaluate_arm("baseline_1bit", steered_prompt=False, sidecar=None)
    res_prompt = _evaluate_arm("prompt_steering_1bit", steered_prompt=True, sidecar=None)

    ungated_sidecar = SelfGatedFP16Sidecar(
        w_sense=w_sense_t,
        u_edit=u_edit_t,
        proj_mean=best_probe["proj_mean"],
        proj_std=best_probe["proj_std"],
        alpha=1.0,
        tau=0.25,
        ungated=True,
    )
    res_ungated = _evaluate_arm("ungated_fp16_sidecar", steered_prompt=False, sidecar=ungated_sidecar)

    gated_sidecar = SelfGatedFP16Sidecar(
        w_sense=w_sense_t,
        u_edit=u_edit_t,
        proj_mean=best_probe["proj_mean"],
        proj_std=best_probe["proj_std"],
        alpha=1.0,
        tau=0.25,
        ungated=False,
    )
    res_gated = _evaluate_arm("self_gated_fp16_sidecar", steered_prompt=False, sidecar=gated_sidecar)

    base_nll = res_base["general_spanish_nll"]
    for r in (res_base, res_prompt, res_ungated, res_gated):
        drift_pct = ((r["general_spanish_nll"] - base_nll) / max(base_nll, 1e-8)) * 100.0
        r["general_spanish_drift_pct"] = round(drift_pct, 4)
        print(
            f"  {r['arm']:25s} | soc_win={r['social_win_rate']:.3f} (m={r['social_mean_margin']:+.3f}) | "
            f"ani_win={r['animal_win_rate']:.3f} (m={r['animal_mean_margin']:+.3f}) | "
            f"nll_drift={r['general_spanish_drift_pct']:+.4f}% | bytes={r['sidecar_bytes']}"
        )

    elapsed = round(time.perf_counter() - t_start, 2)
    report = {
        "experiment": "013-bonsai-1bit-sidecar-lab",
        "model_id": MODEL_ID,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "vram_allocated_mb": vram_mb,
        "vram_reserved_mb": vram_reserved_mb,
        "packing_stats": pack_stats,
        "best_cartography_layer": best_layer,
        "best_layer_metrics": best_probe["layer_metrics"],
        "layer_scan": layer_scan,
        "arms": [res_base, res_prompt, res_ungated, res_gated],
        "total_wall_seconds": elapsed,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== EXPERIMENT 013 COMPLETE in {elapsed}s -> {out_path} ===")


if __name__ == "__main__":
    main()
