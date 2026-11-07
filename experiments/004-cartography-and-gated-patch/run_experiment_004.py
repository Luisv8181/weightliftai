"""Experiment 004: Map-First Layer Cartography & Self-Gated Rank-1 Weight Patching.

Tests:
1. Layer-by-layer semantic cartography of the polysemous homograph 'tigre'
   (social/romantic rival vs. literal wild animal) across all layers of:
   - Qwen/Qwen2.5-0.5B-Instruct
   - google/gemma-4-E2B-it
   - nvidia/AceInstruct-1.5B
2. Comparison of:
   - baseline (unmodified weights)
   - ungated_rank1_mapped (rank-1 patch at mapped layer l*, always active)
   - self_gated_rank1_mapped (probe-gated rank-1 patch at mapped layer l*, active only when
     the layer-l* hidden state satisfies the social-tigre subspace gate)
3. Optional evaluation of HAL-X-AI/THX-02 as a dedicated thinking decision router
   and DISCORS-S semantic verifier.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from homograph_data import (
    ANIMAL_TIGRE_TEST,
    ANIMAL_TIGRE_TRAIN,
    GENERAL_SPANISH_TEST,
    GENERAL_SPANISH_TRAIN,
    SOCIAL_TIGRE_TEST,
    SOCIAL_TIGRE_TRAIN,
    HomographPair,
)


def format_prompt(tokenizer: Any, spanish: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following Spanish sentence into natural English. "
                "Output only the English translation.\n"
                f"Spanish: {spanish}"
            ),
        }
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return (
        "Translate the following Spanish sentence into natural English. "
        f"Output only the English translation.\nSpanish: {spanish}\nEnglish:"
    )


def load_text_causal_lm(model_id: str, device: torch.device) -> tuple[Any, Any, Any]:
    """Load causal LM and return (model, tokenizer, layers_module_list)."""
    print(f"\n[LOAD] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma-4" in model_id.lower():
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu"
        )
        # Drop vision and audio towers so 8GB VRAM holds text layers cleanly
        if hasattr(model, "model"):
            if hasattr(model.model, "vision_tower"):
                model.model.vision_tower = None
            if hasattr(model.model, "audio_tower"):
                model.model.audio_tower = None
            if hasattr(model.model, "embed_vision"):
                model.model.embed_vision = None
            if hasattr(model.model, "embed_audio"):
                model.model.embed_audio = None
        gc.collect()
        torch.cuda.empty_cache()
        model = model.to(device)
        layers = model.model.language_model.layers
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        layers = model.model.layers

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer, layers


def find_tigre_token_index(tokenizer: Any, prompt_ids: list[int]) -> int:
    """Find the last token index inside the word 'tigre' in the prompt."""
    best_idx = len(prompt_ids) - 1
    for idx in range(len(prompt_ids) - 1, -1, -1):
        piece = tokenizer.decode([prompt_ids[idx]]).lower()
        if "tigr" in piece or "tig" in piece:
            return idx
    return best_idx


@torch.no_grad()
def extract_layer_features(
    model: Any,
    tokenizer: Any,
    items: list[HomographPair],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (last_token_states, tigre_token_states) each of shape [N, num_layers, hidden_dim]."""
    last_states_list = []
    tigre_states_list = []

    for item in items:
        prompt = format_prompt(tokenizer, item.spanish)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = enc["input_ids"][0].tolist()
        tigre_idx = find_tigre_token_index(tokenizer, prompt_ids)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states[0] is embeddings; hidden_states[1..L] are layer outputs
        layer_hs = out.hidden_states[1:]
        last_vecs = [h[0, -1, :].float().cpu().numpy() for h in layer_hs]
        tigre_vecs = [h[0, tigre_idx, :].float().cpu().numpy() for h in layer_hs]
        last_states_list.append(np.stack(last_vecs, axis=0))
        tigre_states_list.append(np.stack(tigre_vecs, axis=0))

    return np.stack(last_states_list, axis=0), np.stack(tigre_states_list, axis=0)


def compute_loo_and_cohens_d(
    pos_vecs: np.ndarray, neg_vecs: np.ndarray
) -> tuple[float, float, np.ndarray, float]:
    """Compute Leave-One-Out linear probe accuracy and Cohen's d for layer vectors [N, D]."""
    n_pos = pos_vecs.shape[0]
    n_neg = neg_vecs.shape[0]
    all_vecs = np.concatenate([pos_vecs, neg_vecs], axis=0)
    labels = np.array([1] * n_pos + [0] * n_neg)

    correct = 0
    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False
        tr_vecs = all_vecs[mask]
        tr_lbls = labels[mask]
        mu_pos = tr_vecs[tr_lbls == 1].mean(axis=0)
        mu_neg = tr_vecs[tr_lbls == 0].mean(axis=0)
        diff = mu_pos - mu_neg
        norm = np.linalg.norm(diff) + 1e-8
        w = diff / norm
        thresh = 0.5 * float(np.dot(w, mu_pos + mu_neg))
        score = float(np.dot(w, all_vecs[i]))
        pred = 1 if score > thresh else 0
        if pred == labels[i]:
            correct += 1

    loo_acc = correct / len(labels)

    # Full fit for Cohen's d and probe direction
    mu_pos = pos_vecs.mean(axis=0)
    mu_neg = neg_vecs.mean(axis=0)
    diff = mu_pos - mu_neg
    norm = np.linalg.norm(diff) + 1e-8
    w = diff / norm
    thresh = 0.5 * float(np.dot(w, mu_pos + mu_neg))

    proj_pos = pos_vecs @ w
    proj_neg = neg_vecs @ w
    var_pos = np.var(proj_pos, ddof=1)
    var_neg = np.var(proj_neg, ddof=1)
    pooled_sd = math.sqrt(0.5 * (var_pos + var_neg) + 1e-8)
    cohens_d = float((proj_pos.mean() - proj_neg.mean()) / pooled_sd)

    return loo_acc, cohens_d, w, thresh


def run_cartography_scan(
    model: Any,
    tokenizer: Any,
    num_layers: int,
    device: torch.device,
) -> dict[str, Any]:
    """Scan every layer l in [0, num_layers-1] on the all-homograph 'tigre' dataset."""
    soc_tr_last, soc_tr_tig = extract_layer_features(model, tokenizer, SOCIAL_TIGRE_TRAIN, device)
    ani_tr_last, ani_tr_tig = extract_layer_features(model, tokenizer, ANIMAL_TIGRE_TRAIN, device)
    gen_tr_last, _ = extract_layer_features(model, tokenizer, GENERAL_SPANISH_TRAIN, device)

    soc_te_last, soc_te_tig = extract_layer_features(model, tokenizer, SOCIAL_TIGRE_TEST, device)
    ani_te_last, ani_te_tig = extract_layer_features(model, tokenizer, ANIMAL_TIGRE_TEST, device)
    gen_te_last, _ = extract_layer_features(model, tokenizer, GENERAL_SPANISH_TEST, device)

    layer_metrics: list[dict[str, Any]] = []
    best_layer = num_layers // 2
    best_score = -1e9
    best_probe_params: dict[str, Any] = {}

    min_candidate_layer = max(2, int(0.35 * num_layers))
    max_candidate_layer = min(num_layers - 2, int(0.85 * num_layers))

    for l in range(num_layers):
        loo_last, d_last, w_last, th_last = compute_loo_and_cohens_d(
            soc_tr_last[:, l, :], ani_tr_last[:, l, :]
        )
        loo_tig, d_tig, w_tig, th_tig = compute_loo_and_cohens_d(
            soc_tr_tig[:, l, :], ani_tr_tig[:, l, :]
        )

        # Also build a joint gate separating social_tigre from BOTH animal_tigre and general_spanish
        neg_joint_tr = np.concatenate([ani_tr_last[:, l, :], gen_tr_last[:, l, :]], axis=0)
        loo_joint, d_joint, w_joint, _ = compute_loo_and_cohens_d(
            soc_tr_last[:, l, :], neg_joint_tr
        )
        # Set conservative threshold so 100% of training animal + general Spanish are below threshold
        soc_scores_tr = soc_tr_last[:, l, :] @ w_joint
        neg_scores_tr = neg_joint_tr @ w_joint
        max_neg_tr = float(np.max(neg_scores_tr))
        min_soc_tr = float(np.min(soc_scores_tr))
        if min_soc_tr > max_neg_tr:
            gate_thresh = 0.5 * (min_soc_tr + max_neg_tr)
        else:
            gate_thresh = float(np.percentile(neg_scores_tr, 95) + 0.25 * np.std(neg_scores_tr))

        # Evaluate on unseen test set
        soc_scores_te = soc_te_last[:, l, :] @ w_joint
        ani_scores_te = ani_te_last[:, l, :] @ w_joint
        gen_scores_te = gen_te_last[:, l, :] @ w_joint

        test_soc_recall = float(np.mean(soc_scores_te > gate_thresh))
        test_ani_fp = float(np.mean(ani_scores_te > gate_thresh))
        test_gen_fp = float(np.mean(gen_scores_te > gate_thresh))
        test_acc_last = float(
            0.5 * (np.mean((soc_te_last[:, l, :] @ w_last) > th_last) + np.mean((ani_te_last[:, l, :] @ w_last) <= th_last))
        )
        test_acc_tig = float(
            0.5 * (np.mean((soc_te_tig[:, l, :] @ w_tig) > th_tig) + np.mean((ani_te_tig[:, l, :] @ w_tig) <= th_tig))
        )

        entry = {
            "layer": l,
            "loo_acc_last_token": round(loo_last, 4),
            "cohens_d_last_token": round(d_last, 4),
            "loo_acc_tigre_token": round(loo_tig, 4),
            "cohens_d_tigre_token": round(d_tig, 4),
            "loo_acc_joint_gate": round(loo_joint, 4),
            "cohens_d_joint_gate": round(d_joint, 4),
            "test_acc_last_token": round(test_acc_last, 4),
            "test_acc_tigre_token": round(test_acc_tig, 4),
            "test_gate_social_recall": round(test_soc_recall, 4),
            "test_gate_animal_fp": round(test_ani_fp, 4),
            "test_gate_general_fp": round(test_gen_fp, 4),
        }
        layer_metrics.append(entry)

        # Selection criterion Uses ONLY training metrics (zero test leakage)
        if min_candidate_layer <= l <= max_candidate_layer:
            train_margin = min_soc_tr - max_neg_tr
            composite = loo_joint * 2.0 + loo_tig + 0.15 * d_joint + 0.1 * train_margin
            if composite > best_score:
                best_score = composite
                best_layer = l
                best_probe_params = {
                    "layer": l,
                    "w_joint": w_joint,
                    "gate_thresh": gate_thresh,
                    "min_soc_tr": min_soc_tr,
                    "max_neg_tr": max_neg_tr,
                }

    return {
        "num_layers": num_layers,
        "selected_peak_layer": best_layer,
        "selected_layer_metrics": layer_metrics[best_layer],
        "layer_metrics": layer_metrics,
        "probe_params": best_probe_params,
    }


@torch.no_grad()
def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: torch.device,
    gate_controller: Any | None = None,
) -> tuple[float, int]:
    """Compute average per-token log-probability of `continuation` given `prompt`."""
    if gate_controller is not None:
        gate_controller.reset()
        # Pre-evaluate the gate on the prompt's final token so the decision is fixed for the sentence
        p_enc = tokenizer(prompt, return_tensors="pt").to(device)
        gate_controller.mode = "measure_prompt"
        _ = model(**p_enc, use_cache=False)
        gate_controller.mode = "apply_fixed_gate"

    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    c_ids = tokenizer.encode(" " + continuation.strip(), add_special_tokens=False)
    input_ids = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)

    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]  # [seq_len, vocab]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    start = len(p_ids) - 1
    end = len(p_ids) + len(c_ids) - 1
    target_ids = torch.tensor(c_ids, dtype=torch.long, device=device)
    token_lps = log_probs[start:end].gather(1, target_ids.unsqueeze(1)).squeeze(1)

    if gate_controller is not None:
        gate_controller.mode = "off"

    return float(token_lps.mean().item()), len(c_ids)


@torch.no_grad()
def generate_translation(
    model: Any,
    tokenizer: Any,
    spanish: str,
    device: torch.device,
    gate_controller: Any | None = None,
) -> str:
    prompt = format_prompt(tokenizer, spanish)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if gate_controller is not None:
        gate_controller.reset()
        gate_controller.mode = "measure_prompt"
        _ = model(**enc, use_cache=False)
        gate_controller.mode = "apply_fixed_gate"

    gen_ids = model.generate(
        **enc,
        max_new_tokens=28,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    if gate_controller is not None:
        gate_controller.mode = "off"

    new_tokens = gen_ids[0, enc["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.split("\n")[0].strip()


class SelfGatedRank1Controller:
    """Wraps a layer's mlp.down_proj to apply a rank-1 weight patch ΔW = u @ v^T
    either unconditionally (ungated) or gated by the cartography probe at layer l*."""

    def __init__(
        self,
        layer_module: nn.Module,
        w_joint: np.ndarray,
        gate_thresh: float,
        device: torch.device,
        max_rel_fro: float = 0.05,
    ) -> None:
        self.layer_module = layer_module
        self.down_proj: nn.Linear = layer_module.mlp.down_proj
        self.out_features, self.in_features = self.down_proj.weight.shape
        self.device = device
        self.max_rel_fro = max_rel_fro

        with torch.no_grad():
            self.base_fro = float(self.down_proj.weight.float().norm().item())

        self.u = nn.Parameter(torch.randn(self.out_features, device=device, dtype=torch.float32) * 0.01)
        self.v = nn.Parameter(torch.randn(self.in_features, device=device, dtype=torch.float32) * 0.01)

        self.w_joint = torch.tensor(w_joint, dtype=torch.float32, device=device)
        self.gate_thresh = float(gate_thresh)

        self.mode = "off"  # "off" | "always_on" | "measure_prompt" | "apply_fixed_gate"
        self.gate_active = False
        self.last_probe_score = 0.0

        self._layer_hook = layer_module.register_forward_hook(self._on_layer_output)
        self._down_hook = self.down_proj.register_forward_hook(self._on_down_proj)

    def reset(self) -> None:
        self.gate_active = False
        self.last_probe_score = 0.0

    def remove(self) -> None:
        self._layer_hook.remove()
        self._down_hook.remove()

    def project_fro_budget(self) -> None:
        with torch.no_grad():
            u_norm = self.u.norm()
            v_norm = self.v.norm()
            prod = float((u_norm * v_norm).item())
            max_fro = self.max_rel_fro * self.base_fro
            if prod > max_fro and prod > 0:
                scale = math.sqrt(max_fro / prod)
                self.u.mul_(scale)
                self.v.mul_(scale)

    def _on_down_proj(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if self.mode == "always_on" or (self.mode == "apply_fixed_gate" and self.gate_active):
            x = inputs[0].float()
            # (x @ v) -> [..., 1], outer with u -> [..., out_features]
            xv = torch.matmul(x, self.v)
            delta = xv.unsqueeze(-1) * self.u
            return output + delta.to(output.dtype)
        return output

    def _on_layer_output(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        if self.mode == "measure_prompt":
            h = output[0] if isinstance(output, tuple) else output
            last_h = h[0, -1, :].float()
            score = float(torch.dot(last_h, self.w_joint).item())
            self.last_probe_score = score
            self.gate_active = score > self.gate_thresh
        return output


def train_rank1_patch(
    model: Any,
    tokenizer: Any,
    controller: SelfGatedRank1Controller,
    device: torch.device,
    include_animal_regularization: bool,
    steps: int = 60,
    lr: float = 0.03,
) -> None:
    """Optimize the rank-1 factors (u, v) at the mapped layer l*."""
    optimizer = torch.optim.Adam([controller.u, controller.v], lr=lr)
    controller.mode = "always_on"

    train_items = list(SOCIAL_TIGRE_TRAIN)
    if include_animal_regularization:
        train_items = train_items + list(ANIMAL_TIGRE_TRAIN)

    rng = np.random.default_rng(42)
    for step in range(steps):
        item = train_items[int(rng.integers(0, len(train_items)))]
        prompt = format_prompt(tokenizer, item.spanish)
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        g_ids = tokenizer.encode(" " + item.good_english.strip(), add_special_tokens=False)
        b_ids = tokenizer.encode(" " + item.bad_english.strip(), add_special_tokens=False)

        in_g = torch.tensor([p_ids + g_ids], dtype=torch.long, device=device)
        in_b = torch.tensor([p_ids + b_ids], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out_g = model(input_ids=in_g, use_cache=False)
        lp_g = F.log_softmax(out_g.logits[0].float(), dim=-1)
        tg = torch.tensor(g_ids, dtype=torch.long, device=device)
        score_g = lp_g[len(p_ids) - 1 : len(p_ids) + len(g_ids) - 1].gather(1, tg.unsqueeze(1)).mean()

        out_b = model(input_ids=in_b, use_cache=False)
        lp_b = F.log_softmax(out_b.logits[0].float(), dim=-1)
        tb = torch.tensor(b_ids, dtype=torch.long, device=device)
        score_b = lp_b[len(p_ids) - 1 : len(p_ids) + len(b_ids) - 1].gather(1, tb.unsqueeze(1)).mean()

        # Maximize good translation logprob and margin over bad translation
        margin = score_g - score_b
        loss = -score_g + F.relu(1.2 - margin)
        loss.backward()
        optimizer.step()
        controller.project_fro_budget()

    controller.mode = "off"


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    condition_name: str,
    controller: SelfGatedRank1Controller | None = None,
    use_gate: bool = False,
) -> dict[str, Any]:
    """Evaluate a condition on held-out SOCIAL_TIGRE_TEST, ANIMAL_TIGRE_TEST, and GENERAL_SPANISH_TEST."""
    soc_margins = []
    soc_wins = 0
    soc_gates = 0

    for item in SOCIAL_TIGRE_TEST:
        prompt = format_prompt(tokenizer, item.spanish)
        if controller is not None and not use_gate:
            controller.mode = "always_on"
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, None)
            gate_fired = True
        elif controller is not None and use_gate:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, controller)
            gate_fired = controller.gate_active
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, controller)
        else:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, None)
            gate_fired = False

        margin = lp_g - lp_b
        soc_margins.append(margin)
        if margin > 0:
            soc_wins += 1
        if gate_fired:
            soc_gates += 1

    ani_margins = []
    ani_wins = 0
    ani_gates = 0
    for item in ANIMAL_TIGRE_TEST:
        prompt = format_prompt(tokenizer, item.spanish)
        if controller is not None and not use_gate:
            controller.mode = "always_on"
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, None)
            gate_fired = True
        elif controller is not None and use_gate:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, controller)
            gate_fired = controller.gate_active
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, controller)
        else:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            lp_b, _ = sequence_logprob(model, tokenizer, prompt, item.bad_english, device, None)
            gate_fired = False

        margin = lp_g - lp_b
        ani_margins.append(margin)
        if margin > 0:
            ani_wins += 1
        if gate_fired:
            ani_gates += 1

    gen_nlls = []
    gen_gates = 0
    for item in GENERAL_SPANISH_TEST:
        prompt = format_prompt(tokenizer, item.spanish)
        if controller is not None and not use_gate:
            controller.mode = "always_on"
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            gate_fired = True
        elif controller is not None and use_gate:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, controller)
            gate_fired = controller.gate_active
        else:
            lp_g, _ = sequence_logprob(model, tokenizer, prompt, item.good_english, device, None)
            gate_fired = False
        gen_nlls.append(-lp_g)
        if gate_fired:
            gen_gates += 1

    sample_sentences = [
        SOCIAL_TIGRE_TEST[0].spanish,  # "Mi novia me dejó por un tigre."
        SOCIAL_TIGRE_TEST[1].spanish,  # "Yo siento como que mi novia me dejó por un tigre."
        SOCIAL_TIGRE_TEST[2].spanish,  # "Ese tigre le escribió mensajes a mi esposa toda la noche."
        ANIMAL_TIGRE_TEST[0].spanish,  # "Vimos un tigre enorme en el zoológico de la ciudad."
    ]
    generations = {}
    for s in sample_sentences:
        if controller is not None and not use_gate:
            controller.mode = "always_on"
            generations[s] = generate_translation(model, tokenizer, s, device, None)
        elif controller is not None and use_gate:
            generations[s] = generate_translation(model, tokenizer, s, device, controller)
        else:
            generations[s] = generate_translation(model, tokenizer, s, device, None)

    if controller is not None:
        controller.mode = "off"

    return {
        "condition": condition_name,
        "social_tigre_win_rate": round(soc_wins / len(SOCIAL_TIGRE_TEST), 4),
        "social_tigre_mean_margin": round(float(np.mean(soc_margins)), 4),
        "social_gate_activation_rate": round(soc_gates / len(SOCIAL_TIGRE_TEST), 4),
        "animal_tigre_win_rate": round(ani_wins / len(ANIMAL_TIGRE_TEST), 4),
        "animal_tigre_mean_margin": round(float(np.mean(ani_margins)), 4),
        "animal_gate_activation_rate": round(ani_gates / len(ANIMAL_TIGRE_TEST), 4),
        "general_spanish_mean_nll": round(float(np.mean(gen_nlls)), 4),
        "general_gate_activation_rate": round(gen_gates / len(GENERAL_SPANISH_TEST), 4),
        "sample_generations": generations,
    }


def evaluate_thx02(device: torch.device) -> dict[str, Any]:
    """Evaluate HAL-X-AI/THX-02 both as a decision router/verifier and via 32-layer cartography."""
    import os
    from thx02 import THX02, choice, noul

    print("\n[THX-02] Loading HAL-X-AI/THX-02 decision model (dropping vision tower for 8GB VRAM)...")
    os.environ["THX02_CPU_DTYPE"] = "bfloat16"
    thx = THX02.load("HAL-X-AI/THX-02", device="cpu", fast=False)
    if hasattr(thx.d.backbone, "visual"):
        thx.d.backbone.visual = None
    gc.collect()
    torch.cuda.empty_cache()
    thx.d.backbone.to(device)
    thx.d.readout.to(device)
    thx.d.device = device

    sense_question = choice(
        "In this Spanish utterance, what does the word 'tigre' refer to?",
        social_person="A person (romantic rival, street-smart player, or hustler in Caribbean/Dominican Spanish slang)",
        literal_animal="A literal wild tiger (large striped feline predator in a zoo or jungle)",
    )
    distortion_question = noul(
        "Would translating 'tigre' literally as the English word 'tiger' distort the speaker's intended meaning?"
    )

    soc_correct = 0
    soc_details = []
    for item in SOCIAL_TIGRE_TEST:
        out = thx.decide(
            f"Spanish utterance: {item.spanish}",
            {"sense": sense_question, "literal_tiger_distorts": distortion_question},
            tta=2,
        )
        ans_sense = out["answers"]["sense"]
        ans_dist = out["answers"]["literal_tiger_distorts"]
        pred_choice = ans_sense.get("choice", "")
        is_ok = pred_choice == "social_person"
        if is_ok:
            soc_correct += 1
        soc_details.append({
            "spanish": item.spanish,
            "predicted_sense": pred_choice,
            "sense_distribution": ans_sense.get("distribution", {}),
            "literal_tiger_distorts_prob": round(float(ans_dist.get("noul", 0.0)), 4),
            "latency_ms": out.get("latency_ms", 0.0),
            "correct": is_ok,
        })

    ani_correct = 0
    ani_details = []
    for item in ANIMAL_TIGRE_TEST:
        out = thx.decide(
            f"Spanish utterance: {item.spanish}",
            {"sense": sense_question, "literal_tiger_distorts": distortion_question},
            tta=2,
        )
        ans_sense = out["answers"]["sense"]
        ans_dist = out["answers"]["literal_tiger_distorts"]
        pred_choice = ans_sense.get("choice", "")
        is_ok = pred_choice == "literal_animal"
        if is_ok:
            ani_correct += 1
        ani_details.append({
            "spanish": item.spanish,
            "predicted_sense": pred_choice,
            "sense_distribution": ans_sense.get("distribution", {}),
            "literal_tiger_distorts_prob": round(float(ans_dist.get("noul", 0.0)), 4),
            "latency_ms": out.get("latency_ms", 0.0),
            "correct": is_ok,
        })

    # Also test THX-02 as DISCORS-S Stage 4 semantic verifier on candidate translations
    verifier_checks = []
    for item in [SOCIAL_TIGRE_TEST[0], SOCIAL_TIGRE_TEST[1], ANIMAL_TIGRE_TEST[0]]:
        for label, cand in [("good_english", item.good_english), ("bad_english", item.bad_english)]:
            v_out = thx.decide(
                f"Source Spanish utterance: {item.spanish}\nCandidate English translation: {cand}",
                {
                    "preserves_referent": noul(
                        "Does the English translation accurately preserve the real-world meaning of 'tigre' in the Spanish utterance?"
                    )
                },
            )
            verifier_checks.append({
                "spanish": item.spanish,
                "candidate_type": label,
                "candidate_english": cand,
                "preserves_referent_prob": round(
                    float(v_out["answers"]["preserves_referent"].get("noul", 0.0)), 4
                ),
            })

    # Also run 32-layer cartography scan across THX-02's non-causal Qwen3.5 language model
    print("[THX-02] Running 32-layer cartography scan across THX-02 backbone...")
    tokenizer = thx.d.processor.tokenizer
    lm = thx.d.backbone.language_model
    num_layers = len(lm.layers)

    @torch.no_grad()
    def extract_thx_states(items: list[HomographPair]) -> tuple[np.ndarray, np.ndarray]:
        last_list, tig_list = [], []
        for it in items:
            enc = tokenizer(it.spanish, return_tensors="pt").to(device)
            ids = enc["input_ids"][0].tolist()
            tig_idx = find_tigre_token_index(tokenizer, ids)
            out = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[1:]
            last_list.append(np.stack([h[0, -1, :].float().cpu().numpy() for h in hs], axis=0))
            tig_list.append(np.stack([h[0, tig_idx, :].float().cpu().numpy() for h in hs], axis=0))
        return np.stack(last_list, axis=0), np.stack(tig_list, axis=0)

    soc_tr_l, soc_tr_t = extract_thx_states(SOCIAL_TIGRE_TRAIN)
    ani_tr_l, ani_tr_t = extract_thx_states(ANIMAL_TIGRE_TRAIN)
    soc_te_l, soc_te_t = extract_thx_states(SOCIAL_TIGRE_TEST)
    ani_te_l, ani_te_t = extract_thx_states(ANIMAL_TIGRE_TEST)

    thx_layers = []
    for l in range(num_layers):
        loo_l, d_l, w_l, th_l = compute_loo_and_cohens_d(soc_tr_l[:, l, :], ani_tr_l[:, l, :])
        loo_t, d_t, w_t, th_t = compute_loo_and_cohens_d(soc_tr_t[:, l, :], ani_tr_t[:, l, :])
        te_acc_l = float(0.5 * (np.mean((soc_te_l[:, l, :] @ w_l) > th_l) + np.mean((ani_te_l[:, l, :] @ w_l) <= th_l)))
        te_acc_t = float(0.5 * (np.mean((soc_te_t[:, l, :] @ w_t) > th_t) + np.mean((ani_te_t[:, l, :] @ w_t) <= th_t)))
        thx_layers.append({
            "layer": l,
            "loo_acc_last_token": round(loo_l, 4),
            "cohens_d_last_token": round(d_l, 4),
            "loo_acc_tigre_token": round(loo_t, 4),
            "cohens_d_tigre_token": round(d_t, 4),
            "test_acc_last_token": round(te_acc_l, 4),
            "test_acc_tigre_token": round(te_acc_t, 4),
        })

    return {
        "model_id": "HAL-X-AI/THX-02",
        "social_tigre_accuracy": round(soc_correct / len(SOCIAL_TIGRE_TEST), 4),
        "animal_tigre_accuracy": round(ani_correct / len(ANIMAL_TIGRE_TEST), 4),
        "overall_homograph_accuracy": round(
            (soc_correct + ani_correct) / (len(SOCIAL_TIGRE_TEST) + len(ANIMAL_TIGRE_TEST)), 4
        ),
        "social_predictions": soc_details,
        "animal_predictions": ani_details,
        "discors_s_verifier_checks": verifier_checks,
        "backbone_cartography_num_layers": num_layers,
        "backbone_layer_metrics": thx_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "Qwen/Qwen2.5-0.5B-Instruct",
            "google/gemma-4-E2B-it",
            "nvidia/AceInstruct-1.5B",
        ],
    )
    parser.add_argument("--run-thx02", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/004-cartography-and-gated-patch/results.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {"models": {}, "thx02": None}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for model_id in args.models:
        t0 = time.time()
        model, tokenizer, layers = load_text_causal_lm(model_id, device)
        num_layers = len(layers)
        print(f"[MAP] Scanning all {num_layers} layers of {model_id}...")
        cartography = run_cartography_scan(model, tokenizer, num_layers, device)
        peak_layer = cartography["selected_peak_layer"]
        probe_params = cartography["probe_params"]
        print(
            f"[MAP] {model_id}: Peak layer l* = {peak_layer}/{num_layers - 1} "
            f"| LOO Joint Acc = {cartography['selected_layer_metrics']['loo_acc_joint_gate']} "
            f"| Test Social Gate Recall = {cartography['selected_layer_metrics']['test_gate_social_recall']} "
            f"| Test Animal FP = {cartography['selected_layer_metrics']['test_gate_animal_fp']}"
        )

        # 1. Baseline evaluation
        print(f"[EVAL] {model_id} -> baseline...")
        base_eval = evaluate_condition(model, tokenizer, device, "baseline", None, False)

        # 2. Train rank-1 patch at mapped peak layer l*
        controller = SelfGatedRank1Controller(
            layer_module=layers[peak_layer],
            w_joint=probe_params["w_joint"],
            gate_thresh=probe_params["gate_thresh"],
            device=device,
            max_rel_fro=0.05,
        )
        print(f"[TRAIN] {model_id} -> Optimizing rank-1 patch at Layer {peak_layer}...")
        train_rank1_patch(
            model,
            tokenizer,
            controller,
            device,
            include_animal_regularization=False,
            steps=70,
            lr=0.03,
        )

        # 3. Evaluate ungated rank-1 patch (always active at layer l*)
        print(f"[EVAL] {model_id} -> ungated_rank1_mapped (Layer {peak_layer})...")
        ungated_eval = evaluate_condition(
            model, tokenizer, device, f"ungated_rank1_L{peak_layer}", controller, use_gate=False
        )

        # 4. Evaluate self-gated rank-1 patch (active only when cartography gate fires)
        print(f"[EVAL] {model_id} -> self_gated_rank1_mapped (Layer {peak_layer})...")
        gated_eval = evaluate_condition(
            model, tokenizer, device, f"self_gated_rank1_L{peak_layer}", controller, use_gate=True
        )

        base_nll = base_eval["general_spanish_mean_nll"]
        ungated_eval["general_spanish_nll_delta_pct"] = round(
            100.0 * (ungated_eval["general_spanish_mean_nll"] - base_nll) / max(base_nll, 1e-6), 2
        )
        gated_eval["general_spanish_nll_delta_pct"] = round(
            100.0 * (gated_eval["general_spanish_mean_nll"] - base_nll) / max(base_nll, 1e-6), 2
        )

        controller.remove()
        del cartography["probe_params"]

        existing["models"][model_id] = {
            "model_id": model_id,
            "elapsed_sec": round(time.time() - t0, 2),
            "cartography": cartography,
            "evaluations": {
                "baseline": base_eval,
                "ungated_rank1_mapped": ungated_eval,
                "self_gated_rank1_mapped": gated_eval,
            },
        }
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if args.run_thx02:
        existing["thx02"] = evaluate_thx02(device)
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[DONE] Saved Experiment 004 results to {out_path}")


if __name__ == "__main__":
    main()
