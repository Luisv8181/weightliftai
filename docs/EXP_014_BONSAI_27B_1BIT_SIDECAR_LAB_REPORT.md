# Experiment 013b & 014 Report: 1-Bit FP16 Self-Gated Activation Steering Sidecar Lab (Bonsai-4B & Bonsai-27B)

**Date:** 2026-11-26 (Updated with B1-B5 audit findings)  
**Repository:** `weightliftai` (branch `feat/bonsai-27b-wsp-001b`)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.55 GB VRAM), PyTorch 2.10.0+cu128, Transformers 5.17.0  
**Method Clarification:** `SelfGatedFP16Sidecar` adds a context-gated activation vector to the residual stream via forward hooks. This is **activation steering**, not direct weight-space editing (the base 1-bit weights remain frozen).

> **BLOCKING VALIDATION NOTICE (Exp 014, B1):** Every Experiment 014 number below is **pending loader validation**. Although the `HAL-X-AI/THX-02` tokenizer matches `Bonsai-27B-Q1_0.gguf`'s embedded vocabulary (`0/5000` token mismatches), a parity check against Ollama (`bonsai-27b-q1:latest`) showed a clear mismatch: Ollama raw completion on `"The capital of France is"` produces `" Paris."`, whereas `gguf_1bit_loader.py` produced `' the' (16.1%) / ' a' (12.5%) / ' Lyon' (10.4%) / ' Paris' (6.7%)` and an unusually high baseline General Spanish NLL of **`7.19`** (vs. `2.45` for Gemma 4 E2B in Exp 004). Until a 20-prompt logit/NLL parity test against `llama.cpp` / Ollama passes, treat all Exp 014 metrics as preliminary and unvalidated.

---

## 1. Executive Summary

1. **Experiment 013b (`prism-ml/Bonsai-4B-unpacked`, 1.29 GB 1-bit VRAM, 41 KB sidecar):**
   - **Headline Result:** **No effect on the target (`0/8` unseen social `"tigre"` accuracy, unchanged from `0/8` baseline); no collateral drift (`+0.0000%` General Spanish NLL drift, `0/8` animal FPR, `0/8` general Spanish FPR).**
   - Under the `max_vec_norm = 3.5` norm budget against residual norms of `~85`, the 41 KB activation steering vectors shifted social margin from `-12.870` to `-10.411` ungated (`-12.179` gated) and did not flip any unseen social prediction.

2. **Experiment 014 (`Bonsai-27B-Q1_0.gguf`, 3.81 GB loaded / 4.94 GB peak VRAM, 61.5 KB sidecar, *pending loader validation*):**
   - **Directional (`n=8`, Fisher exact `p ~ 0.31`, not statistically distinguishable):** On unseen Dominican/Caribbean social `"tigre"` test items, the 61.5 KB self-gated activation steering sidecar (`L35` to `L37`) scored `5/8` (`62.5%`) vs. `2/8` (`25.0%`) baseline and `0/8` (`0.0%`) prompt-steered.
   - **Collateral & Gate Limitations (Summary):** The Layer 36 self-gate misfired on **`37.5%` (`3/8`) of unseen animal sentences** and **`12.5%` (`1/8`) of unseen general Spanish prompts**, and the self-gated sidecar still caused **`+8.54%` General Spanish NLL drift** (vs. `+10.12%` ungated) and dropped animal win rate from `100.0%` (`8/8`) to `87.5%` (`7/8`).
   - **Unresolved Anomaly:** The sidecar vector was distilled from the *prompt-steered* hidden states, even though the prompt-steered condition itself reduced accuracy (`25.0% -> 0.0%`). Because the loader parity bug (B1) is still open, this discrepancy may reflect a scoring or loader artifact and must be re-tested after B1 parity is fixed.

---

## 2. Experiment 013b: Bonsai-4B 3-Way Discriminative Self-Gate (No Target Effect; No Collateral Drift)

**Script:** `experiments/013-bonsai-1bit-sidecar-lab/run_sidecar_multilayer_contrastive.py`  
**Results:** `experiments/013-bonsai-1bit-sidecar-lab/results_013b_multilayer.json`  
**Verdict:** **Not replicated (target repair)** / **No collateral drift** (`0/8` unseen social wins; `+0.0000%` NLL drift)

| Condition | Social Win Rate (`n=8`) | Social Mean Margin | Animal Win Rate (`n=8`) | Animal Mean Margin | General Spanish NLL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_1bit` | `0.0%` (`0/8`) | `-12.870` | `100.0%` (`8/8`) | `+19.528` | `+0.0000%` |
| `prompt_steering_1bit` | `12.5%` (`1/8`) | `-6.110` | `100.0%` (`8/8`) | `+15.826` | `+0.0000%` |
| `ungated_multilayer_sidecar` (`41,088 B`) | `0.0%` (`0/8`) | `-10.411` | `100.0%` (`8/8`) | `+18.495` | `-0.0644%` |
| `self_gated_multilayer_sidecar` (`41,088 B`) | `0.0%` (`0/8`) | `-12.179` | `100.0%` (`8/8`) | `+19.528` | `+0.0000%` |

---

## 3. Experiment 014: Native `Q1_0` GGUF Bonsai-27B Activation Sidecar Lab *(Pending Loader Validation)*

**Loader:** `src/weightliftai/weight_programming/gguf_1bit_loader.py`  
**Script:** `experiments/014-bonsai-27b-1bit-sidecar-lab/run_bonsai_27b_sidecar_lab.py`  
**Results:** `experiments/014-bonsai-27b-1bit-sidecar-lab/results.json`  
**Verdict:** **Pending loader validation (B1)** / **Inconclusive (n too small, directional `n=8`, Fisher `p ~ 0.31`)**

### 3.1 Memory Profile
- **Allocated CUDA VRAM (Loaded 64-Layer Model):** `3,814.11 MB` (`3.81 GB`)
- **Peak CUDA VRAM (Chunked `lm_head`):** `4,943.27 MB` (`4.94 GB`)
- **Activation Sidecar Footprint (`L35` to `L37`):** `61,536 bytes` (`61.5 KB`)

### 3.2 4-Condition Evaluation (`n=8` per split, *pending loader validation*)

| Condition | Unseen Social Win Rate | Social Mean Margin | Unseen Animal Win Rate | Animal Mean Margin | General Spanish NLL | NLL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_1bit_27b` | `25.0%` (`2/8`) | `-5.252` | `100.0%` (`8/8`) | `+8.701` | `7.18900` *(B1 flag)* | `+0.0000%` |
| `prompt_steering_1bit_27b` | `0.0%` (`0/8`) | `-10.756` | `100.0%` (`8/8`) | `+12.466` | `7.18900` | `+0.0000%` |
| `ungated_fp16_sidecar_27b` (`61.5 KB`) | `62.5%` (`5/8`) | `-2.839` | `62.5%` (`5/8`) | `+3.160` | `7.91663` | `+10.1215%` |
| `self_gated_fp16_sidecar_27b` (`61.5 KB`) | `62.5%` (`5/8`, `p~0.31`) | `-3.517` | `87.5%` (`7/8`) | `+7.959` | `7.80263` | `+8.5358%` |

---

## 4. Blocking Next Step Before Merging PR #17

1. **Fix and verify `gguf_1bit_loader.py` parity against Ollama / `llama.cpp` (B1):** Diagnose the exact `Q1_0` / `Qwen3_5` tensor unpacking discrepancy causing `NLL = 7.19` and `' the' / ' Lyon'` instead of `' Paris.'`, and add a 20-prompt parity test (`@pytest.mark.gpu`) comparing top-5 next tokens and per-token loss against Ollama.
2. **Re-run Exp 014 after parity passes** and investigate the steered-prompt (`25% -> 0%`) vs. sidecar (`25% -> 62.5%`) discrepancy on a larger sample (`n >= 60` per class per Priority 2).
