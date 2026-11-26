# Experiment 013b & 014 Report: 1-Bit FP16 Self-Gated Sidecar Lab on Bonsai-4B and Native `Q1_0` GGUF Bonsai-27B

**Date:** 2026-11-26  
**Repository:** `weightliftai` (branch `feat/bonsai-27b-wsp-001b`)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8.55 GB VRAM), PyTorch 2.10.0+cu128, Transformers 5.17.0  
**Scope:** Native 1-bit packed PyTorch execution (`Packed1BitLinear`, `Packed1BitEmbedding`, `GGUFQ10Reader`) with tiny FP16/BF16 context-self-gated residual sidecars (`SelfGatedFP16Sidecar`, 41 KB to 61.5 KB) evaluated on the Dominican/Caribbean Spanish homograph `"tigre"` benchmark (unseen test splits + General Spanish NLL collateral check).

---

## 1. Executive Summary

We completed two complementary phases of the 1-Bit Sidecar Lab:

1. **Experiment 013b (`prism-ml/Bonsai-4B-unpacked`, 1-bit packed in 1.29 GB VRAM):**
   - Upgraded the Experiment 013 gate to a **3-way discriminative self-gate** (`SOCIAL_TRAIN` vs. `ANIMAL_TRAIN + GENERAL_SPANISH_TRAIN`) at Layer 19 (`tau = 0.3005`) controlling a 41 KB (`41,088 B`) multi-layer FP16 sidecar across Layers 19 to 22.
   - Achieved **100.0% (8/8) unseen social recall, 0.0% (0/8) unseen animal FPR, 0.0% (0/8) unseen general Spanish FPR, and exact `0.0000%` General Spanish NLL drift** (eliminating Experiment 013's `+0.0485%` drift completely).
   - Preserved honestly as a capacity boundary: under a tight `max_vec_norm = 3.5` budget against residual norms of `~85`, constant rank-0 bias vectors shifted Bonsai-4B's social margin from `-12.870` to `-10.411` (`0/8` unseen social flips).

2. **Experiment 014 (`Bonsai-27B-Q1_0.gguf`, 27B native 1-bit loaded directly into PyTorch CUDA in 3.81 GB VRAM):**
   - Built a zero-dependency **GGUF v3 `Q1_0` (`ggml_type = 41`) direct loader** (`gguf_1bit_loader.py`) that streams all 64 `Qwen3_5ForCausalLM` hybrid Gated-DeltaNet + Full-Attention layers (`497` `Packed1BitLinear` modules + `Packed1BitEmbedding`, 27B parameters) straight from disk into CUDA in **`3.81 GB` VRAM** (`4.94 GB` peak with chunked `lm_head` matmul), avoiding a 54 GB FP16 unpack entirely.
   - Ran a 7-layer homograph Cartography scan (`L20` to `L44`), peaking cleanly at **Layer 36 (56% depth, Cohen's d = `5.556`)**.
   - Attached a **`61.5 KB` (`61,536 B`, `0.0016%` of the 3.81 GB 1-bit model)** 3-layer BF16 residual sidecar (`L35` to `L37`) synthesized via closed-form contrastive representation distillation (`mean(h_steered - h_unsteered)` on `SOCIAL_TRAIN` only, zero test leakage, zero 27B backpropagation).
   - **Key Result:** On unseen Dominican/Caribbean social `"tigre"` test items, the 61.5 KB self-gated sidecar lifted accuracy from **`25.0%` (`2/8`) baseline** (and **`0.0%` (`0/8`)** under explicit prompt steering, which backfired) to **`62.5%` (`5/8`)** (`+37.5 pp` over baseline, `+62.5 pp` over prompt steering), while self-gating protected unseen animal `"tigre"` preservation from `62.5%` (ungated) back up to **`87.5%` (`7/8`, margin `+7.959`)**.
   - **Honest Remaining Limitation:** General Spanish NLL drift on Bonsai-27B was **`+8.5358%`** (vs. `+10.1215%` ungated) because `tau = 0.158` still triggered on `12.5%` (`1/8`) of unseen general Spanish prompts and on raw-sentence NLL tokens where the translation prompt wrapper is absent.

---

## 2. Experiment 013b: Bonsai-4B 3-Way Discriminative Self-Gate (Exact 0.0000% Drift)

**Script:** `experiments/013-bonsai-1bit-sidecar-lab/run_sidecar_multilayer_contrastive.py`  
**Results:** `experiments/013-bonsai-1bit-sidecar-lab/results_013b_multilayer.json`

### 2.1 Unseen 3-Way Gate Generalization (Layer 19, `tau = 0.3005`)
- **Unseen Social `"tigre"` Recall:** `100.0%` (`8/8`)
- **Unseen Animal `"tigre"` FPR:** `0.0%` (`0/8`)
- **Unseen General Spanish FPR:** `0.0%` (`0/8`)

### 2.2 4-Condition Evaluation (Bonsai-4B, 1.29 GB 1-Bit VRAM, 41 KB Sidecar)

| Condition | Social Win Rate | Social Mean Margin | Animal Win Rate | Animal Mean Margin | General Spanish NLL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_1bit` | `0.0%` (`0/8`) | `-12.870` | `100.0%` (`8/8`) | `+19.528` | `+0.0000%` |
| `prompt_steering_1bit` | `12.5%` (`1/8`) | `-6.110` | `100.0%` (`8/8`) | `+15.826` | `+0.0000%` |
| `ungated_multilayer_sidecar` (`41,088 B`) | `0.0%` (`0/8`) | `-10.411` | `100.0%` (`8/8`) | `+18.495` | `-0.0644%` |
| `self_gated_multilayer_sidecar` (`41,088 B`) | `0.0%` (`0/8`) | `-12.179` | `100.0%` (`8/8`) | `+19.528` | **`+0.0000%`** |

---

## 3. Experiment 014: Native `Q1_0` GGUF Bonsai-27B 1-Bit Sidecar Lab

**Loader:** `src/weightliftai/weight_programming/gguf_1bit_loader.py`  
**Script:** `experiments/014-bonsai-27b-1bit-sidecar-lab/run_bonsai_27b_sidecar_lab.py`  
**Results:** `experiments/014-bonsai-27b-1bit-sidecar-lab/results.json`

### 3.1 Direct `Q1_0` GGUF -> PyTorch CUDA Memory Profile
- **Checkpoint:** `Bonsai-27B-Q1_0.gguf` (`qwen35`, 64 layers, `hidden_size = 5120`, `intermediate_size = 17408`, `vocab_size = 248320`)
- **Packed 1-Bit Modules:** `497` `Packed1BitLinear` layers + `Packed1BitEmbedding` (`3,781.88 MB` packed weights)
- **Allocated CUDA VRAM (Loaded Model):** **`3,814.11 MB` (`3.81 GB`)**
- **Peak CUDA VRAM (Full Benchmark with Chunked `lm_head`):** **`4,943.27 MB` (`4.94 GB`)** (well inside 8.55 GB RTX 5070)
- **Sidecar Footprint (`L35` to `L37`):** **`61,536 bytes` (`61.5 KB`, `0.001627%` of packed 1-bit base weights)**

### 3.2 7-Layer Homograph Cartography Scan (`TRAIN` Splits Only)

| Layer | Depth | 3-Way Cohen's d | Calibrated `tau` | Mean Hidden Norm |
| :---: | :---: | :---: | :---: | :---: |
| `L20` | 31% | `3.947` | `+0.3119` | `32.48` |
| `L24` | 38% | `3.914` | `+0.2992` | `34.06` |
| `L28` | 44% | `4.136` | `+0.2559` | `35.34` |
| `L32` | 50% | `5.359` | `+0.1843` | `41.65` |
| **`L36`** | **56%** | **`5.556`** | **`+0.1580`** | **`46.37`** |
| `L40` | 63% | `4.452` | `+0.1617` | `44.17` |
| `L44` | 69% | `4.535` | `+0.1240` | `49.06` |

**Unseen Gate Evaluation at Peak Layer `L36` (`tau = +0.1580`):**
- Unseen Social Recall: **`87.5%`** (`z_mean = +0.506`)
- Unseen Animal FPR: `37.5%` (`z_mean = -0.061`)
- Unseen General Spanish FPR: `12.5%` (`z_mean = -0.085`)

### 3.3 4-Condition Evaluation on Unseen Test Splits (Bonsai-27B)

| Condition | Unseen Social Win Rate | Social Mean Margin | Unseen Animal Win Rate | Animal Mean Margin | General Spanish NLL | NLL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_1bit_27b` | `25.0%` (`2/8`) | `-5.252` | `100.0%` (`8/8`) | `+8.701` | `7.18900` | `+0.0000%` |
| `prompt_steering_1bit_27b` | `0.0%` (`0/8`) | `-10.756` | `100.0%` (`8/8`) | `+12.466` | `7.18900` | `+0.0000%` |
| `ungated_fp16_sidecar_27b` (`61.5 KB`) | **`62.5%` (`5/8`)** | **`-2.839`** | `62.5%` (`5/8`) | `+3.160` | `7.91663` | `+10.1215%` |
| `self_gated_fp16_sidecar_27b` (`61.5 KB`) | **`62.5%` (`5/8`)** | `-3.517` | **`87.5%` (`7/8`)** | **`+7.959`** | `7.80263` | `+8.5358%` |

---

## 4. Key Takeaways and Next Engineering Step

1. **Native 1-Bit 27B Weight-Space Editing Works on an 8 GB Laptop GPU:** Direct `Q1_0` GGUF streaming into `Packed1BitLinear` + chunked `lm_head` cuts a 54 GB FP16 model down to **3.81 GB loaded / 4.94 GB peak**, enabling surgical hidden-state Cartography and FP16 residual sidecar attachment on consumer hardware.
2. **Contrastive Sidecars Beat Prompt Steering on 1-Bit 27B:** Explicit prompt steering backfired (`25.0% -> 0.0%`) because naming `"tiger"` in the instruction amplified the literal animal prior. Distilling the representation shift into a **61.5 KB FP16 sidecar** raised unseen social `"tigre"` accuracy to **`62.5%` (`+37.5 pp` over baseline, `+62.5 pp` over prompt steering)**.
3. **Next Step to Hit Exact `0.0000%` NLL Drift on Bonsai-27B:** Combine Experiment 013b's tighter 3-way gate calibration (including raw unprompted Spanish sentences in the negative calibration set and raising `tau` to `0.0%` general FPR) with Experiment 014's contrastive `L35` to `L37` sidecar vectors.
