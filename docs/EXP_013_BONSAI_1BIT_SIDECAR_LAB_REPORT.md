# Experiment 013: 1-Bit Sidecar Lab on PyTorch PrismML Bonsai-4B

**Repository:** `Luisv8181/weightliftai`  
**Date:** 2026-11-26  
**Model:** `prism-ml/Bonsai-4B-unpacked` (36-layer Qwen3 architecture, `hidden_size=2560`, native 1-bit weights)  
**Hardware:** `NVIDIA GeForce RTX 5070 Laptop GPU` (`8.55 GB` VRAM)  
**Implementation:** `src/weightliftai/weight_programming/onebit_sidecar.py` (`Packed1BitLinear`, `SelfGatedFP16Sidecar`)  
**Raw Results:** `experiments/013-bonsai-1bit-sidecar-lab/results.json`

---

## 1. Objective

Test whether a native **1-bit LLM** (`prism-ml/Bonsai-4B-unpacked`) can be:
1. Packed in PyTorch into true **1-bit `uint8` bit-planes + BF16 group scales** (`group_size=128`) to fit comfortably inside 8.55 GB VRAM without losing its native 1-bit weights,
2. Mapped across all 36 layers via **100% homograph `"tigre"` Cartography** (`experiments/004-cartography-and-gated-patch/homograph_data.py`), and
3. Patched using a **tiny FP16 self-gated residual sidecar (`10.0 KB`)** attached to a frozen 1-bit middle layer without degrading literal-animal controls or general Spanish fluency.

---

## 2. Phase 1: Native 1-Bit Packing Verification (`Packed1BitLinear`)

| Metric | Measured Value |
| :--- | :---: |
| Transformer `nn.Linear` Layers Packed | **252 / 252** |
| Linear Weight Memory Before Packing (BF16) | `7,266.63 MB` (`7.27 GB`) |
| Linear Weight Memory After 1-Bit Packing (`uint8` + BF16 scales) | **`510.94 MB` (`0.51 GB`)** |
| Linear Weight Compression Ratio | **`14.22x`** |
| Mean / Max Weight Reconstruction Error | `0.000244` / `0.001953` |
| Total GPU VRAM Allocated (Packed Body + Embeddings + `lm_head`) | **`1,293.94 MB` (`1.29 GB`)** |
| Sidecar Footprint (`SelfGatedFP16Sidecar`, `d=2560`, FP16) | **`10,272 bytes` (`10.0 KB`)** |

---

## 3. Phase 2: 36-Layer Homograph Cartography on 1-Bit Bonsai-4B

Even under extreme 1-bit weight quantization, Bonsai-4B develops a sharp linear separation between the Dominican social sense and the literal zoological sense of the exact homograph `"tigre"` in its middle layers:

- **Best Cartography Layer:** **Layer `19`** (`53%` depth)
- **Train Cohen's $d$:** **`7.6866`**
- **Unseen Test Probe Accuracy:** **`87.5%`**
- **Social Gate Recall ($\tau = 0.25$):** **`87.5%`** (`7/8`)
- **Literal Animal Gate False-Positive Rate ($\tau = 0.25$):** **`0.0%`** (`0/8`)
- **General Spanish Control Gate FPR ($\tau = 0.25$):** `50.0%` (`4/8`, indicating that a 3-class or higher-threshold gate is needed to reach exact `0.00%` general Spanish drift).

---

## 4. Phase 3: 4-Arm Controlled Evaluation on Unseen Homograph Test Split

| Arm | Social Win Rate (`8` unseen) | Social Mean Margin ($\Delta \log P$) | Animal Win Rate (`8` unseen) | Animal Mean Margin ($\Delta \log P$) | General Spanish NLL Drift | Sidecar Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_1bit`** | `0.000` (`0/8`) | `-12.870` | `1.000` (`8/8`) | `+19.528` | `0.0000%` | `0 B` |
| **`prompt_steering_1bit`** | `0.125` (`1/8`) | `-6.110` | `1.000` (`8/8`) | `+15.826` | `0.0000%` | `0 B` |
| **`ungated_fp16_sidecar` (L19, $\alpha=1.0$)** | `0.000` (`0/8`) | `-11.806` (`+1.06`) | `1.000` (`8/8`) | `+19.262` (`-0.27`) | `+1.3464%` | `10.0 KB` |
| **`self_gated_fp16_sidecar` (L19, $\alpha=1.0$)** | `0.000` (`0/8`) | `-12.343` (`+0.53`) | **`1.000` (`8/8`)** | **`+19.528` (`0.000` drift)** | **`+0.0485%`** *(28x lower)* | **`10.0 KB`** |

---

## 5. Honest Conclusions & Next Steps

1. **What Worked (Tier 1 & Tier 2 Verified):**
   - **1-Bit Packing Works Cleanly on RTX 5070:** Packing all 252 linear layers of `Bonsai-4B-unpacked` into `Packed1BitLinear` shrunk linear weight memory `14.22x` (`7.27 GB -> 0.51 GB`, total GPU VRAM `1.29 GB`) with `0.000244` mean error.
   - **1-Bit Linear Concept Geometry Is Intact:** Layer 19 achieved Cohen's $d = 7.69$, `87.5%` unseen test accuracy, and `0.0%` animal false positives.
   - **Self-Gating Cuts Collateral Drift by 28x:** The `10 KB` FP16 self-gated sidecar reduced general Spanish NLL drift from `+1.3464%` (`ungated`) down to `+0.0485%` while keeping literal animal margin bit-identical (`+19.528`).
2. **What Did Not Flip Yet (Preserved Partial/Negative Result):**
   - At `alpha=1.0` on a single layer (`L19`), the directional shift moved the social margin positively (`-12.870 -> -12.343`, `+0.53`), but did not overcome Bonsai-4B's very strong literal `"tiger"` prior (`-12.87` log-prob gap; even explicit prompt steering only reached `1/8` win rate at `-6.11`).
3. **Clear Next Iteration:**
   - Optimize `u_edit` / `alpha` directly on the contrastive social log-prob margin across Layers `18–22` (still `< 60 KB` total FP16 sidecar) and include `GENERAL_SPANISH_TRAIN` as negative examples when fitting `w_sense` so general Spanish FPR drops from `50%` to `0.0%` (`0.0000%` NLL drift).
