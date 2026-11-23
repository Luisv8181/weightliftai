# Experiment 003 Results — Rank-1 Weight Patch vs LoRA for Dominican *tíguere / tigre*

- **Date (UTC):** 2026-11-07T06:11:48Z
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (fp32, CUDA on NVIDIA GeForce RTX 5070 Laptop GPU)
- **Seeds:** `0, 1, 2` (`STEPS = 80`)
- **Selected Layer (by seed-0 training loss only):** Layer 12 (`{6: 0.0126, 12: 0.0040, 18: 0.1706}`)
- **Pre-registered Overall Verdict:** **FAIL (2 of 4 criteria passed)**

---

## 1. Quantitative Summary (Held-Out Test Splits)

| Condition | Trainable Params | Held-Out Slang Win Rate ($n=10$) | Ambiguous Spelling (*tigre*) Win Rate ($n=3$) | Held-Out Slang Margin Mean [95% CI] | Held-Out Animal Win Rate ($n=8$) | Regression NLL (Rel. Change) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** (frozen) | 0 | 0.200 | 0.000 | -6.09 [-9.76, -2.76] | 1.000 | 0.1522 (0.0%) |
| **`random_rank1`** (norm-matched control) | 0 (random $\Delta W$) | 0.200 | 0.000 | -6.10 [-9.77, -2.77] | 1.000 | 0.1527 (+0.3%) |
| **`rank1_patch`** (Layer 12 `down_proj`, $\|\Delta W\|_F = 1.902$) | 5,760 | **0.633** (0.6, 0.6, 0.7) | **0.000** | **+1.37** [-3.68, +6.00] | **1.000** | **0.2756 (+81.1%)** |
| **`lora_r4`** (all `q_proj`, `v_proj`) | 270,336 | 0.833 (0.8, 0.9, 0.8) | 0.444 | +16.94 [+8.85, +24.35] | 1.000 | 0.5156 (+238.8%) |

*Note on weight folding:* Folding $\Delta W = u v^T$ directly into `model.layers[12].mlp.down_proj.weight` matched the forward-hook logits to within $\max |\Delta \text{logit}| \le 7.44 \times 10^{-5}$ across all 3 seeds.

---

## 2. Pre-Registered Criteria Check (`rank1_patch`)

| Criterion | Threshold | Measured | Pass? |
| :--- | :--- | :--- | :---: |
| Held-out slang win rate | $\ge 0.80$ | $0.633$ | **FAIL** |
| Held-out animal win rate | $\ge 0.875$ | $1.000$ | **PASS** |
| Regression NLL relative increase | $\le +5.0\%$ | $+81.1\%$ | **FAIL** |
| Slang margin 95% CI lower bound $>$ `random_rank1` mean | $> -6.10$ | $-3.68 > -6.10$ | **PASS** |

---

## 3. Greedy Decoding Inspection (Seed 0)

| Held-Out Input | `baseline` | `rank1_patch` | `lora_r4` |
| :--- | :--- | :--- | :--- |
| *"Yo siento como que mi novia me dejó por un tigre."* | "I feel as if my girlfriend has let a lion out of her cage." | "I feel as if my girlfriend is cheating on me with a tiger." | "I feel like I betrayed her like she left a tiger for me." |
| *"Ella se fue con un tigre que tiene un carro nuevo."* | "She went with a lion that has a new car." | "Eli was going to go home with a new tiger and his new car." | "She went with a tiger that has a new car." |
| *"Mi mamá dice que mi exnovia anda con un tigre."* | "My mommy says that my ex-niece is with a lion." | "My mom says my exwife is dating a tiger." | "My mother says that my girlfriend plays with a tiger." |
| *"Vimos un tigre enorme en el zoológico."* | "We saw a huge lion in the zoo." | "We saw a big cat in the zoo." | "We watched an enormous tiger in the zoo." |
| *"El tigre rugió dentro de la jaula."* | "The lion roared inside the cage." | "The tiger played inside the cage." | "The tiger shouted inside the cage." |

---

## 4. What This Experiment Shows

1. **The rank-1 patch learned a surface-form shortcut, not contextual disambiguation:**
   It improved forced-choice margins on items spelled *tíguere* / *tígueres* (7 of 10 test items, raising overall win rate from `0.20` to `0.633`), but achieved **`0.000` win rate** on the 3 held-out slang items spelled *tigre* (*"mi novia me dejó por un tigre"*). In free greedy generation, it still produced *"cheating on me with a tiger"*.
2. **Even a 5%-norm single-layer rank-1 edit caused +81.1% NLL degradation on unrelated sentences:**
   Unregularized gradient steps on 14 small-corpus examples shifted general output distributions substantially (and 80-step LoRA degraded unrelated NLL by +238.8%).
3. **Implication for Optimust Connect / DISCORS-S:**
   A rank-1 MLP weight patch on a 0.5B model **cannot** replace the explicit symbolic **Life Language Glossary (`GlossaryPreserveSet`)** for polysemous dialect idioms like *tigre*. Symbolic lookup/preservation + contextual notes remain necessary for reliable clinical translation and meaning preservation.
