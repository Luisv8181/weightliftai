# Experiment 009: Dialogic Generation Qualitative Example Under Null-Space Patch

**Investigator:** Coding agent (Antigravity, role-play of the Astra protocol; not OpenAI GPT-6 Astra)  
**Date:** 2026-11-07 (Relabeled 2026-11-26)  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 Layers, $d_{model}=896, d_{ff}=4864$)  
**Intervention:** Layer 14 `mlp.down_proj.weight` Rank-5 Joint Subspace Patch (from Exp 008)  
**Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (CUDA)  
**Verdict:** **Inconclusive (n too small)**: Qualitative example (`n=1` conversation with 3 prompts; research-only, not clinical validation)

---

## 1. Objective

To inspect qualitatively how the Layer 14 parameter update $\Delta W \in \mathbb{R}^{896 \times 4864}$ from Experiment 008 affects multi-token greedy decoding on 3 sample Spanish prompts (2 Dominican idiom prompts + 1 factual control). Per `docs/AGENT_HANDOFF.md`, this repo is research-only and makes no clinical efficacy claims.

---

## 2. Qualitative Example Transcripts (`n=1` per prompt)

### Turn 1: Anger Idiom Example (Dominican "pique" and "quilla")
- **Prompt:** *"Llegué al trabajo y mi jefe me culpó de un error suyo delante de todos. Me dio un tremendo pique y una quilla que casi exploto. No sé qué hacer con esta emoción."*
- **Base Model (Unpatched):**
  > *"Es completamente comprensible que sientas tanta emoción después de recibir un golpe de jefe... 1. **Reconozca el error**: Asegúrate de entender cuál fue el error en tu trabajo..."*
  > *Observation:* In this example, the unpatched model produced "golpe de jefe" and suggested "reconozca el error".
- **Patched Model:**
  > *"Es completamente comprensible que sientas tanta emoción cuando te culpa tu jefe por un error. Aquí hay algunas sugerencias para manejar la situación: 1. **Respira profundamente**: Una vez que tengas suficiente aire, respira profundamente. Esto puede ayudar a calmarte..."*
  > *Observation:* In this example, the patched model's reply referenced "cuando te culpa tu jefe" and suggested deep breathing instead of "golpe de jefe".

### Turn 2: Exhaustion Idiom Example (Dominican "desbarata'o")
- **Prompt:** *"Doctor, la verdad es que ya no puedo más. Llevo tres semanas doblando turno en la clínica y estoy desbarata'o de tanto trabajar. Siento que el cuerpo no me responde."*
- **Base Model (Unpatched):**
  > *"1. **Revisar las condiciones laborales**: Puedes hablar con tu jefe o directivo..."*
- **Patched Model:**
  > *"1. **Revisar tus condiciones físicas**: Siempre es importante revisar tu estado físico para asegurarte de que estés trabajando correctamente. Puedes hacer una consulta médica..."*
  > *Observation:* In this example, the patched model's reply shifted from workplace conditions ("condiciones laborales") to physical state ("condiciones físicas... consulta médica").

### Turn 3: Neutral Factual Control (Photosynthesis)
- **Prompt:** *"Explícame en dos oraciones claras qué es la fotosíntesis y cuál es su producto principal."*
- **Base Model:** *"La fotosíntesis es el proceso de convertir los nutrientes orgánicos en energía que se utiliza por las plantas... Su principal producto principal es oxígeno..."*
- **Patched Model:** *"La fotosíntesis es el proceso de convertir los nutrientes orgánicos en energía que se utiliza por las plantas... Su principal producto principal es oxígeno (O2)..."*
- *Observation:* In this single control prompt, the output remained stable.

---

## 3. Limitations

This is a single qualitative example (`n=1` per prompt) on `Qwen/Qwen2.5-0.5B-Instruct`. It demonstrates that the Experiment 008 patch can change multi-token generation without immediate collapse on these three prompts, but does not establish general behavioral or clinical efficacy.
