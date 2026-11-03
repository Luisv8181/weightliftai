# Experiment 001 — Emotion feature map in Gemma 3 1B (EN/ES)

**Status:** set up, not yet run on the real model. The notebook's analysis code was tested on synthetic features, and its model cells were tested on a tiny randomly initialized Gemma 3. Neither test produces findings.

## Research question

Does Gemma 3 1B keep closely related emotional concepts distinct inside the model (**grief vs. sadness**, **fear vs. anxiety**)? And does the same distinction hold in **Spanish** as in English?

## Hypotheses (to confirm or reject)

1. Each concept has a small set of sparse-autoencoder features that fire on it noticeably more than on neutral sentences.
2. Grief and sadness share many features; the model separates them less than fear and anxiety.
3. A classifier trained on English feature patterns transfers to Spanish above chance (0.25). If so, part of the representation is language-independent.

## Setup

| Item | Value |
|---|---|
| Base model | `google/gemma-3-1b-it` (record the revision hash when run) |
| Interpretability map | Gemma Scope 2, SAELens release `gemma-scope-2-1b-it-res`, SAE `layer_13_width_16k_l0_medium` (residual stream after layer 13) |
| Feature browser | https://www.neuronpedia.org/gemma-3-1b-it/13-gemmascope-2-res-16k |
| Probe set | `emotion_probe_set_v0.csv`: 5 concepts (sadness, grief, anxiety, fear, neutral) × 2 languages × 10 sentences = 100 |
| Pooling | Max activation per feature over each sentence's tokens, excluding BOS and padding |
| Hardware | Colab T4 (free tier), float32 |
| Weights | Unchanged in sections 1–12; section 13 adds a LoRA adapter as a separate artifact |

## Metrics

- **Concept specificity:** mean activation on the concept minus mean on neutral, per feature.
- **Centroid similarity:** cosine similarity of mean feature vectors, per language (lower = more distinct).
- **Probe accuracy:** 5-fold logistic regression on log-scaled features, with the top 128 features selected inside each fold (chance = 0.25).
- **Cross-lingual transfer:** train on EN, test on ES, and the reverse.
- **SAE fit:** fraction of variance explained. Re-check after fine-tuning to measure map drift.

## How to run

1. Accept the Gemma license at https://huggingface.co/google/gemma-3-1b-it and create a read token.
2. Open `gemma_emotion_sae_map.ipynb` in Colab with a T4 GPU and run the cells in order. The probe set is embedded in the notebook; to swap in a new version, edit the CSV and reload it in section 6.
3. Save `emotion_map_results.csv` and `top_features_by_concept.csv` into `outputs/` here. That folder is git-ignored, so commit a summary table instead.

## Visual outputs (sections 15–16)

Saved to `outputs/` as self-contained HTML files that open offline in any browser:

| File | What it shows |
|---|---|
| `concept_heatmap.html` | Every probe sentence × the top features for each emotion, grouped by emotion and language |
| `word_highlights.html` | Pick a feature; each word glows by how strongly it fires |
| `feature_map.html` | Emotion features as a constellation: similar directions sit together, color = emotion, size = specificity |
| `trace_grief_es.html`, `trace_grief_en.html` | Animated replay of the model answering, with features firing at layers 7, 13, 17 and 22 on every word |

The live trace also streams into the notebook while the model writes. The renderers live in `weightliftai.feature_viz` and the tracing in `weightliftai.feature_trace`, so other experiments can reuse them.

Lines in the trace replay show **co-activation, not causation**. Causal tracing (attribution graphs with Gemma Scope 2 transcoders) is a natural follow-up experiment.

## Run locally (Windows + NVIDIA GPU)

Use Python 3.11 or 3.12, from the repo root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# Install CUDA PyTorch FIRST. Plain `pip install torch` on Windows gives a CPU-only build.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[core,adaptation,interp]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
hf auth login        # paste your Hugging Face read token (accept the Gemma license first)
```

Then open `gemma_emotion_sae_map.ipynb` in VS Code (or `jupyter lab`), choose the `.venv` kernel, **skip sections 0 and 1** (install and login are already done), and run the rest.

- The CUDA check should print `True` and your GPU name. If it prints `False`, reinstall torch with the line above. If `cu128` doesn't match your driver, pick the matching command at https://pytorch.org/get-started/locally/.
- Gemma 3 1B in float32 needs about 5 GB of VRAM. With 12 GB or more, you can try the 4B model (settings are in the notebook's last cell) with `dtype=torch.bfloat16`.
- With no GPU, it still runs on CPU. Sections 7 and 10 take a few minutes, and LoRA training is slow.
- Put outputs in `experiments/001-gemma-emotion-sae-map/outputs/` (git-ignored).

## Known limitations

- **Small probe set.** 20 sentences per concept is exploratory. Before drawing conclusions, expand to 100+ per concept per language, with native-speaker and clinical review.
- **Dialect.** The Spanish sentences lean Dominican (*habichuelas*, *guagua*, *colmado*). That is intentional, but it confounds dialect with language. A neutral-Spanish split would separate the two.
- **Features are not concepts.** An SAE feature that fires on grief sentences may track funerals, family words, or past tense rather than grief itself. Check every candidate on Neuronpedia and with `token_view` before naming it.
- **LoRA placeholder data.** The six training pairs in section 13 only show the format. They cannot support a claim about improvement.
- **SAE drift.** The published SAE was trained on the unmodified model. After fine-tuning, a lower variance explained means the map is less trustworthy.

## Results

| Metric | Baseline | After LoRA | Difference |
|---|---:|---:|---:|
| SAE variance explained | | | |
| grief~sadness similarity (EN / ES) | | | |
| fear~anxiety similarity (EN / ES) | | | |
| Probe accuracy EN / ES | | | |
| Transfer EN→ES / ES→EN | | | |

*Not yet run.*
