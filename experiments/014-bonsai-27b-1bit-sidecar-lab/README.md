# Experiment 014: Native `Q1_0` GGUF Bonsai-27B Activation Sidecar Lab

## Prerequisites & Download Instructions

1. **Download the 1-bit GGUF checkpoint (`Bonsai-27B-Q1_0.gguf`):**
   Download PrismML's `Bonsai-27B-Q1_0.gguf` from Hugging Face (`prism-ml/Bonsai-27B-gguf` or via Ollama `ollama pull hf.co/prism-ml/Bonsai-27B-gguf:Q1_0`) and place it in a local models directory.
2. **Configure environment variables (no hardcoded local paths required):**
   - `BONSAI_27B_GGUF`: Path to `Bonsai-27B-Q1_0.gguf` (default: `./models/Bonsai-27B-Q1_0.gguf`).
   - `BONSAI_27B_TOKENIZER`: Hugging Face tokenizer repo ID (default: `HAL-X-AI/THX-02`, verified against the GGUF embedded vocabulary).
   - `HF_HUB_CACHE` / `BONSAI_HF_CACHE`: Optional local Hugging Face cache directory (default: standard Hugging Face cache).

## Running

```bash
export BONSAI_27B_GGUF=/path/to/Bonsai-27B-Q1_0.gguf
python experiments/014-bonsai-27b-1bit-sidecar-lab/smoke_test_27b.py
python experiments/014-bonsai-27b-1bit-sidecar-lab/run_bonsai_27b_sidecar_lab.py
```

> **Validation Status:** See `docs/EXP_014_BONSAI_27B_1BIT_SIDECAR_LAB_REPORT.md` for current `Q1_0` parity validation status against `llama.cpp` / Ollama.
