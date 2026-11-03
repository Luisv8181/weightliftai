# Activation exploration

Activation analysis captures selected module outputs for one prompt and reports descriptive statistics. It does not automatically explain what a neuron means.

Install dependencies:

```bash
pip install -e ".[core,dev]"
```

List leaf modules whose names contain `layers.0`:

```bash
weightliftai modules google/gemma-3-1b-it --contains layers.0
```

Capture a selected module output:

```bash
weightliftai activations google/gemma-3-1b-it \
  --text "Translate 'me hace falta' into natural English." \
  --module "model.layers.0.self_attn"
```

Module names vary by architecture. Use the `modules` command first and copy an exact name. Start with one module; loading a model and running a forward pass still consumes RAM/VRAM.

Keep prompts non-sensitive. Activation summaries can still reveal information about the input, even when raw tensors are not saved.
