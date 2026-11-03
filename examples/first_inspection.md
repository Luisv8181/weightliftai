# First local inspection

Install the project and its core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
pip install -e ".[core,dev]"
```

Check hardware:

```bash
weightliftai hardware
```

Inspect configuration only (does not load model weights):

```bash
weightliftai inspect google/gemma-3-1b-it --manifest experiments/gemma-baseline/manifest.json
```

Inspect parameter tensors (loads the model; memory requirements vary):

```bash
weightliftai inspect google/gemma-3-1b-it --load-weights
```

For a fully offline run, use a model already present in the Hugging Face cache:

```bash
weightliftai inspect /path/to/local/model --local-files-only
```

Do not enable `--trust-remote-code` for repositories you have not reviewed.
