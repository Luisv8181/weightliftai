# Experiment 002: Dialect split and counterfactual controls (Gemma 3 1B)

**Status:** Executed 2026-11-04 on an RTX 5070 Laptop GPU (36.9 s). Full write-up in [`docs/ROLLING_PAPER.md` §5](../../docs/ROLLING_PAPER.md).

## Questions

- **Part A (RQ2):** Do Dominican Spanish idioms of distress reach the same middle-layer emotion representation as English and pan-Hispanic Spanish?
- **Part B (RQ3):** Do Experiment 001's "grief features" (#293, #1886, #4284) encode grief, or the words that tend to appear with it (family, death, "still/todavía")?

## Data: `dialect_and_counterfactual_probes_v0.csv` (160 rows)

| Split | Design | Rows |
|---|---|---:|
| `dialect_parallel` | 40 items (10 each: sadness, grief, anxiety, fear), each written as `en`, `es_neutral` (pan-Hispanic), and `es_do` (Dominican idioms: *bajito*, *dándome mente*, *los nervios de punta*, *guagua*, *colmado*, *me cuelgan*, *candela*…) | 120 |
| `counterfactual_2x2` | `explicit_grief` · `implicit_grief` (no death/family/"still" words) · `lexical_trap_neutral` (no grief, many of those words) · `baseline_neutral`; 5 en + 5 es per cell | 40 |

> ⚠️ Written with AI assistance. **Not yet reviewed by native Dominican speakers or clinicians.** Some idioms may be pan-Caribbean rather than specifically Dominican.

## Headline results

**Part A, layer 13** (best layer for every variant):

| | en | es_neutral | es_do |
|---|---:|---:|---:|
| 5-fold CV accuracy | 0.875 | 0.875 | 0.775 |
| Transfer from en | — | 1.000 | 0.925 |
| Item cosine with en | — | 0.743 | 0.616 |

- Dominican idioms sit further from English at every layer tested.
- Transfer is asymmetric: en→es_do 0.925, but es_do→en 0.775.
- Accuracy gaps are **not significant at n=40**.

**Part B:** Features #293, #1886, and #4284 fire **as strongly on neutral trap sentences** (a mortician, a probate lawyer, "the battery died") as on real grief. A grief probe flags **10/10** trap sentences as grief and catches only **1/10** implicit-grief sentences. Zeroing the three features barely helps, so the keyword shortcut is spread across many features. The model's own output shows the same pattern: p(" grief") is ~40× lower after implicit grief than after explicit grief.

**Withdrawn claim:** "#293/#1886/#4284 are grief features" (Exp. 001). They are keyword features.

## Files

| File | Contents |
|---|---|
| `run_experiment_002.py` | Builds the CSV, runs Parts A and B, writes `outputs/` |
| `results/experiment_002_summary.json` | All metrics plus provenance (dataset SHA-256, runtime, versions) |
| `results/dialect_layer_metrics.csv` | Part A table across layers 7/13/17/22 |
| `results/counterfactual_key_features.csv` | Part B.1 feature activations per cell |
| `results/true_grief_features.csv` | Part B.2 candidate keyword-independent features (**selection-biased; hypotheses only**) |

## Reproduce

```bash
python experiments/002-dialect-and-counterfactuals/run_experiment_002.py
```

Requires the Gemma license and `pip install -e ".[core,interp]"`. Uses about 6 GB of VRAM in float32.

## Next: Experiment 002b (pre-registered in the paper, §10)

- Test candidate features #602/#701 on new held-out implicit-grief and trap sentences.
- Bootstrap a confidence interval for the dialect cosine gap.
- Add a random-feature control for the causal clamp.
