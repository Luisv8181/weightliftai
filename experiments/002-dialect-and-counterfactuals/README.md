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

### Part A: 3-Way Dialect Representation Geometry (RQ2)

![Figure 3: Representational Geometry of Dialect](../../docs/figures/fig3_dialect_geometry_gap.png)

**Layer 13 Performance** (peak representation depth):

| Metric | en | es_neutral | es_do |
|---|---:|---:|---:|
| 5-fold CV accuracy | 0.875 [0.74, 0.95] | 0.875 [0.74, 0.95] | 0.775 [0.62, 0.88] |
| Zero-shot transfer from en | — | 1.000 [0.91, 1.00] | 0.925 [0.80, 0.97] |
| Item-level cosine with en | — | 0.743 | 0.616 |

- **Persistent dialect gap:** Dominican idioms sit further from English at every layer tested (item cosine `0.616` vs `0.743` at L13).
- **Asymmetric transfer:** en→es_do scores 0.925, but es_do→en scores 0.775.
- **Sample size limitation:** Classification accuracy gaps are not statistically significant at n=40 (confidence intervals overlap).

### Part B: 2×2 Counterfactual Lexical Shortcut Trap (RQ3)

![Figure 2: The Lexical Shortcut Trap](../../docs/figures/fig2_lexical_shortcut_trap.png)

Features #293, #1886, and #4284 fire **as strongly on neutral trap sentences** (a mortician, a probate lawyer, "the battery died") as on real grief. A grief probe flags **10/10 (100%)** trap sentences as grief and catches only **1/10 (10%)** implicit-grief sentences. Zeroing the three features barely helps, showing that the keyword shortcut is distributed across many features in the SAE dictionary. The model's own output shows the same pattern: p(" grief") is ~40× lower after implicit grief than after explicit grief.

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
