# Experiment Protocol

Every experiment must be reproducible and comparable.

## Required record

- Experiment ID and timestamp
- Research question and hypothesis
- Base model name, revision, and source
- Tokenizer and revision
- Dataset name/version and split
- Hardware, OS, Python, PyTorch, Transformers versions
- Precision / quantization settings
- Random seeds
- Intervention type and full parameters
- Baseline and candidate metrics
- Runtime, peak memory, and latency where available
- Regression findings
- Artifact locations and checksums

## Procedure

1. Define one measurable behavior before changing the model.
2. Create a baseline on fixed development and held-out sets.
3. Inspect failure examples and check whether the failure is actually caused by model behavior rather than prompt, tokenizer, or data issues.
4. Choose the least invasive intervention that tests the hypothesis.
5. Save all candidate artifacts separately from the base model.
6. Run the same evaluation settings on baseline and candidate.
7. Run regression checks across unrelated tasks.
8. Repeat with multiple seeds or samples when appropriate.
9. Report uncertainty and failures; do not cherry-pick examples.
10. Retain or reject the intervention based on predeclared criteria.

## Minimum reporting table

| Metric | Baseline | Candidate | Difference |
|---|---:|---:|---:|
| Target-task score | | | |
| Held-out score | | | |
| Regression suite | | | |
| Median latency | | | |
| Peak memory | | | |

## Acceptance rule

A candidate is not an improvement merely because one example looks better. It must improve the predeclared target metric on held-out data without unacceptable regressions. Report trade-offs rather than collapsing everything into one score.

## Privacy

Use public, synthetic, or explicitly authorized data. Do not place confidential client, patient, student, or other identifiable records into model experiments. Keep prompts and outputs local by default and scrub sensitive data from logs.
