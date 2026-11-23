"""Execute WSP-001B (Prompts 7-10) Empirical Ablation & Method Comparison Study.

Evaluates:
1. Real Local LLM (`Qwen/Qwen2.5-0.5B-Instruct`) via `LLMWeightRepairProposer`
2. `GradientRepairProposer`
3. `DeterministicHeuristicProposer`

Across all 4 defect families (`fc2_row0_zero_mask`, `fc2_row0_sign_inversion`,
`fc2_row0_scale_attenuation`, `fc1_hidden_corrupt_channel`) and all 5 diagnostic
information conditions (`behavioral_only`, `weight_summaries_only`,
`weights_and_failures`, `weights_activations_failures`, `full_diagnostic_toolkit`),
using evaluator-authoritative private seeds (`EvaluatorRegistryAuthority` + Ed25519 receipts).
"""

from __future__ import annotations

import json
from pathlib import Path

from weightliftai.weight_programming.ablation_study import (
    run_ablation_and_comparison_study,
)
from weightliftai.weight_programming.llm_proposer import LocalTransformersLLMBackend


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "wsp_001b_ledger.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()

    print("Loading LocalTransformersLLMBackend (Qwen/Qwen2.5-0.5B-Instruct)...")
    llm_backend = LocalTransformersLLMBackend(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=180,
    )

    print("Running WSP-001B ablation and baseline comparison across 4 defect families x 5 conditions...")
    report = run_ablation_and_comparison_study(
        llm_backend=llm_backend,
        runs_per_cell=2,
        ledger_path=ledger_path,
    )
    report["model_id"] = llm_backend.model_id

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved empirical results to {results_path}")


if __name__ == "__main__":
    main()
