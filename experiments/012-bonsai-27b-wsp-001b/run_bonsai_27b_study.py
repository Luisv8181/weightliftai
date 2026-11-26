"""Experiment 012: Bonsai-27B (1-Bit Local Model via Ollama) on Sealed WSP-001B Blind Repair.

Tests Fan et al. (arXiv:2611.21885) 27B self-rewriting scale threshold on a local 1-bit 27B
model (`bonsai-27b-q1:latest` in Ollama, ~55 tok/s, 8 GB VRAM) against the sealed `WSP-001A`
`EvaluatorRegistryAuthority` (private CSPRNG seeds, `CONSUME_BEFORE_VALIDATION` attempt budget,
Ed25519 signed receipts).

Compares 4 methods on the exact same sealed evaluator across 4 defect families x 3 diagnostic
conditions x 2 runs per cell (96 total sealed runs):
1. `bonsai_27b_direct_json`: Direct floating-point JSON `StructuredPatchProposal` (apples-to-apples vs. Qwen-0.5B's 0/40).
2. `bonsai_27b_projection_solver`: Gradient-free `StructuredProjectionPlan` (Fan et al. style activation projection).
3. `deterministic_baseline`: Closed-form deterministic heuristic.
4. `gradient_baseline`: Adam gradient repair upper bound.

Zero files in `bonsai-chat` are touched; communicates read-only with Ollama at 127.0.0.1:11434.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from weightliftai.weight_programming.defect_families import DefectFamily, DiagnosticCondition
from weightliftai.weight_programming.llm_proposer import (
    DeterministicHeuristicProposer,
    GradientRepairProposer,
    LLMWeightRepairProposer,
    OllamaLLMBackend,
    SandboxedProjectionLLMProposer,
)
from weightliftai.weight_programming.proposer_api import BlindProposerEnvironment


def _stats(vals: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_low": round(mean - 1.96 * se, 4),
        "ci95_high": round(mean + 1.96 * se, 4),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ledger_path = out_dir / "bonsai_27b_ledger.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()

    model_id = "bonsai-27b-q1:latest"
    backend = OllamaLLMBackend(
        model_id=model_id,
        base_url="http://127.0.0.1:11434",
        max_new_tokens=450,
        temperature=0.0,
        repeat_penalty=1.15,
        timeout_seconds=120.0,
    )

    proposers = [
        ("bonsai_27b_direct_json", LLMWeightRepairProposer(backend=backend)),
        ("bonsai_27b_projection_solver", SandboxedProjectionLLMProposer(backend=backend)),
        ("deterministic_baseline", DeterministicHeuristicProposer()),
        ("gradient_baseline", GradientRepairProposer(steps=80, lr=0.05)),
    ]

    families = [f.value for f in DefectFamily]
    conditions = [
        DiagnosticCondition.BEHAVIORAL_ONLY.value,
        DiagnosticCondition.WEIGHT_SUMMARIES_ONLY.value,
        DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
    ]
    runs_per_cell = 2

    t_start = time.perf_counter()
    raw_records: list[dict[str, Any]] = []

    for fam in families:
        for cond in conditions:
            for run_idx in range(runs_per_cell):
                for method_label, proposer in proposers:
                    env = BlindProposerEnvironment(
                        defect_family=fam,
                        diagnostic_condition=cond,
                        ledger_path=ledger_path,
                    )
                    rec = proposer.run_on_environment(env)
                    rec["method_label"] = method_label
                    rec["run_index"] = run_idx
                    raw_records.append(rec)
                    status = rec.get("proposal_status")
                    passed = rec.get("passed_gates", False)
                    gain = rec.get("target_held_out_gain", 0.0)
                    op = rec.get("operation_chosen", "-")
                    print(
                        f"[{method_label:28s}] fam={fam:27s} cond={cond:23s} r={run_idx} "
                        f"status={status:24s} op={op:24s} pass={str(passed):5s} gain={gain}"
                    )

    total_elapsed = round(time.perf_counter() - t_start, 2)

    # Aggregate summary by method x condition and overall
    summary_by_method: dict[str, Any] = {}
    for method_label, _ in proposers:
        m_recs = [r for r in raw_records if r["method_label"] == method_label]
        by_cond: dict[str, Any] = {}
        for cond in conditions:
            c_recs = [r for r in m_recs if r["diagnostic_condition"] == cond]
            gains = [float(r.get("target_held_out_gain", 0.0) or 0.0) for r in c_recs]
            passes = [1.0 if r.get("passed_gates", False) else 0.0 for r in c_recs]
            eval_cnt = sum(1 for r in c_recs if r.get("proposal_status") == "EVALUATED")
            by_cond[cond] = {
                "total_runs": len(c_recs),
                "evaluated_runs": eval_cnt,
                "invalid_or_rejected": len(c_recs) - eval_cnt,
                "pass_count": int(sum(passes)),
                "pass_rate": round(float(np.mean(passes)), 4) if passes else 0.0,
                "target_held_out_gain": _stats(gains),
            }
        all_gains = [float(r.get("target_held_out_gain", 0.0) or 0.0) for r in m_recs]
        all_passes = [1.0 if r.get("passed_gates", False) else 0.0 for r in m_recs]
        all_eval = sum(1 for r in m_recs if r.get("proposal_status") == "EVALUATED")
        summary_by_method[method_label] = {
            "total_runs": len(m_recs),
            "evaluated_runs": all_eval,
            "pass_count": int(sum(all_passes)),
            "pass_rate": round(float(np.mean(all_passes)), 4) if all_passes else 0.0,
            "target_held_out_gain": _stats(all_gains),
            "by_condition": by_cond,
        }

    report = {
        "experiment": "012-bonsai-27b-wsp-001b",
        "model_id": model_id,
        "defect_families": families,
        "diagnostic_conditions": conditions,
        "runs_per_cell": runs_per_cell,
        "total_sealed_runs": len(raw_records),
        "total_wall_seconds": total_elapsed,
        "summary_by_method": summary_by_method,
        "raw_records": raw_records,
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== EXPERIMENT 012 COMPLETE ({len(raw_records)} sealed runs in {total_elapsed}s) ===")
    for m, s in summary_by_method.items():
        print(
            f"  {m:30s}: pass={s['pass_count']}/{s['total_runs']} ({s['pass_rate']*100:.1f}%) "
            f"eval={s['evaluated_runs']}/{s['total_runs']} mean_gain={s['target_held_out_gain']['mean']}"
        )
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
