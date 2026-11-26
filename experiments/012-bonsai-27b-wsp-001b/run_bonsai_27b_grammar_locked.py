"""Experiment 012c: Dynamic JSON-Schema Grammar Locking on `bonsai-27b-q1:latest`.

Keeps ALL 4 operations open in `StructuredProjectionPlan.model_json_schema()` (nothing given away!)
so Ollama's GBNF sampler enforces syntax validity while Bonsai-27B makes its own diagnosis and
tool choice on `full_diagnostic_toolkit` across all 4 defect families x 2 runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from weightliftai.weight_programming.defect_families import DefectFamily, DiagnosticCondition
from weightliftai.weight_programming.llm_proposer import (
    DeterministicHeuristicProposer,
    GradientRepairProposer,
    OllamaLLMBackend,
    SandboxedProjectionLLMProposer,
    StructuredProjectionPlan,
)
from weightliftai.weight_programming.proposer_api import BlindProposerEnvironment


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ledger_path = out_dir / "bonsai_27b_012c_ledger.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()

    # Pass full dynamic Pydantic schema with ALL 4 operations open
    full_schema = StructuredProjectionPlan.model_json_schema()
    backend = OllamaLLMBackend(
        model_id="bonsai-27b-q1:latest",
        base_url="http://127.0.0.1:11434",
        max_new_tokens=450,
        temperature=0.0,
        repeat_penalty=1.15,
        think=False,
        json_schema=full_schema,
        timeout_seconds=120.0,
    )

    proposers = [
        ("bonsai_27b_grammar_locked", SandboxedProjectionLLMProposer(backend=backend)),
        ("deterministic_baseline", DeterministicHeuristicProposer()),
        ("gradient_baseline", GradientRepairProposer(steps=80, lr=0.05)),
    ]

    families = [f.value for f in DefectFamily]
    cond = DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value
    runs_per_cell = 2

    t0 = time.perf_counter()
    raw_records: list[dict[str, Any]] = []

    for fam in families:
        for run_idx in range(runs_per_cell):
            for label, prop in proposers:
                env = BlindProposerEnvironment(
                    defect_family=fam,
                    diagnostic_condition=cond,
                    ledger_path=ledger_path,
                )
                rec = prop.run_on_environment(env)
                rec["method_label"] = label
                rec["run_index"] = run_idx
                raw_records.append(rec)
                print(
                    f"[{label:28s}] fam={fam:27s} r={run_idx} "
                    f"status={rec.get('proposal_status'):24s} op={str(rec.get('operation_chosen', '-')):24s} "
                    f"pass={str(rec.get('passed_gates', False)):5s} gain={rec.get('target_held_out_gain', 0.0)}"
                )

    elapsed = round(time.perf_counter() - t0, 2)
    summary: dict[str, Any] = {}
    for label, _ in proposers:
        recs = [r for r in raw_records if r["method_label"] == label]
        gains = [float(r.get("target_held_out_gain", 0.0) or 0.0) for r in recs]
        passes = sum(1 for r in recs if r.get("passed_gates", False))
        evals = sum(1 for r in recs if r.get("proposal_status") == "EVALUATED")
        summary[label] = {
            "total_runs": len(recs),
            "evaluated_runs": evals,
            "pass_count": passes,
            "pass_rate": round(passes / len(recs), 4) if recs else 0.0,
            "mean_held_out_gain": round(float(np.mean(gains)), 4) if gains else 0.0,
        }

    out_path = out_dir / "results_012c_grammar_locked.json"
    out_path.write_text(
        json.dumps(
            {
                "experiment": "012c-bonsai-27b-grammar-locked-full-toolkit",
                "total_wall_seconds": elapsed,
                "summary": summary,
                "raw_records": raw_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n=== EXP 012C COMPLETE ({len(raw_records)} sealed runs in {elapsed}s) ===")
    for label, s in summary.items():
        print(f"  {label:30s}: pass={s['pass_count']}/{s['total_runs']} ({s['pass_rate']*100:.1f}%) eval={s['evaluated_runs']}/{s['total_runs']} mean_gain={s['mean_held_out_gain']}")


if __name__ == "__main__":
    main()
