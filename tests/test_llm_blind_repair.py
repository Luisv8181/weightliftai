"""Tests for Phase B (WSP-001B, Prompts 7–10): Real LLM Blind Repair, 5-Condition Ablation, and Baseline Comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weightliftai.weight_programming.ablation_study import (
    analyze_incremental_weight_value,
    run_ablation_and_comparison_study,
)
from weightliftai.weight_programming.defect_families import DefectFamily, DiagnosticCondition
from weightliftai.weight_programming.llm_proposer import (
    DeterministicHeuristicProposer,
    GradientRepairProposer,
    LLMWeightRepairProposer,
    LocalTransformersLLMBackend,
    build_diagnostic_prompt_for_condition,
    parse_and_validate_llm_proposal,
)
from weightliftai.weight_programming.proposer_api import BlindProposerEnvironment
from weightliftai.weight_programming.schemas import AllowlistViolationError, BudgetViolationError


def test_parse_and_validate_llm_proposal_accepts_valid_and_rejects_invalid(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)

    # 1. Valid col_deltas JSON
    valid_json = json.dumps(
        {
            "target_tensor": "fc2.weight",
            "row_index": 0,
            "col_deltas": [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            "hypothesis": "Restore zeroed incoming connections on row 0.",
            "rationale": "First four columns of row 0 in fc2.weight were zeroed.",
        }
    )
    patch = parse_and_validate_llm_proposal(
        raw_llm_text=f"```json\n{valid_json}\n```",
        env=env,
        proposed_by="test_llm",
        patch_id="test-valid-1",
    )
    assert patch.deltas["fc2.weight"].shape == (2, 8)
    assert np.allclose(patch.deltas["fc2.weight"][0, :4], 0.25)
    assert np.allclose(patch.deltas["fc2.weight"][1, :], 0.0)

    # 2. Unauthorized tensor name -> AllowlistViolationError
    bad_tensor = json.dumps(
        {
            "target_tensor": "fc1.weight",
            "row_index": 0,
            "col_deltas": [0.1, 0.1, 0.1, 0.1],
            "hypothesis": "Edit fc1.",
            "rationale": "Testing allowlist rejection.",
        }
    )
    with pytest.raises(AllowlistViolationError):
        parse_and_validate_llm_proposal(bad_tensor, env, "test_llm", "bad-tensor")

    # 3. Out-of-bounds index edit -> ValueError
    bad_idx = json.dumps(
        {
            "target_tensor": "fc2.weight",
            "index_edits": [{"row": 5, "col": 99, "delta": 0.2}],
            "hypothesis": "Bad index.",
            "rationale": "Testing index bounds.",
        }
    )
    with pytest.raises(ValueError, match="out of bounds"):
        parse_and_validate_llm_proposal(bad_idx, env, "test_llm", "bad-idx")

    # 4. Budget violation (L2 norm > 5.0) -> BudgetViolationError
    over_budget = json.dumps(
        {
            "target_tensor": "fc2.weight",
            "row_index": 0,
            "col_deltas": [10.0] * 8,
            "hypothesis": "Huge norm.",
            "rationale": "Exceeds L2 budget.",
        }
    )
    with pytest.raises(BudgetViolationError):
        parse_and_validate_llm_proposal(over_budget, env, "test_llm", "over-budget")

    # 5. Malformed JSON -> ValueError
    with pytest.raises(ValueError, match="Malformed JSON|does not contain a JSON object"):
        parse_and_validate_llm_proposal("Not a JSON response at all", env, "test_llm", "malformed")


def test_llm_proposer_records_failures_without_fallback(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(seed=42, ledger_path=ledger)

    # Backend returning malformed output must record INVALID_PROPOSAL_SCHEMA and passed_gates=False
    bad_proposer = LLMWeightRepairProposer(
        backend=lambda _prompt: {
            "raw_text": "I cannot repair this model.",
            "model_id": "test-bad-llm",
            "model_revision": "v1",
            "decoding_config": {"temperature": 0.0},
            "input_token_count": 100,
            "output_token_count": 6,
        }
    )
    rec = bad_proposer.run_on_environment(env)
    assert rec["proposal_status"] == "INVALID_PROPOSAL_SCHEMA"
    assert rec["passed_gates"] is False
    assert rec["evaluation_result"] is None


def test_five_condition_prompts_prevent_information_leakage(tmp_path: Path):
    ledger = tmp_path / "ledger.sqlite3"
    for cond in DiagnosticCondition:
        env = BlindProposerEnvironment(diagnostic_condition=cond.value, ledger_path=ledger)
        p_bundle = build_diagnostic_prompt_for_condition(env)
        text = p_bundle["prompt_text"]

        # Never leak private seeds, clean weights, or oracle deltas in any condition
        assert "private_seed" not in text
        assert "clean_state_dict" not in text
        assert "oracle_delta" not in text
        assert "defect_mask" not in text

        if cond == DiagnosticCondition.BEHAVIORAL_ONLY:
            assert "weight_summaries" not in text
            assert "activation_summaries" not in text
            assert "toolkit_sensitivity" not in text
            assert "behavioral_failures" in text
        elif cond == DiagnosticCondition.WEIGHT_SUMMARIES_ONLY:
            assert "behavioral_failures" not in text
            assert "activation_summaries" not in text
            assert "toolkit_sensitivity" not in text
            assert "weight_summaries" in text


def test_real_local_llm_proposer_end_to_end(tmp_path: Path):
    """Run an actual un-mocked local LLM (`Qwen/Qwen2.5-0.5B-Instruct`) end-to-end through the sealed evaluator (Prompt 7)."""
    ledger = tmp_path / "ledger.sqlite3"
    env = BlindProposerEnvironment(
        defect_family=DefectFamily.FC2_ROW0_ZERO_MASK.value,
        diagnostic_condition=DiagnosticCondition.FULL_DIAGNOSTIC_TOOLKIT.value,
        ledger_path=ledger,
    )
    backend = LocalTransformersLLMBackend(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=220,
        temperature=0.0,
    )
    proposer = LLMWeightRepairProposer(backend=backend)
    record = proposer.run_on_environment(env)

    if record.get("proposal_status") == "PROVIDER_ERROR":
        pytest.skip(
            f"Local model weights not present or failed to load in this environment: {record.get('error_message')}"
        )

    assert record["method"] == "llm_proposer"
    assert record["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert record["prompt_version"] == "wsp-llm-proposer-v1.0"
    assert len(record["prompt_sha256"]) == 64
    assert record["input_token_count"] > 50
    assert record["output_token_count"] > 5
    assert record["proposal_status"] in {"EVALUATED", "INVALID_PROPOSAL_SCHEMA", "OUT_OF_BUDGET_OR_ALLOWLIST"}
    if record["proposal_status"] == "EVALUATED":
        assert len(record["patch_sha256"]) == 64
        assert record["evaluation_result"]["signed_receipt"]["experiment_id"] == env.experiment_id


def test_ablation_and_comparison_study_across_families_and_conditions(tmp_path: Path):
    """Verify paired ablation and 3-method comparison study runner (Prompts 8, 9, 10)."""
    ledger = tmp_path / "ledger.sqlite3"

    def _structured_diagnostic_llm_reader(prompt_text: str) -> dict[str, Any]:
        # Parses only the authorized JSON inside prompt_text to test how an ideal JSON-following
        # LLM responds under each of the 5 diagnostic conditions without any hidden access.
        marker = "Authorized Diagnostic Package:\n"
        end_marker = "\n\nRespond with ONLY"
        s_idx = prompt_text.find(marker) + len(marker)
        e_idx = prompt_text.find(end_marker)
        diag = json.loads(prompt_text[s_idx:e_idx])

        deltas = [0.05] * 8
        if "toolkit_sensitivity" in diag:
            neg_grad = diag["toolkit_sensitivity"]["negative_gradient_direction_row0"]
            deltas = [round(float(g) * 0.35, 4) for g in neg_grad]
        elif "activation_summaries" in diag:
            proj = diag["activation_summaries"]["diagnostic_hidden_residual_projection"]
            deltas = [round(float(p) * 0.5, 4) for p in proj]
        elif "weight_summaries" in diag:
            w = diag["weight_summaries"]["fc2.weight"]["rounded_slice"]
            r0, r1 = w[0], w[1]
            deltas = [0.35 if abs(x) < 1e-4 else round(-0.2 * x, 4) for x in r0]

        proposal = {
            "target_tensor": "fc2.weight",
            "row_index": 0,
            "col_deltas": deltas,
            "hypothesis": f"Condition-grounded repair under {diag['diagnostic_condition']}.",
            "rationale": "Computed from authorized diagnostic package fields.",
        }
        return {
            "raw_text": json.dumps(proposal),
            "model_id": "diagnostic-schema-follower-v1",
            "model_revision": "1.0",
            "decoding_config": {"temperature": 0.0},
            "input_token_count": len(prompt_text.split()),
            "output_token_count": 42,
        }

    study = run_ablation_and_comparison_study(
        llm_backend=_structured_diagnostic_llm_reader,
        defect_families=[
            DefectFamily.FC2_ROW0_ZERO_MASK.value,
            DefectFamily.FC2_ROW0_SCALE_ATTENUATION.value,
        ],
        conditions=[c.value for c in DiagnosticCondition],
        runs_per_cell=2,
        ledger_path=ledger,
    )

    assert study["total_experiment_runs"] == 2 * 5 * 2 * 3  # 60 total blind runs
    analysis = study["analysis"]
    assert "by_method_and_condition" in analysis
    assert "by_defect_family" in analysis
    assert "incremental_weight_effect_vs_behavioral_only" in analysis
    assert "conclusion" in analysis

    # Verify gradient baseline abstains honestly under weights_only (where behavioral_failures are absent)
    grad_weights_only = analysis["by_method_and_condition"]["gradient_baseline"]["weights_only"]
    assert grad_weights_only["evaluated_runs"] == 0
    assert grad_weights_only["pass_rate"] == 0.0
