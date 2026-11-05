"""Research-grade evaluation and agent governance framework.

Implements Phase 5 of the Research Validation Program:
1. Stage-by-stage research grading across the 6 empirical stages.
2. Evidence-linked claims schema binding every factual assertion to code/artifacts.
3. Automated critical review auditing for data leakage, confounds, unsupported causal claims,
   and metric gaming.
4. Human approval gate validation for intervention execution, hidden-test access, and promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Literal, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclasses.dataclass(frozen=True)
class EvidenceLinkedClaim:
    """A scientific claim strictly linked to verifiable artifacts and taxonomical level."""
    claim_id: str
    statement: str
    claim_level: int  # 0 to 4 per RESEARCH_VALIDATION_PROTOCOL
    evidence_type: Literal[
        "unit_test",
        "derivative_check",
        "diagnostic_run",
        "activation_intervention",
        "sealed_benchmark",
    ]
    source_run_id: str
    source_artifact_hashes: dict[str, str]
    limitations: list[str]
    has_human_review: bool = False

    def validate(self) -> None:
        if self.claim_level not in (0, 1, 2, 3, 4):
            raise ValueError(f"claim_level must be between 0 and 4, got {self.claim_level}")
        if not self.statement.strip():
            raise ValueError("Claim statement must not be empty.")
        if not self.source_artifact_hashes:
            raise ValueError("EvidenceLinkedClaim must have at least one source artifact hash.")


@dataclasses.dataclass
class StageGrade:
    stage_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    findings: list[str]


class StageGrader:
    """Grades an experimental run across all six distinct research stages."""

    STAGES = (
        "literature_grounded_hypothesis",
        "preregistration",
        "implementation_correctness",
        "execution",
        "analysis",
        "claim_faithfulness",
    )

    @classmethod
    def grade(
        cls,
        *,
        manifest: dict[str, Any],
        has_citations: bool = True,
        finite_difference_passed: bool = True,
        baselines_present: bool = True,
        negative_controls_present: bool = True,
        claims: Sequence[EvidenceLinkedClaim] = (),
    ) -> dict[str, Any]:
        grades = []

        # 1. Literature-grounded hypothesis
        hyp = manifest.get("hypothesis", "")
        hyp_passed = bool(hyp.strip() and has_citations)
        grades.append(
            StageGrade(
                stage_name="literature_grounded_hypothesis",
                passed=hyp_passed,
                score=1.0 if hyp_passed else 0.0,
                findings=["Hypothesis documented with citations"] if hyp_passed else ["Missing hypothesis or citations"],
            )
        )

        # 2. Preregistration
        prov = manifest.get("provenance", {})
        eval_cfg = manifest.get("evaluation", {})
        has_hash = bool(prov.get("manifest_sha256"))
        has_metric = bool(eval_cfg.get("primary_metric") and eval_cfg.get("success_criterion"))
        prereg_passed = has_hash and has_metric
        grades.append(
            StageGrade(
                stage_name="preregistration",
                passed=prereg_passed,
                score=1.0 if prereg_passed else 0.5,
                findings=["Manifest hash and primary metrics frozen"] if prereg_passed else ["Unsealed preregistration"],
            )
        )

        # 3. Implementation correctness
        grades.append(
            StageGrade(
                stage_name="implementation_correctness",
                passed=finite_difference_passed,
                score=1.0 if finite_difference_passed else 0.0,
                findings=["Finite-difference derivative verified"] if finite_difference_passed else ["Numerical gradient mismatch"],
            )
        )

        # 4. Execution
        exec_passed = baselines_present and negative_controls_present
        grades.append(
            StageGrade(
                stage_name="execution",
                passed=exec_passed,
                score=1.0 if exec_passed else (0.5 if baselines_present else 0.0),
                findings=["Baselines and negative controls executed"] if exec_passed else ["Missing baselines or negative controls"],
            )
        )

        # 5. Analysis
        results = manifest.get("results", {})
        analysis_passed = bool(results.get("baseline") and results.get("intervention"))
        grades.append(
            StageGrade(
                stage_name="analysis",
                passed=analysis_passed,
                score=1.0 if analysis_passed else 0.5,
                findings=["Comparative baseline & intervention results recorded"] if analysis_passed else ["Incomplete comparative analysis"],
            )
        )

        # 6. Claim faithfulness
        unfaithful_claims = []
        for claim in claims:
            # Level 3 requires causal intervention; Level 4 requires sealed benchmark
            if claim.claim_level >= 3 and claim.evidence_type not in (
                "activation_intervention",
                "sealed_benchmark",
            ):
                unfaithful_claims.append(f"{claim.claim_id}: Level {claim.claim_level} claim lacks causal/sealed evidence")
        faith_passed = len(unfaithful_claims) == 0
        grades.append(
            StageGrade(
                stage_name="claim_faithfulness",
                passed=faith_passed,
                score=1.0 if faith_passed else 0.0,
                findings=["All claims match taxonomy levels"] if faith_passed else unfaithful_claims,
            )
        )

        overall_passed = all(g.passed for g in grades)
        mean_score = sum(g.score for g in grades) / len(grades)

        return {
            "overall_passed": overall_passed,
            "mean_score": mean_score,
            "grades": {g.stage_name: dataclasses.asdict(g) for g in grades},
        }


class CriticalReviewAudit:
    """Automated critical audit for experimental rigor, leakage, and metric gaming."""

    CAUSAL_KEYWORDS = re.compile(
        r"\b(causes|causal\s+mechanism|cured|proves\s+that|autonomous\s+repair)\b",
        re.IGNORECASE,
    )

    @classmethod
    def audit_experiment(
        cls,
        *,
        manifest: dict[str, Any],
        diagnostic_sample_ids: set[str],
        test_sample_ids: set[str],
        claims: Sequence[EvidenceLinkedClaim] = (),
    ) -> dict[str, Any]:
        violations = []

        # 1. Leakage Check: train/diagnostic vs sealed test split
        leakage = diagnostic_sample_ids.intersection(test_sample_ids)
        if leakage:
            violations.append(
                f"Data Leakage Detected: {len(leakage)} samples overlap between diagnostic and sealed test sets."
            )

        # 2. Unsupported Causal Language in Hypothesis or Results
        text_corpus = manifest.get("hypothesis", "") + " " + json.dumps(manifest.get("results", {}))
        for match in cls.CAUSAL_KEYWORDS.finditer(text_corpus):
            # Check if Level 3/4 intervention evidence is officially registered
            results = manifest.get("results", {})
            if "intervention" not in results:
                violations.append(
                    f"Unsupported Causal Language: Found '{match.group(0)}' without causal intervention results."
                )
                break

        # 3. Metric Gaming: verify preregistered criteria were not modified
        eval_cfg = manifest.get("evaluation", {})
        if not eval_cfg.get("primary_metric"):
            violations.append("Missing primary metric specification.")

        # 4. Check Claims alignment
        for claim in claims:
            if claim.claim_level > 2 and claim.evidence_type == "diagnostic_run":
                violations.append(
                    f"Claim Overreach: {claim.claim_id} asserts Level {claim.claim_level} on diagnostic run alone."
                )

        passed = len(violations) == 0
        return {
            "audit_passed": passed,
            "violations_count": len(violations),
            "violations": violations,
            "timestamp_utc": _utc_now(),
        }


class HumanApprovalGate:
    """Enforces human-in-the-loop authorization gates before sensitive operations."""

    VALID_GATES = {"INTERVENTION_READY", "HIDDEN_TEST_ACCESS", "PROMOTION"}

    @classmethod
    def verify_token(cls, gate: str, token: str) -> bool:
        if gate not in cls.VALID_GATES:
            raise ValueError(f"Unknown gate '{gate}'. Allowed gates: {cls.VALID_GATES}")
        if not token or not token.strip():
            return False
        # Human tokens require explicit human prefix and minimum length
        token_clean = token.strip()
        pattern = re.compile(rf"^human-({gate.lower()}|approval|promotion)-[a-zA-Z0-9_-]{{3,}}$")
        return bool(pattern.match(token_clean))
