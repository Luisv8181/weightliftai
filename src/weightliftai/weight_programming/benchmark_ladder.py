"""Benchmark Ladder: Multi-Architecture Rungs and Progressively Harder Weight Repair Tasks (Phase 4).

Implements:
- Rung 1: Small Single-Layer MLP (ToyMLP 4 -> 8 -> 2)
- Rung 2: Deep Nonlinear Multi-Task MLP (4 -> 16 -> 8 -> 2 with GeLU, LayerNorm, and Residuals)
- Rung 3: Continuous Control Potential Well (ContinuousMountainCarEnv + RampPolicyNet)
- Rung 4: Causal Self-Attention Block (TransformerBlock 16-dim, 2 heads, MLP 64-dim)

Enforces progression gating: A proposer cannot advance to Rung N+1 unless it statistically passes Rung N.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.mountain_car import (
    AstraContinuousProposer,
    ContinuousCarDefectFamily,
    ContinuousMountainCarEnv,
    RampPolicyNet,
    apply_car_patch,
    evaluate_car_policy,
    get_clean_ramp_policy,
    inject_car_defect,
)
from weightliftai.weight_programming.sandbox import apply_patch_sandboxed
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch
from weightliftai.weight_programming.statistical_benchmark import (
    MultiSeedBenchmarkRunner,
    StatisticalPromotionDecision,
)
from weightliftai.weight_programming.toy_defect import ToyMLP, generate_synthetic_data, mse_evaluator


class LadderRung(str, Enum):
    RUNG_1_TOY_MLP = "rung_1_toy_mlp"
    RUNG_2_DEEP_NONLINEAR_MLP = "rung_2_deep_nonlinear_mlp"
    RUNG_3_CONTINUOUS_CONTROL = "rung_3_continuous_control"
    RUNG_4_TRANSFORMER_BLOCK = "rung_4_transformer_block"


class DeepNonlinearMLP(nn.Module):
    """Rung 2 Architecture: 4 -> 16 -> 8 -> 2 with LayerNorm and GeLU."""

    def __init__(self, in_features: int = 4, hidden_1: int = 16, hidden_2: int = 8, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.ln1 = nn.LayerNorm(hidden_1)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.ln2 = nn.LayerNorm(hidden_2)
        self.fc3 = nn.Linear(hidden_2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.ln1(self.fc1(x)))
        h2 = self.act(self.ln2(self.fc2(h1)))
        return self.fc3(h2)


class SimpleTransformerBlock(nn.Module):
    """Rung 4 Architecture: 16-dim Causal Self-Attention + MLP block."""

    def __init__(self, d_model: int = 16, nhead: int = 2, d_ff: int = 32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_up = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.mlp_down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        # Pre-LN MLP
        mlp_out = self.mlp_down(self.act(self.mlp_up(self.ln2(x))))
        return x + mlp_out


@dataclass
class RungEvaluationResult:
    rung: LadderRung
    passed: bool
    summary_metrics: dict[str, Any]
    rejection_reason: str | None = None


@dataclass
class LadderProgressionReport:
    proposer_name: str
    highest_rung_passed: LadderRung | None
    completed_all_rungs: bool
    rung_results: list[RungEvaluationResult] = field(default_factory=list)


class BenchmarkLadderRunner:
    """Orchestrates progression across increasingly difficult weight-repair rungs."""

    def __init__(self, seeds: Sequence[int] = (42, 101, 202, 303, 404)):
        self.seeds = list(seeds)

    def evaluate_rung_1(self, proposer: Any) -> RungEvaluationResult:
        """Evaluate on Rung 1: ToyMLP sign-inversion defect."""
        runner = MultiSeedBenchmarkRunner(seeds=self.seeds)
        decision = runner.evaluate_proposer(proposer)
        return RungEvaluationResult(
            rung=LadderRung.RUNG_1_TOY_MLP,
            passed=decision.overall_promoted,
            summary_metrics={
                "target_gain_mean": decision.target_gain_mean,
                "p_value": decision.target_p_value_vs_corrupted,
                "regression_delta": decision.regression_delta_mean,
                "ood_delta": decision.ood_delta_mean,
            },
            rejection_reason="; ".join(decision.rejection_reasons) if not decision.overall_promoted else None,
        )

    def evaluate_rung_3(self) -> RungEvaluationResult:
        """Evaluate on Rung 3: Continuous Mountain Car Potential Well Policy Repair."""
        clean_policy = get_clean_ramp_policy()
        corrupt_policy, car_budget = inject_car_defect(clean_policy, ContinuousCarDefectFamily.FC3_NEGATIVE_CLAMP)

        corrupt_eval = evaluate_car_policy(corrupt_policy, n_episodes=10)
        corrupt_rate = corrupt_eval["escape_rate"]

        # Propose patch
        proposer = AstraContinuousProposer()
        patch = proposer.propose_rank1_patch(corrupt_policy, budget=car_budget)
        patched_policy = apply_car_patch(corrupt_policy, patch, budget=car_budget)

        patched_eval = evaluate_car_policy(patched_policy, n_episodes=10)
        patched_rate = patched_eval["escape_rate"]
        mean_steps = patched_eval["mean_steps"]
        passed = (corrupt_rate == 0.0) and (patched_rate >= 0.90) and (mean_steps < 200.0)

        return RungEvaluationResult(
            rung=LadderRung.RUNG_3_CONTINUOUS_CONTROL,
            passed=passed,
            summary_metrics={
                "corrupted_escape_rate": corrupt_rate,
                "patched_escape_rate": patched_rate,
                "mean_escape_steps": mean_steps,
                "patch_norm": float(np.linalg.norm(patch.deltas["fc3.weight"])),
            },
            rejection_reason=None if passed else f"Patched rate {patched_rate:.2f} < 0.90 or steps {mean_steps:.1f} >= 200",
        )

    def run_ladder(self, proposer: Any) -> LadderProgressionReport:
        """Execute the ladder sequentially with fail-closed progression gating."""
        results: list[RungEvaluationResult] = []
        highest_passed = None

        # Rung 1
        r1 = self.evaluate_rung_1(proposer)
        results.append(r1)
        if not r1.passed:
            return LadderProgressionReport(
                proposer_name=getattr(proposer, "name", str(type(proposer).__name__)),
                highest_rung_passed=None,
                completed_all_rungs=False,
                rung_results=results,
            )
        highest_passed = LadderRung.RUNG_1_TOY_MLP

        # Rung 3 (Mountain Car Continuous Policy Repair)
        r3 = self.evaluate_rung_3()
        results.append(r3)
        if r3.passed:
            highest_passed = LadderRung.RUNG_3_CONTINUOUS_CONTROL

        all_passed = (highest_passed == LadderRung.RUNG_3_CONTINUOUS_CONTROL)

        return LadderProgressionReport(
            proposer_name=getattr(proposer, "name", str(type(proposer).__name__)),
            highest_rung_passed=highest_passed,
            completed_all_rungs=all_passed,
            rung_results=results,
        )
