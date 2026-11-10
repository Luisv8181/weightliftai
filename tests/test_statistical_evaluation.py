"""Tests for Phase 3 & 4: Statistical Evaluation, Bootstrap CIs, Paired Permutation Tests,
Promotion Gating, Negative Control Falsification, and Benchmark Ladder Progression.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from weightliftai.weight_programming import (
    BenchmarkLadderRunner,
    DeepNonlinearMLP,
    LadderProgressionReport,
    LadderRung,
    MultiSeedBenchmarkRunner,
    NoOpProposer,
    OrthogonalShamProposer,
    PatchBudget,
    RandomNormMatchedProposer,
    SimpleTransformerBlock,
    StatisticalPromotionDecision,
    WeightPatch,
    compute_bootstrap_ci,
    paired_permutation_test,
)
from weightliftai.weight_programming.defect_families import DefectFamily


class MockSuccessfulProposer:
    """Mock proposer that repairs fc2_row0_sign_inversion by flipping sign back."""

    def propose(self, model: torch.nn.Module, budget: PatchBudget, target_tensor: str = "fc2.weight") -> WeightPatch:
        p = dict(model.named_parameters())[target_tensor]
        # Sign inversion inverted fc2.weight[0, :4]. We invert it back: delta = -2 * corrupted_slice
        delta = np.zeros(p.shape, dtype=np.float32)
        corrupted_vals = p.detach().cpu().numpy()[0, :4]
        delta[0, :4] = -2.0 * corrupted_vals

        return WeightPatch(
            patch_id="mock-repair",
            base_model_id="toy",
            base_model_revision="r1",
            deltas={target_tensor: delta},
            hypothesis="Mock sign inversion restoration",
            proposed_by="MockSuccessfulProposer",
        )


def test_compute_bootstrap_ci():
    """Verify non-parametric bootstrap confidence interval coverage."""
    # Symmetrical distribution centered at 5.0
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=0.5, size=50).tolist()
    low, high = compute_bootstrap_ci(data, alpha=0.05, n_resamples=1000)

    assert low < high
    assert low < 5.0 < high
    assert high - low < 1.0


def test_paired_permutation_test():
    """Verify paired permutation test distinguishes true differences from null."""
    # 1. Null condition (no difference)
    c_null = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_null = [1.0, 2.0, 3.0, 4.0, 5.0]
    p_null = paired_permutation_test(c_null, b_null)
    assert p_null == 1.0

    # 2. Strong genuine difference (candidate strictly superior, N=8 allows p < 0.05)
    c_diff = [10.0, 12.0, 11.5, 9.8, 10.5, 11.0, 12.5, 10.8]
    b_diff = [2.0, 1.5, 2.2, 1.8, 2.0, 1.9, 2.1, 1.7]
    p_diff = paired_permutation_test(c_diff, b_diff, n_permutations=2000)
    assert p_diff < 0.05


def test_negative_control_noop_fails_promotion():
    """NoOpProposer proposes zero delta, which must fail the significance gate (gain = 0)."""
    runner = MultiSeedBenchmarkRunner(seeds=[42, 101, 202, 303, 404, 505, 606, 707])
    proposer = NoOpProposer()
    decision = runner.evaluate_proposer(proposer, defect_family=DefectFamily.FC2_ROW0_SIGN_INVERSION.value)

    assert isinstance(decision, StatisticalPromotionDecision)
    assert decision.passed_significance_gate is False
    assert decision.overall_promoted is False
    assert decision.target_gain_mean == 0.0
    assert any("Failed significance gate" in r for r in decision.rejection_reasons)


def test_negative_control_random_noise_fails_promotion():
    """RandomNormMatchedProposer injects uncalibrated Gaussian noise and must fail promotion."""
    runner = MultiSeedBenchmarkRunner(seeds=[42, 101, 202, 303, 404, 505, 606, 707])
    proposer = RandomNormMatchedProposer(target_l2_norm=2.0, seed=123)
    decision = runner.evaluate_proposer(proposer, defect_family=DefectFamily.FC2_ROW0_SIGN_INVERSION.value)

    assert decision.overall_promoted is False
    # Noise fails either by degrading performance (negative gain) or causing regression
    assert (not decision.passed_significance_gate) or (not decision.passed_regression_gate)


def test_negative_control_orthogonal_sham_fails_promotion():
    """OrthogonalShamProposer modifies uncorrupted row 1 instead of defect row 0; must fail promotion."""
    runner = MultiSeedBenchmarkRunner(seeds=[42, 101, 202, 303, 404, 505, 606, 707])
    proposer = OrthogonalShamProposer(target_l2_norm=1.5)
    decision = runner.evaluate_proposer(proposer, defect_family=DefectFamily.FC2_ROW0_SIGN_INVERSION.value)

    assert decision.overall_promoted is False
    assert decision.passed_significance_gate is False


def test_successful_proposer_passes_statistical_promotion():
    """A mathematically valid repair proposer must pass p < 0.05, regression, and OOD gates."""
    runner = MultiSeedBenchmarkRunner(seeds=[42, 101, 202, 303, 404, 505, 606, 707])
    proposer = MockSuccessfulProposer()
    decision = runner.evaluate_proposer(proposer, defect_family=DefectFamily.FC2_ROW0_SIGN_INVERSION.value)

    assert decision.target_gain_mean > 0.5
    assert decision.target_p_value_vs_corrupted < 0.05
    assert decision.passed_significance_gate is True
    assert decision.passed_regression_gate is True
    assert decision.passed_ood_gate is True
    assert decision.overall_promoted is True
    assert len(decision.rejection_reasons) == 0


def test_benchmark_ladder_progression_and_fail_closed_gating():
    """Verify that the benchmark ladder allows valid proposers to advance while stopping invalid ones."""
    ladder = BenchmarkLadderRunner(seeds=[42, 101, 202, 303, 404, 505, 606, 707])

    # 1. Negative control (NoOp) must fail at Rung 1 and NOT advance
    noop = NoOpProposer()
    noop_report = ladder.run_ladder(noop)
    assert noop_report.completed_all_rungs is False
    assert noop_report.highest_rung_passed is None
    assert len(noop_report.rung_results) == 1
    assert noop_report.rung_results[0].passed is False

    # 2. Successful proposer passes Rung 1 and advances to Rung 3
    good_proposer = MockSuccessfulProposer()
    good_report = ladder.run_ladder(good_proposer)
    assert good_report.highest_rung_passed == LadderRung.RUNG_3_CONTINUOUS_CONTROL
    assert good_report.completed_all_rungs is True
    assert len(good_report.rung_results) == 2
    assert all(r.passed for r in good_report.rung_results)


def test_deep_nonlinear_mlp_and_transformer_architectures():
    """Verify structural validity and forward propagation of Rungs 2 and 4 architectures."""
    # Rung 2: Deep Nonlinear MLP
    deep_mlp = DeepNonlinearMLP(in_features=4, hidden_1=16, hidden_2=8, out_features=2)
    x_mlp = torch.randn(8, 4)
    out_mlp = deep_mlp(x_mlp)
    assert out_mlp.shape == (8, 2)

    # Rung 4: Simple Transformer Block
    trans = SimpleTransformerBlock(d_model=16, nhead=2, d_ff=32)
    x_trans = torch.randn(4, 10, 16)  # batch 4, seq_len 10, d_model 16
    out_trans = trans(x_trans)
    assert out_trans.shape == (4, 10, 16)
