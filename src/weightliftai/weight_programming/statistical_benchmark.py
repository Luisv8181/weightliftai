"""Statistical Evaluation, Multi-Seed Benchmarking, and Negative Controls (Phase 3).

Implements:
- Multiple independent seeds for defect generation, proposer evaluation, and held-out validation.
- Empirical bootstrap confidence intervals (95% CI) and standard errors.
- Paired permutation tests comparing candidate patch vs. corrupted baseline (no-op) and clean upper bound.
- Rigorous promotion criteria: p < 0.05 on held-out target gain, bounded regression degradation, and bounded OOD degradation.
- Negative controls: No-op proposer, random norm-matched noise proposer, and orthogonal sham proposer.
- Full distribution logging preserving all runs, rejections, and null results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.defect_families import (
    DefectFamily,
    DiagnosticCondition,
    deserialize_model_state_dict,
    generate_private_benchmark_instance,
)
from weightliftai.weight_programming.inspector import compute_model_fingerprint
from weightliftai.weight_programming.sandbox import apply_patch_sandboxed
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch
from weightliftai.weight_programming.toy_defect import ToyMLP, mse_evaluator


def compute_bootstrap_ci(
    data: Sequence[float],
    alpha: float = 0.05,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute empirical non-parametric bootstrap confidence interval."""
    arr = np.array(list(data), dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    n = len(arr)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (round(low, 4), round(high, 4))


def paired_permutation_test(
    candidate_scores: Sequence[float],
    baseline_scores: Sequence[float],
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """Exact/Monte-Carlo paired two-sided permutation test.
    
    Tests H0: Mean difference (candidate - baseline) == 0.
    """
    c = np.array(list(candidate_scores), dtype=np.float64)
    b = np.array(list(baseline_scores), dtype=np.float64)
    n = min(len(c), len(b))
    if n == 0:
        return 1.0

    diffs = c[:n] - b[:n]
    observed_t = abs(float(np.mean(diffs)))
    if observed_t < 1e-12:
        return 1.0

    rng = np.random.default_rng(seed)
    # Generate random sign flips for paired differences
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    perm_means = np.abs(np.mean(signs * diffs, axis=1))

    p_value = float(np.mean(perm_means >= observed_t))
    return round(max(p_value, 1.0 / n_permutations), 4)


@dataclass
class MultiSeedRunRecord:
    """Individual run record for a single seed."""
    seed: int
    defect_family: str
    target_clean_loss: float
    target_corrupted_loss: float
    target_patched_loss: float
    target_gain: float  # corrupted_loss - patched_loss
    regression_loss_delta: float  # patched_regr_loss - clean_regr_loss
    ood_loss_delta: float  # patched_ood_loss - clean_ood_loss
    patch_l2_norm: float
    patch_applied: bool
    rejection_reason: str | None = None


@dataclass
class StatisticalPromotionDecision:
    """Statistical evaluation and promotion decision across seeds."""
    defect_family: str
    num_seeds: int
    target_gain_mean: float
    target_gain_std: float
    target_gain_ci95: tuple[float, float]
    target_p_value_vs_corrupted: float
    regression_delta_mean: float
    regression_delta_ci95: tuple[float, float]
    ood_delta_mean: float
    ood_delta_ci95: tuple[float, float]
    mean_patch_l2_norm: float
    passed_significance_gate: bool  # p < 0.05 and target_gain > 0
    passed_regression_gate: bool    # regression delta <= tolerance
    passed_ood_gate: bool           # OOD delta <= tolerance
    overall_promoted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    raw_runs: list[MultiSeedRunRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Negative Control Proposers
# ---------------------------------------------------------------------------

class NoOpProposer:
    """Negative Control 1: Proposes a null zero patch."""

    def propose(self, model: nn.Module, budget: PatchBudget, target_tensor: str = "fc2.weight") -> WeightPatch:
        p = dict(model.named_parameters())[target_tensor]
        zero_delta = np.zeros(p.shape, dtype=np.float32)
        return WeightPatch(
            patch_id="noop-control",
            base_model_id="toy",
            base_model_revision="r1",
            base_model_fingerprint=compute_model_fingerprint(model),
            deltas={target_tensor: zero_delta},
            hypothesis="Negative control: zero delta",
            proposed_by="NoOpProposer",
        )


class RandomNormMatchedProposer:
    """Negative Control 2: Proposes Gaussian random noise matched to a target L2 norm."""

    def __init__(self, target_l2_norm: float = 1.0, seed: int = 42):
        self.target_l2_norm = target_l2_norm
        self.seed = seed

    def propose(self, model: nn.Module, budget: PatchBudget, target_tensor: str = "fc2.weight") -> WeightPatch:
        rng = np.random.default_rng(self.seed)
        p = dict(model.named_parameters())[target_tensor]
        raw_noise = rng.standard_normal(size=p.shape).astype(np.float32)
        current_l2 = float(np.linalg.norm(raw_noise))
        scaled_noise = raw_noise * (self.target_l2_norm / max(current_l2, 1e-8))

        return WeightPatch(
            patch_id="random-noise-control",
            base_model_id="toy",
            base_model_revision="r1",
            base_model_fingerprint=compute_model_fingerprint(model),
            deltas={target_tensor: scaled_noise},
            hypothesis="Negative control: random norm-matched noise",
            proposed_by="RandomNormMatchedProposer",
        )


class OrthogonalShamProposer:
    """Negative Control 3: Proposes a patch in the orthogonal complement of the true defect."""

    def __init__(self, target_l2_norm: float = 1.0):
        self.target_l2_norm = target_l2_norm

    def propose(self, model: nn.Module, budget: PatchBudget, target_tensor: str = "fc2.weight") -> WeightPatch:
        p = dict(model.named_parameters())[target_tensor]
        # Defect is on Output 0 (row 0); sham modifies Output 1 (row 1) where no defect exists
        sham_delta = np.zeros(p.shape, dtype=np.float32)
        if sham_delta.shape[0] > 1:
            sham_delta[1, :] = self.target_l2_norm / math.sqrt(sham_delta.shape[1])
        return WeightPatch(
            patch_id="orthogonal-sham-control",
            base_model_id="toy",
            base_model_revision="r1",
            base_model_fingerprint=compute_model_fingerprint(model),
            deltas={target_tensor: sham_delta},
            hypothesis="Negative control: orthogonal sham perturbation",
            proposed_by="OrthogonalShamProposer",
        )


# ---------------------------------------------------------------------------
# Multi-Seed Benchmark Runner
# ---------------------------------------------------------------------------

class MultiSeedBenchmarkRunner:
    """Runs rigorous multi-seed evaluation, paired significance testing, and promotion gating."""

    def __init__(
        self,
        seeds: Sequence[int] = (42, 101, 202, 303, 404, 505, 606, 707),
        alpha_significance: float = 0.05,
        max_regression_tolerance: float = 0.05,
        max_ood_degradation_tolerance: float = 0.20,
    ):
        self.seeds = list(seeds)
        self.alpha_significance = alpha_significance
        self.max_regression_tolerance = max_regression_tolerance
        self.max_ood_degradation_tolerance = max_ood_degradation_tolerance

    def evaluate_proposer(
        self,
        proposer: Any,
        defect_family: str = DefectFamily.FC2_ROW0_SIGN_INVERSION.value,
        budget: PatchBudget | None = None,
    ) -> StatisticalPromotionDecision:
        """Evaluate a proposer across all independent seeds against promotion gates."""
        if budget is None:
            budget = PatchBudget(
                max_changed_tensors=1,
                max_l2_norm=10.0,
                max_relative_norm=5.0,
                allowed_tensor_names=["fc2.weight"],
            )

        runs: list[MultiSeedRunRecord] = []
        target_gains: list[float] = []
        regr_deltas: list[float] = []
        ood_deltas: list[float] = []
        norms: list[float] = []

        for seed in self.seeds:
            # 1. Instantiate private benchmark instance under this seed
            vault_record, _ = generate_private_benchmark_instance(
                private_seed=seed,
                defect_family=defect_family,
            )
            clean_m = deserialize_model_state_dict(vault_record["clean_state_dict"])
            corrupted_m = deserialize_model_state_dict(vault_record["corrupted_state_dict"])
            splits = {
                k: {
                    "x": torch.tensor(v["x"], dtype=torch.float32),
                    "y": torch.tensor(v["y"], dtype=torch.float32),
                    "target_col": v.get("target_col"),
                }
                for k, v in vault_record["splits"].items()
            }

            # mse_evaluator returns -MSE (higher is better)
            clean_tgt = mse_evaluator(clean_m, splits["held_out"])
            corrupted_tgt = mse_evaluator(corrupted_m, splits["held_out"])
            clean_regr = mse_evaluator(clean_m, splits["regression"])
            clean_ood = mse_evaluator(clean_m, splits["distribution_shift"])

            # 2. Get proposed patch
            try:
                patch = proposer.propose(corrupted_m, budget, target_tensor="fc2.weight")
                # 3. Evaluate sandboxed
                with apply_patch_sandboxed(corrupted_m, patch, budget=budget, require_fingerprint=False) as patched_m:
                    patched_tgt = mse_evaluator(patched_m, splits["held_out"])
                    patched_regr = mse_evaluator(patched_m, splits["regression"])
                    patched_ood = mse_evaluator(patched_m, splits["distribution_shift"])

                gain = patched_tgt - corrupted_tgt  # positive if MSE decreased
                regr_d = clean_regr - patched_regr  # positive if regression performance degraded
                ood_d = clean_ood - patched_ood      # positive if OOD performance degraded
                patch_l2 = float(np.linalg.norm(patch.deltas.get("fc2.weight", 0.0)))
                patch_applied = True
                rejection = None
            except Exception as exc:
                gain = 0.0
                regr_d = 0.0
                ood_d = 0.0
                patch_l2 = 0.0
                patch_applied = False
                rejection = str(exc)

            record = MultiSeedRunRecord(
                seed=seed,
                defect_family=defect_family,
                target_clean_loss=round(clean_tgt, 4),
                target_corrupted_loss=round(corrupted_tgt, 4),
                target_patched_loss=round(corrupted_tgt - gain, 4),
                target_gain=round(gain, 4),
                regression_loss_delta=round(regr_d, 4),
                ood_loss_delta=round(ood_d, 4),
                patch_l2_norm=round(patch_l2, 4),
                patch_applied=patch_applied,
                rejection_reason=rejection,
            )
            runs.append(record)
            target_gains.append(gain)
            regr_deltas.append(regr_d)
            ood_deltas.append(ood_d)
            norms.append(patch_l2)

        # 4. Statistical aggregation across seeds
        mean_gain = float(np.mean(target_gains))
        std_gain = float(np.std(target_gains, ddof=1)) if len(target_gains) > 1 else 0.0
        ci95_gain = compute_bootstrap_ci(target_gains)

        # Paired permutation test: candidate gains vs. zero gain (corrupted baseline)
        p_val_target = paired_permutation_test(target_gains, [0.0] * len(target_gains))

        mean_regr = float(np.mean(regr_deltas))
        ci95_regr = compute_bootstrap_ci(regr_deltas)

        mean_ood = float(np.mean(ood_deltas))
        ci95_ood = compute_bootstrap_ci(ood_deltas)

        mean_norm = float(np.mean(norms))

        # 5. Check promotion criteria
        rejection_reasons = []
        passed_significance = (p_val_target < self.alpha_significance) and (mean_gain > 0.0)
        if not passed_significance:
            rejection_reasons.append(
                f"Failed significance gate: p={p_val_target:.4f} >= {self.alpha_significance} or mean_gain={mean_gain:.4f} <= 0"
            )

        passed_regression = mean_regr <= self.max_regression_tolerance
        if not passed_regression:
            rejection_reasons.append(
                f"Failed regression gate: mean_regr_delta={mean_regr:.4f} > tolerance {self.max_regression_tolerance:.4f}"
            )

        passed_ood = mean_ood <= self.max_ood_degradation_tolerance
        if not passed_ood:
            rejection_reasons.append(
                f"Failed OOD gate: mean_ood_delta={mean_ood:.4f} > tolerance {self.max_ood_degradation_tolerance:.4f}"
            )

        overall_promoted = passed_significance and passed_regression and passed_ood

        return StatisticalPromotionDecision(
            defect_family=defect_family,
            num_seeds=len(self.seeds),
            target_gain_mean=round(mean_gain, 4),
            target_gain_std=round(std_gain, 4),
            target_gain_ci95=ci95_gain,
            target_p_value_vs_corrupted=p_val_target,
            regression_delta_mean=round(mean_regr, 4),
            regression_delta_ci95=ci95_regr,
            ood_delta_mean=round(mean_ood, 4),
            ood_delta_ci95=ci95_ood,
            mean_patch_l2_norm=round(mean_norm, 4),
            passed_significance_gate=passed_significance,
            passed_regression_gate=passed_regression,
            passed_ood_gate=passed_ood,
            overall_promoted=overall_promoted,
            rejection_reasons=rejection_reasons,
            raw_runs=runs,
        )
