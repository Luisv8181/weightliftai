"""Unit and integration tests for Continuous Car-on-a-Ramp Policy Repair (Section 5).

Verifies:
- Continuous Mountain Car physics simulation and underpowered energy barrier.
- Clean policy baseline achieving 100% escape rate.
- Defect injection causing complete valley trough stalling (0% escape).
- Astra Frontier Proposer rank-1 patch synthesis restoring >= 90% escape within budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
from weightliftai.weight_programming.schemas import PatchBudget


def test_mountain_car_physics_underpowered_barrier() -> None:
    """Verify that pure forward thrust cannot summit the right ramp from rest."""
    env = ContinuousMountainCarEnv(max_steps=200)
    # Start at bottom of trough
    env.position = -0.5
    env.velocity = 0.0

    # Apply maximum forward thrust for 100 steps
    for _ in range(100):
        obs, _r, done, info = env.step(1.0)
        if done:
            break

    # Direct forward climb must fail because engine power 0.0015 < gravity peak 0.0025
    assert not info["success"], "Direct forward thrust should not summit without energy pumping"
    assert env.position < 0.45


def test_clean_policy_achieves_perfect_escape() -> None:
    """Verify pre-calibrated clean policy achieves 100% escape rate across 20 seeds."""
    policy = get_clean_ramp_policy(pretrained=True)
    res = evaluate_car_policy(policy, n_episodes=20, max_steps=500)

    assert res["escape_rate"] == 1.0, f"Expected 100% escapes, got {res['escape_rate']}"
    assert res["mean_steps"] < 130.0, f"Expected fast escape (<130 steps), got {res['mean_steps']}"
    assert res["success_count"] == 20


@pytest.mark.parametrize(
    "defect_type",
    [
        ContinuousCarDefectFamily.FC3_THRUST_INVERSION,
        ContinuousCarDefectFamily.FC3_ZERO_MASK,
        ContinuousCarDefectFamily.FC3_NEGATIVE_CLAMP,
    ],
)
def test_defect_injection_causes_trough_stall(defect_type: ContinuousCarDefectFamily) -> None:
    """Verify that defect injection collapses policy performance to 0% escape rate."""
    clean_policy = get_clean_ramp_policy(pretrained=True)
    corrupted_policy, budget = inject_car_defect(clean_policy, defect_type)

    res = evaluate_car_policy(corrupted_policy, n_episodes=10, max_steps=500)
    assert res["escape_rate"] == 0.0, f"Defect {defect_type} failed to induce 0% stall (got {res['escape_rate']})"
    assert res["mean_steps"] == 500.0
    assert budget.max_changed_tensors == 1
    assert "fc3.weight" in budget.allowed_tensor_names


@pytest.mark.parametrize(
    "defect_type",
    [
        ContinuousCarDefectFamily.FC3_THRUST_INVERSION,
        ContinuousCarDefectFamily.FC3_ZERO_MASK,
        ContinuousCarDefectFamily.FC3_NEGATIVE_CLAMP,
    ],
)
def test_astra_rank1_patch_synthesis_and_restoration(defect_type: ContinuousCarDefectFamily) -> None:
    """Verify Astra Frontier Proposer synthesizes rank-1 patch restoring >= 90% escape across defects."""
    clean_policy = get_clean_ramp_policy(pretrained=True)
    corrupted_policy, budget = inject_car_defect(clean_policy, defect_type)

    # Initial corrupted check
    pre_res = evaluate_car_policy(corrupted_policy, n_episodes=10, max_steps=500)
    assert pre_res["escape_rate"] == 0.0

    proposer = AstraContinuousProposer(lr=0.1, max_steps=100)
    patch = proposer.propose_rank1_patch(corrupted_policy, budget, layer_name="fc3.weight")

    # Verify patch conforms to rank-1 and budget
    assert "fc3.weight" in patch.deltas
    delta_arr = patch.deltas["fc3.weight"]
    assert delta_arr.shape == (1, 8)
    # Norm check
    l2_norm = float(np.linalg.norm(delta_arr))
    assert l2_norm <= budget.max_l2_norm

    # Apply patch
    repaired_policy = apply_car_patch(corrupted_policy, patch)

    # Post-repair evaluation
    post_res = evaluate_car_policy(repaired_policy, n_episodes=20, max_steps=500)
    assert post_res["escape_rate"] >= 0.90, (
        f"Astra repair on {defect_type} failed to restore target escape rate (got {post_res['escape_rate']})"
    )
    assert post_res["mean_steps"] < 130.0


def test_disallowed_layer_raises_budget_violation() -> None:
    """Verify proposing a patch for a layer not in allowed_tensor_names raises ValueError."""
    clean_policy = get_clean_ramp_policy(pretrained=True)
    corrupted_policy, budget = inject_car_defect(clean_policy)

    proposer = AstraContinuousProposer()
    with pytest.raises(ValueError, match="not permitted by budget"):
        proposer.propose_rank1_patch(corrupted_policy, budget, layer_name="fc1.weight")
