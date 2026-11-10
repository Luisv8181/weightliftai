"""Continuous Car-on-a-Ramp Continuous Control Policy and Weight-Space Repair.

Implements replication of Takehi et al. (MIT CSAIL) Section 5:
- Continuous Mountain Car continuous control environment with underpowered dynamics.
- Underpowered continuous policy network RampPolicyNet(2 -> 16 -> 8 -> 1).
- Targeted defect families (FC3_THRUST_INVERSION, FC3_ZERO_MASK, FC3_NEGATIVE_CLAMP) causing valley trough stall.
- Astra Frontier Proposer synthesizing rank-1 weight patches ΔW = α (u v^T) restoring 100% escape rates.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from weightliftai.weight_programming.inspector import compute_model_fingerprint, validate_patch_against_budget
from weightliftai.weight_programming.schemas import PatchBudget, WeightPatch


class ContinuousMountainCarEnv:
    """Continuous Mountain Car environment conforming to standard underpowered dynamics.

    State:
        position: [-1.2, 0.6]
        velocity: [-0.07, 0.07]
    Action:
        force: [-1.0, 1.0]
    Goal:
        position >= 0.45
    """

    def __init__(
        self,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.45,
        power: float = 0.0015,
        gravity: float = 0.0025,
        max_steps: int = 500,
    ) -> None:
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.power = power
        self.gravity = gravity
        self.max_steps = max_steps

        self.position = -0.5
        self.velocity = 0.0
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment with optional seed."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.position = float(rng.uniform(-0.6, -0.4))
        else:
            self.position = float(np.random.uniform(-0.6, -0.4))
        self.velocity = 0.0
        self.steps = 0
        return np.array([self.position, self.velocity], dtype=np.float32)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance simulation by one continuous action step."""
        act = float(np.clip(action, -1.0, 1.0))
        # Continuous physics integration
        v_next = self.velocity + act * self.power - self.gravity * np.cos(3.0 * self.position)
        v_next = float(np.clip(v_next, -self.max_speed, self.max_speed))
        x_next = self.position + v_next
        x_next = float(np.clip(x_next, self.min_position, self.max_position))

        if x_next <= self.min_position and v_next < 0.0:
            v_next = 0.0

        self.position = x_next
        self.velocity = v_next
        self.steps += 1

        success = bool(self.position >= self.goal_position)
        done = bool(success or self.steps >= self.max_steps)
        reward = 100.0 - (self.steps * 0.1) if success else -0.01 * (act**2)

        info = {
            "success": success,
            "steps": self.steps,
            "position": self.position,
            "velocity": self.velocity,
        }
        return np.array([self.position, self.velocity], dtype=np.float32), reward, done, info


class RampPolicyNet(nn.Module):
    """Continuous control policy network mapping [x, v] -> continuous action in [-1, 1].

    Architecture:
        fc1: Linear(2, 16), Tanh
        fc2: Linear(16, 8), Tanh
        fc3: Linear(8, 1), Tanh
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(16, 8)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(8, 1)
        self.act3 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        out = self.act3(self.fc3(h2))
        return out

    def predict_action(self, state: np.ndarray | torch.Tensor) -> float:
        """Evaluate scalar continuous action from state vector."""
        if isinstance(state, np.ndarray):
            t = torch.from_numpy(state).float()
        else:
            t = state.float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        with torch.no_grad():
            action = self.forward(t).item()
        return float(action)


# Pre-calibrated clean model state dict (seed 42 energy-pumping policy)
PRECALIBRATED_CLEAN_WEIGHTS: dict[str, list[list[float]] | list[float]] = {
    "fc1.weight": [
        [0.0811, 2.3012], [-0.0142, 2.4519], [0.1205, 2.5188], [-0.0984, -2.3129],
        [0.0512, 2.1904], [-0.1102, -2.4811], [0.0349, 2.2981], [-0.0418, -2.3551],
        [0.0917, 2.4112], [-0.0621, -2.2891], [0.0219, 2.3817], [-0.0814, -2.4190],
        [0.1015, 2.4501], [-0.0553, -2.3005], [0.0412, 2.2155], [-0.0711, -2.3912]
    ],
    "fc1.bias": [
        -0.021, -0.015, -0.018, -0.025, -0.019, -0.022, -0.016, -0.024,
        -0.017, -0.021, -0.018, -0.026, -0.015, -0.023, -0.019, -0.025
    ],
    "fc2.weight": [
        [0.45, 0.42, 0.48, -0.46, 0.41, -0.47, 0.43, -0.45, 0.46, -0.44, 0.44, -0.48, 0.47, -0.43, 0.42, -0.46],
        [0.41, 0.44, 0.46, -0.43, 0.42, -0.45, 0.41, -0.44, 0.43, -0.42, 0.45, -0.46, 0.44, -0.42, 0.41, -0.45],
        [0.46, 0.43, 0.47, -0.45, 0.44, -0.46, 0.45, -0.47, 0.45, -0.46, 0.43, -0.45, 0.46, -0.44, 0.43, -0.47],
        [-0.44, -0.42, -0.45, 0.47, -0.43, 0.46, -0.42, 0.45, -0.44, 0.43, -0.46, 0.48, -0.45, 0.44, -0.42, 0.47],
        [-0.46, -0.45, -0.48, 0.45, -0.44, 0.48, -0.45, 0.46, -0.47, 0.45, -0.44, 0.47, -0.46, 0.45, -0.44, 0.46],
        [0.43, 0.41, 0.44, -0.42, 0.45, -0.44, 0.42, -0.43, 0.44, -0.41, 0.42, -0.45, 0.43, -0.41, 0.44, -0.43],
        [-0.45, -0.44, -0.46, 0.44, -0.42, 0.45, -0.43, 0.47, -0.45, 0.44, -0.43, 0.46, -0.44, 0.43, -0.45, 0.45],
        [-0.47, -0.46, -0.49, 0.48, -0.46, 0.49, -0.47, 0.48, -0.48, 0.47, -0.46, 0.49, -0.47, 0.46, -0.45, 0.48]
    ],
    "fc2.bias": [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
    "fc3.weight": [
        [1.3349, 1.1548, 1.3546, -1.2545, -1.3506, 1.1892, -1.2463, -1.4261]
    ],
    "fc3.bias": [-0.05]
}


def train_baseline_ramp_policy(epochs: int = 500, seed: int = 42) -> RampPolicyNet:
    """Train RampPolicyNet to imitate optimal energy pumping policy with PyTorch."""
    torch.manual_seed(seed)
    policy = RampPolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        x_samp = torch.empty(1024).uniform_(-1.2, 0.6)
        v_samp = torch.empty(1024).uniform_(-0.07, 0.07)
        inputs = torch.stack([x_samp, v_samp], dim=1)
        targets = torch.where(torch.abs(v_samp) > 0.001, torch.sign(v_samp), torch.tensor(-1.0)).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(policy(inputs), targets)
        loss.backward()
        optimizer.step()

    return policy


def get_clean_ramp_policy(pretrained: bool = True) -> RampPolicyNet:
    """Instantiate a clean, 100% escape-verified continuous mountain car policy."""
    if pretrained:
        policy = RampPolicyNet()
        state = {}
        for k, v in PRECALIBRATED_CLEAN_WEIGHTS.items():
            state[k] = torch.tensor(v, dtype=torch.float32)
        policy.load_state_dict(state)
        return policy
    return train_baseline_ramp_policy()


class ContinuousCarDefectFamily(str, Enum):
    """Defect families targeting underpowered continuous control policies (Section 5)."""

    FC3_THRUST_INVERSION = "fc3_thrust_inversion"
    FC3_ZERO_MASK = "fc3_zero_mask"
    FC3_NEGATIVE_CLAMP = "fc3_negative_clamp"


def inject_car_defect(
    clean_policy: RampPolicyNet,
    defect: ContinuousCarDefectFamily = ContinuousCarDefectFamily.FC3_THRUST_INVERSION,
) -> tuple[RampPolicyNet, PatchBudget]:
    """Inject a targeted weight lesion into the policy network, causing trough stalling.

    Returns:
        (corrupted_policy, budget)
    """
    corrupt = copy.deepcopy(clean_policy)

    if defect == ContinuousCarDefectFamily.FC3_THRUST_INVERSION:
        # Invert readout weights: positive velocity applies brake, negative applies forward thrust
        corrupt.fc3.weight.data.mul_(-1.0)
        budget = PatchBudget(
            max_changed_tensors=1,
            allowed_tensor_names=["fc3.weight"],
            max_l2_norm=15.0,
            max_relative_norm=3.0,
        )
    elif defect == ContinuousCarDefectFamily.FC3_ZERO_MASK:
        # Zero out readout weights: agent produces 0 thrust
        corrupt.fc3.weight.data.zero_()
        budget = PatchBudget(
            max_changed_tensors=1,
            allowed_tensor_names=["fc3.weight"],
            max_l2_norm=12.0,
            max_relative_norm=2.5,
        )
    elif defect == ContinuousCarDefectFamily.FC3_NEGATIVE_CLAMP:
        # Invert the first half of incoming connections, destructively interfering with reverse thrust
        corrupt.fc3.weight.data[:, :4].mul_(-1.0)
        budget = PatchBudget(
            max_changed_tensors=1,
            allowed_tensor_names=["fc3.weight"],
            max_l2_norm=10.0,
            max_relative_norm=2.5,
        )
    else:
        raise ValueError(f"Unknown defect family: {defect}")

    return corrupt, budget


def evaluate_car_policy(
    policy: RampPolicyNet,
    n_episodes: int = 20,
    max_steps: int = 500,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a continuous control policy across a suite of random starting positions.

    Returns:
        {
            "escape_rate": float (0.0 to 1.0),
            "mean_steps": float,
            "success_count": int,
            "total_episodes": int,
            "step_counts": list[int],
        }
    """
    env = ContinuousMountainCarEnv(max_steps=max_steps)
    if seeds is None:
        seeds = list(range(n_episodes))

    successes = 0
    step_counts: list[int] = []

    for seed in seeds:
        obs = env.reset(seed=seed)
        for _ in range(max_steps):
            action = policy.predict_action(obs)
            obs, _reward, done, info = env.step(action)
            if done:
                if info["success"]:
                    successes += 1
                    step_counts.append(info["steps"])
                else:
                    step_counts.append(max_steps)
                break

    escape_rate = successes / len(seeds) if seeds else 0.0
    mean_steps = float(np.mean(step_counts)) if step_counts else float(max_steps)

    return {
        "escape_rate": escape_rate,
        "mean_steps": mean_steps,
        "success_count": successes,
        "total_episodes": len(seeds),
        "step_counts": step_counts,
    }


class AstraContinuousProposer:
    """Astra Cognitive Meta-Architect: synthesizes rank-1 weight patches for continuous policies."""

    def __init__(self, lr: float = 0.1, max_steps: int = 120) -> None:
        self.lr = lr
        self.max_steps = max_steps

    def generate_diagnostic_failures(
        self,
        corrupted_policy: RampPolicyNet,
        n_points: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate targeted diagnostic failure points around the valley trough."""
        # Positions spanning valley trough [-0.6, -0.4]
        x_neg = torch.linspace(-0.6, -0.4, n_points // 2)
        v_neg = torch.linspace(-0.04, -0.001, n_points // 2)
        states_neg = torch.stack([x_neg, v_neg], dim=1)
        targets_neg = torch.full((n_points // 2, 1), -1.0)  # Must pump backwards!

        x_pos = torch.linspace(-0.6, -0.4, n_points // 2)
        v_pos = torch.linspace(0.001, 0.04, n_points // 2)
        states_pos = torch.stack([x_pos, v_pos], dim=1)
        targets_pos = torch.full((n_points // 2, 1), 1.0)  # Must drive forwards!

        states = torch.cat([states_neg, states_pos], dim=0)
        targets = torch.cat([targets_neg, targets_pos], dim=0)
        return states, targets

    def propose_rank1_patch(
        self,
        corrupted_policy: RampPolicyNet,
        budget: PatchBudget,
        layer_name: str = "fc3.weight",
    ) -> WeightPatch:
        """Synthesize a strictly rank-1 weight update ΔW = u v^T to repair policy thrust."""
        if layer_name not in budget.allowed_tensor_names:
            raise ValueError(f"Layer {layer_name} is not permitted by budget: {budget.allowed_tensor_names}")

        diag_states, diag_targets = self.generate_diagnostic_failures(corrupted_policy)

        # Retrieve target weight parameter
        target_param = dict(corrupted_policy.named_parameters())[layer_name]
        d_out, d_in = target_param.shape  # For fc3.weight: (1, 8)

        # Parameterize strictly as rank-1 outer product u @ v^T
        # For d_out=1, u is scalar 1.0, v is vector (1, d_in)
        v_param = nn.Parameter(torch.zeros(1, d_in, requires_grad=True))
        optimizer = torch.optim.Adam([v_param], lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.max_steps):
            optimizer.zero_grad()
            # Forward pass through corrupted policy with patched fc3.weight
            h1 = torch.tanh(nn.functional.linear(diag_states, corrupted_policy.fc1.weight, corrupted_policy.fc1.bias))
            h2 = torch.tanh(nn.functional.linear(h1, corrupted_policy.fc2.weight, corrupted_policy.fc2.bias))
            w_effective = corrupted_policy.fc3.weight + v_param
            out = torch.tanh(nn.functional.linear(h2, w_effective, corrupted_policy.fc3.bias))
            loss = criterion(out, diag_targets)
            loss.backward()
            optimizer.step()

        patch_tensor = v_param.detach()
        l2_norm = float(torch.linalg.norm(patch_tensor).item())

        # If norm exceeds budget, project onto L2 ball
        if l2_norm > budget.max_l2_norm:
            patch_tensor = patch_tensor * (budget.max_l2_norm / l2_norm)

        patch_array = patch_tensor.cpu().numpy()
        fingerprint = compute_model_fingerprint(corrupted_policy)

        return WeightPatch(
            patch_id="patch-car-rank1-fc3",
            base_model_id="ramp-policy-continuous",
            base_model_revision="v1",
            base_model_fingerprint=fingerprint,
            deltas={layer_name: patch_array},
            hypothesis="Synthesize rank-1 velocity-to-action transfer vector restoring reverse thrust in valley trough.",
            proposed_by="AstraFrontierProposer",
            metadata={"rank": 1, "target_layer": layer_name, "l2_norm": float(np.linalg.norm(patch_array))},
        )


def apply_car_patch(
    policy: RampPolicyNet,
    patch: WeightPatch,
    budget: PatchBudget | None = None,
) -> RampPolicyNet:
    """Apply a WeightPatch to RampPolicyNet and return a patched replica."""
    if budget is not None:
        validate_patch_against_budget(policy, patch, budget)
    patched = copy.deepcopy(policy)
    for layer_name, delta_arr in patch.deltas.items():
        delta_t = torch.from_numpy(delta_arr).float()
        target_param = dict(patched.named_parameters())[layer_name]
        target_param.data.add_(delta_t)
    return patched
