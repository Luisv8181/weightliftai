"""1-Bit Packed Linear Layers and Tiny FP16 Self-Gated Residual Sidecars.

Enables Weight-Space Programming on native 1-bit models (e.g., PrismML Bonsai 1.7B/4B/8B)
in PyTorch without dequantizing the base model to full FP16:
- `Packed1BitLinear`: Packs binary (+/- scale) group weights into `uint8` bit-planes (8 weights/byte)
  plus per-group FP16 scale factors, cutting linear weight memory by ~12-14x.
- `SelfGatedFP16Sidecar`: Wraps a frozen `Packed1BitLinear` (or any `nn.Linear`) and adds a
  context-self-gated FP16 rank-1 residual update (< 5 MB, typically ~10-20 KB) that stays
  strictly closed (exact 0.00% drift) outside the target sense manifold.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Packed1BitLinear(nn.Module):
    """Memory-efficient 1-bit bit-packed linear layer with per-group FP16 scale factors.

    Stores weights as:
    - `packed_signs`: `uint8` tensor of shape `[out_features, in_features // 8]` (1 bit per weight)
    - `scales`: `float16` / `bfloat16` tensor of shape `[out_features, in_features // group_size]`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        has_bias: bool = False,
        scale_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by 8.")
        if in_features % group_size != 0:
            group_size = math.gcd(in_features, group_size)
            if group_size < 8 or group_size % 8 != 0:
                group_size = 8

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.scale_dtype = scale_dtype

        self.register_buffer(
            "packed_signs",
            torch.zeros((out_features, in_features // 8), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones((out_features, in_features // group_size), dtype=scale_dtype),
        )
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=scale_dtype))
        else:
            self.bias = None

        # Precomputed bit masks for fast vectorized unpack
        self.register_buffer(
            "_bit_shifts",
            torch.arange(8, dtype=torch.uint8).view(1, 1, 8),
            persistent=False,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float16,
    ) -> tuple["Packed1BitLinear", float]:
        """Pack an existing `nn.Linear` into 1-bit `uint8` bit-planes + group scales.

        Returns `(packed_module, max_abs_reconstruction_error)`.
        """
        w = linear.weight.detach().float()
        out_f, in_f = w.shape
        eff_group = group_size if (in_f % group_size == 0) else 32
        if in_f % eff_group != 0:
            eff_group = 8

        mod = cls(
            in_features=in_f,
            out_features=out_f,
            group_size=eff_group,
            has_bias=(linear.bias is not None),
            scale_dtype=scale_dtype,
        )

        # Group-wise mean absolute magnitude as 1-bit scale factor
        w_grouped = w.view(out_f, in_f // eff_group, eff_group)
        scales = w_grouped.abs().mean(dim=-1).clamp(min=1e-8)
        signs_bool = (w >= 0).view(out_f, in_f // 8, 8).to(torch.uint8)
        shifts = torch.arange(8, dtype=torch.uint8, device=w.device).view(1, 1, 8)
        packed = (signs_bool << shifts).sum(dim=-1).to(torch.uint8)

        mod.packed_signs.copy_(packed.cpu())
        mod.scales.copy_(scales.to(scale_dtype).cpu())
        if linear.bias is not None and mod.bias is not None:
            mod.bias.copy_(linear.bias.detach().to(scale_dtype).cpu())

        # Measure exact reconstruction error on CPU before discarding unpacked weights
        w_rec = mod.unpack_weight(dtype=torch.float32)
        max_err = float((w.cpu() - w_rec).abs().max().item())
        return mod, max_err

    def unpack_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Unpack 1-bit signs and multiply by per-group scale factors."""
        target_dtype = dtype if dtype is not None else self.scales.dtype
        bits = (self.packed_signs.unsqueeze(-1) >> self._bit_shifts) & 1
        # Map {0, 1} -> {-1.0, +1.0}
        signs = bits.to(target_dtype).mul_(2.0).sub_(1.0).view(self.out_features, self.in_features)
        scales_expanded = (
            self.scales.to(target_dtype)
            .unsqueeze(-1)
            .expand(self.out_features, self.in_features // self.group_size, self.group_size)
            .reshape(self.out_features, self.in_features)
        )
        return signs.mul_(scales_expanded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.unpack_weight(dtype=x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def nbytes(self) -> int:
        total = self.packed_signs.numel() * self.packed_signs.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return int(total)


def pack_transformer_linears_to_1bit(
    model: nn.Module,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.bfloat16,
    skip_names: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> dict[str, Any]:
    """Replace eligible transformer `nn.Linear` submodules in-place with `Packed1BitLinear`."""
    replaced = 0
    bytes_before = 0
    bytes_after = 0
    max_errors: list[float] = []

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced, bytes_before, bytes_after
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if any(skip in full_name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear) and child.in_features % 8 == 0:
                b_before = child.weight.numel() * child.weight.element_size()
                if child.bias is not None:
                    b_before += child.bias.numel() * child.bias.element_size()
                packed, err = Packed1BitLinear.from_linear(
                    child, group_size=group_size, scale_dtype=scale_dtype
                )
                setattr(parent, name, packed)
                replaced += 1
                bytes_before += b_before
                bytes_after += packed.nbytes()
                max_errors.append(err)
            else:
                _recurse(child, full_name)

    _recurse(model)
    return {
        "layers_packed": replaced,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "mb_before": round(bytes_before / 1e6, 2),
        "mb_after": round(bytes_after / 1e6, 2),
        "compression_ratio": round(bytes_before / max(bytes_after, 1), 2),
        "max_reconstruction_error": round(max(max_errors) if max_errors else 0.0, 6),
        "mean_reconstruction_error": round(float(np.mean(max_errors)) if max_errors else 0.0, 6),
    }


class SelfGatedFP16Sidecar(nn.Module):
    """Tiny FP16 context-self-gated residual sidecar attached to a frozen 1-bit layer.

    When attached as a forward hook on a transformer layer output `h` (shape `[B, T, D]`):
    - Computes linear-probe projection `z = ((h_ref @ w_sense) - mean) / std`.
    - If `ungated=False` and `z <= tau`, gate `g = 0.0` -> EXACT 0.00% drift!
    - If `z > tau` (or `ungated=True`), injects `alpha * g * u_edit` into the residual stream.
    """

    def __init__(
        self,
        w_sense: torch.Tensor,
        u_edit: torch.Tensor,
        proj_mean: float,
        proj_std: float,
        alpha: float = 1.0,
        tau: float = 0.25,
        temperature: float = 0.35,
        ungated: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.register_buffer("w_sense", w_sense.detach().to(dtype))
        self.register_buffer("u_edit", u_edit.detach().to(dtype))
        self.proj_mean = float(proj_mean)
        self.proj_std = max(float(proj_std), 1e-6)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.temperature = max(float(temperature), 1e-4)
        self.ungated = bool(ungated)
        self.last_gate_value: float = 0.0
        self.last_z_score: float = 0.0

    def nbytes(self) -> int:
        return int(
            self.w_sense.numel() * self.w_sense.element_size()
            + self.u_edit.numel() * self.u_edit.element_size()
            + 32  # scalar metadata
        )

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.w_sense.detach().cpu().numpy().tobytes())
        h.update(self.u_edit.detach().cpu().numpy().tobytes())
        h.update(f"{self.proj_mean:.6f}:{self.proj_std:.6f}:{self.alpha:.6f}:{self.tau:.6f}:{self.ungated}".encode())
        return h.hexdigest()

    def apply_to_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply sidecar edit to `[B, T, D]` hidden state tensor."""
        # Use last token (prompt_end / active generation token) to compute context sense score
        h_last = hidden[:, -1, :].float()
        w_s = self.w_sense.to(device=hidden.device, dtype=torch.float32)
        proj = torch.matmul(h_last, w_s)  # [B]
        z = (proj - self.proj_mean) / self.proj_std
        self.last_z_score = float(z.mean().item())

        if self.ungated:
            gate = torch.ones_like(z)
        else:
            # Hard cutoff at tau: strictly 0.0 for all z <= tau (exact 0.00% collateral drift)
            active = (z > self.tau).float()
            smooth = torch.sigmoid((z - self.tau) / self.temperature)
            gate = active * smooth

        self.last_gate_value = float(gate.mean().item())
        if not torch.any(gate > 0.0):
            return hidden

        u_e = self.u_edit.to(device=hidden.device, dtype=hidden.dtype)
        delta = (self.alpha * gate.to(hidden.dtype)).view(-1, 1, 1) * u_e.view(1, 1, -1)
        return hidden + delta
