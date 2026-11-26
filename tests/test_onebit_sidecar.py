"""Unit tests for 1-Bit Packed Linear Layers and Tiny FP16 Self-Gated Residual Sidecars."""

from __future__ import annotations

import torch
import torch.nn as nn

from weightliftai.weight_programming.onebit_sidecar import (
    Packed1BitLinear,
    SelfGatedFP16Sidecar,
    pack_transformer_linears_to_1bit,
)


def test_packed_1bit_linear_bit_exact_on_binary_weights():
    """Packing a native 1-bit (+/- scale) Linear layer must be bit-exact and cut memory >10x."""
    torch.manual_seed(42)
    in_f, out_f, group_size = 512, 256, 128
    lin = nn.Linear(in_f, out_f, bias=False)
    # Construct exact 1-bit weights with per-group scales
    signs = torch.where(torch.randn(out_f, in_f) >= 0, 1.0, -1.0)
    scales = torch.rand(out_f, in_f // group_size) * 0.2 + 0.05
    w_1bit = signs * scales.repeat_interleave(group_size, dim=1)
    lin.weight.data.copy_(w_1bit)

    packed, max_err = Packed1BitLinear.from_linear(lin, group_size=group_size, scale_dtype=torch.float32)
    assert max_err < 1e-6

    x = torch.randn(4, 6, in_f)
    y_ref = lin(x)
    y_packed = packed(x)
    assert torch.allclose(y_ref, y_packed, atol=1e-4)

    # Compression ratio vs FP32 weights should be >20x (and >10x vs FP16)
    fp16_bytes = lin.weight.numel() * 2
    assert packed.nbytes() < fp16_bytes / 10.0


def test_pack_transformer_linears_replaces_in_place_and_skips_lm_head():
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 128)
            self.block = nn.Sequential(nn.Linear(128, 256, bias=False), nn.Linear(256, 128, bias=False))
            self.lm_head = nn.Linear(128, 32, bias=False)

    m = MiniModel()
    report = pack_transformer_linears_to_1bit(m, group_size=64, scale_dtype=torch.float16)
    assert report["layers_packed"] == 2
    assert report["compression_ratio"] > 10.0
    assert isinstance(m.block[0], Packed1BitLinear)
    assert isinstance(m.block[1], Packed1BitLinear)
    assert isinstance(m.lm_head, nn.Linear)


def test_self_gated_fp16_sidecar_exact_zero_drift_when_closed():
    """When z <= tau, SelfGatedFP16Sidecar must return bit-exact identical hidden states."""
    d = 2560
    w_sense = torch.zeros(d)
    w_sense[0] = 1.0
    u_edit = torch.ones(d) * 0.5

    sidecar = SelfGatedFP16Sidecar(
        w_sense=w_sense,
        u_edit=u_edit,
        proj_mean=0.0,
        proj_std=1.0,
        alpha=2.0,
        tau=0.25,
        ungated=False,
    )
    # Footpint for d=2560 FP16 must be tiny (< 20 KB << 5 MB)
    assert sidecar.nbytes() < 20_000
    assert len(sidecar.sha256()) == 64

    # Control input with z = -1.0 <= tau -> exact 0.00% drift
    h_ctrl = torch.zeros(1, 5, d, dtype=torch.float16)
    h_ctrl[0, -1, 0] = -1.0
    out_ctrl = sidecar.apply_to_hidden(h_ctrl)
    assert sidecar.last_gate_value == 0.0
    assert torch.equal(out_ctrl, h_ctrl)

    # Target input with z = +3.0 > tau -> gate opens and edits hidden state
    h_tgt = torch.zeros(1, 5, d, dtype=torch.float16)
    h_tgt[0, -1, 0] = 3.0
    out_tgt = sidecar.apply_to_hidden(h_tgt)
    assert sidecar.last_gate_value > 0.9
    assert not torch.equal(out_tgt, h_tgt)
