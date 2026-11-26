"""Unit tests for zero-dependency GGUF v3 Q1_0 1-bit loader and Packed1BitEmbedding."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import torch

from weightliftai.weight_programming.gguf_1bit_loader import (
    GGML_TYPE_Q1_0,
    GGUFQ10Reader,
    Packed1BitEmbedding,
)
from weightliftai.weight_programming.onebit_sidecar import Packed1BitLinear


def test_packed_1bit_embedding_matches_unpacked_lookup():
    torch.manual_seed(7)
    lin = torch.nn.Linear(256, 32, bias=False)
    signs = torch.where(torch.randn(32, 256) >= 0, 1.0, -1.0)
    scales = torch.rand(32, 2) * 0.1 + 0.05
    lin.weight.data.copy_(signs * scales.repeat_interleave(128, dim=1))

    packed_lin, err = Packed1BitLinear.from_linear(lin, group_size=128, scale_dtype=torch.float32)
    assert err < 1e-6

    emb = Packed1BitEmbedding(packed_lin)
    ids = torch.tensor([[0, 5, 31], [3, 12, 7]])
    out = emb(ids)
    expected = lin.weight.data[ids]
    assert out.shape == (2, 3, 256)
    assert torch.allclose(out, expected, atol=1e-5)


def test_gguf_q1_0_reader_parses_synthetic_q1_0_file(tmp_path: Path):
    """Build a minimal valid GGUF v3 file with one Q1_0 tensor (128x4) and verify exact loading."""
    gguf_file = tmp_path / "tiny_q1_0.gguf"
    in_f, out_f = 128, 4
    scales_f16 = np.array([[0.25], [0.50], [0.125], [0.75]], dtype=np.float16)
    signs_u8 = np.full((out_f, in_f // 8), 0xAA, dtype=np.uint8)  # alternating 0/1 bits

    # Pack into 18-byte blocks (2-byte scale + 16-byte signs)
    block_bytes = bytearray()
    for r in range(out_f):
        block_bytes.extend(scales_f16[r].tobytes())
        block_bytes.extend(signs_u8[r].tobytes())

    with gguf_file.open("wb") as f:
        # Header: magic, version=3, n_tensors=1, n_kv=0
        f.write(struct.pack("<4sIQQ", b"GGUF", 3, 1, 0))
        # Tensor info: name, ndims=2, dims=(128, 4), type=41 (Q1_0), offset=0
        tname = b"test.weight"
        f.write(struct.pack("<Q", len(tname)) + tname)
        f.write(struct.pack("<IQQIQ", 2, in_f, out_f, GGML_TYPE_Q1_0, 0))
        # Pad to 32-byte alignment
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(block_bytes)

    reader = GGUFQ10Reader(gguf_file)
    assert "test.weight" in reader.tensors
    packed = reader.read_q1_0_as_packed_linear("test.weight", scale_dtype=torch.float32)
    assert packed.in_features == 128
    assert packed.out_features == 4
    assert torch.allclose(packed.scales.squeeze(-1), torch.tensor([0.25, 0.50, 0.125, 0.75]))
    w = packed.unpack_weight(dtype=torch.float32)
    assert w.shape == (4, 128)
    # 0xAA = 0b10101010 -> bit 0 is 0 (-scale), bit 1 is 1 (+scale)
    assert float(w[0, 0]) == -0.25
    assert float(w[0, 1]) == +0.25
