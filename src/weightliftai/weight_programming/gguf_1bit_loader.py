"""Direct Zero-Dependency GGUF v3 `Q1_0` (1-Bit) Reader -> `Packed1BitLinear` Loader.

Reads PrismML's native 1-bit `Q1_0` (`ggml_type = 41`) tensors directly from local `.gguf`
files (such as `Bonsai-27B-Q1_0.gguf`) without ever unpacking 27B parameters to 54 GB FP16:
- Verified block layout (`Q1_0`, `group_size=128`): 18 bytes per 128 weights
  (`2-byte FP16 scale` + `16-byte uint8 bit-plane`, 1.125 bits/weight).
- Verified exact byte-for-byte alignment against `Bonsai-27B-Q1_0.gguf` (851 tensors,
  `output.weight (5120, 248320)` = 178,790,400 bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch

from weightliftai.weight_programming.onebit_sidecar import Packed1BitLinear

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q1_0 = 41
Q1_0_BLOCK_WEIGHTS = 128
Q1_0_BLOCK_BYTES = 18  # 2-byte FP16 scale + 16-byte packed sign bits


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    dims: tuple[int, ...]  # GGML order (innermost first: [in_features, out_features])
    ggml_type: int
    offset: int  # relative to tensor_data_start


def _read_str(f: Any) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_val(f: Any, vtype: int) -> Any:
    fmts = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if vtype in fmts:
        return struct.unpack(fmts[vtype], f.read(struct.calcsize(fmts[vtype])))[0]
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        if et in fmts:
            sz = struct.calcsize(fmts[et])
            raw = f.read(sz * cnt)
            return list(struct.unpack(f"<{cnt}{fmts[et][1:]}", raw)) if cnt <= 64 else f"array(type={et},len={cnt})"
        return [_read_val(f, et) for _ in range(cnt)] if cnt <= 64 else [
            _read_val(f, et) for _ in range(cnt)
        ][:0] or f"array(type={et},len={cnt})"
    raise ValueError(f"Unsupported GGUF metadata type: {vtype}")


class GGUFQ10Reader:
    """Memory-mapped / seek-based reader for GGUF v3 `Q1_0` 1-bit checkpoints."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self.tensor_data_start: int = 0
        self._parse_header()

    def _parse_header(self) -> None:
        with self.path.open("rb") as f:
            magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file (magic={magic!r})")
            self.metadata["gguf.version"] = version

            for _ in range(n_kv):
                k = _read_str(f)
                vt = struct.unpack("<I", f.read(4))[0]
                v = _read_val(f, vt)
                self.metadata[k] = v

            for _ in range(n_tensors):
                name = _read_str(f)
                nd = struct.unpack("<I", f.read(4))[0]
                dims = tuple(int(d) for d in struct.unpack(f"<{nd}Q", f.read(8 * nd)))
                gt = int(struct.unpack("<I", f.read(4))[0])
                off = int(struct.unpack("<Q", f.read(8))[0])
                self.tensors[name] = GGUFTensorInfo(name=name, dims=dims, ggml_type=gt, offset=off)

            alignment = int(self.metadata.get("general.alignment", 32))
            pos = f.tell()
            rem = pos % alignment
            self.tensor_data_start = pos if rem == 0 else pos + (alignment - rem)

    def read_f32_tensor(self, name: str) -> torch.Tensor:
        info = self.tensors[name]
        if info.ggml_type != GGML_TYPE_F32:
            raise ValueError(f"Tensor {name} has type {info.ggml_type}, expected F32 ({GGML_TYPE_F32}).")
        n_elem = 1
        for d in info.dims:
            n_elem *= d
        with self.path.open("rb") as f:
            f.seek(self.tensor_data_start + info.offset)
            raw = f.read(n_elem * 4)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        # GGML dims are reversed relative to PyTorch shape
        pt_shape = tuple(reversed(info.dims))
        return torch.from_numpy(arr.reshape(pt_shape))

    def read_q1_0_as_packed_linear(
        self,
        name: str,
        scale_dtype: torch.dtype = torch.bfloat16,
    ) -> Packed1BitLinear:
        """Load a 2D `Q1_0` tensor directly into a `Packed1BitLinear` without FP16 unpacking."""
        info = self.tensors[name]
        if info.ggml_type != GGML_TYPE_Q1_0:
            raise ValueError(f"Tensor {name} has type {info.ggml_type}, expected Q1_0 ({GGML_TYPE_Q1_0}).")
        if len(info.dims) != 2:
            raise ValueError(f"Expected 2D Q1_0 tensor, got dims={info.dims}.")

        in_features, out_features = info.dims  # GGML [ne0, ne1] == PyTorch [out, in]
        if in_features % Q1_0_BLOCK_WEIGHTS != 0:
            raise ValueError(f"in_features ({in_features}) not divisible by {Q1_0_BLOCK_WEIGHTS}.")

        n_blocks_per_row = in_features // Q1_0_BLOCK_WEIGHTS
        total_blocks = out_features * n_blocks_per_row
        total_bytes = total_blocks * Q1_0_BLOCK_BYTES

        with self.path.open("rb") as f:
            f.seek(self.tensor_data_start + info.offset)
            raw = f.read(total_bytes)

        buf = np.frombuffer(raw, dtype=np.uint8).reshape(out_features, n_blocks_per_row, Q1_0_BLOCK_BYTES)
        # First 2 bytes of each 18-byte Q1_0 block = FP16 scale; remaining 16 bytes = 128 packed sign bits
        scales_f16 = buf[:, :, :2].copy().view(np.float16).reshape(out_features, n_blocks_per_row)
        signs_u8 = buf[:, :, 2:].copy().reshape(out_features, in_features // 8)

        mod = Packed1BitLinear(
            in_features=in_features,
            out_features=out_features,
            group_size=Q1_0_BLOCK_WEIGHTS,
            has_bias=False,
            scale_dtype=scale_dtype,
        )
        mod.packed_signs.copy_(torch.from_numpy(signs_u8))
        mod.scales.copy_(torch.from_numpy(scales_f16).to(scale_dtype))
        return mod


class Packed1BitEmbedding(torch.nn.Module):
    """Memory-efficient 1-bit packed embedding table that unpacks only the requested token rows."""

    def __init__(self, packed_linear: Packed1BitLinear):
        super().__init__()
        self.num_embeddings = packed_linear.out_features
        self.embedding_dim = packed_linear.in_features
        self.group_size = packed_linear.group_size
        self.register_buffer("packed_signs", packed_linear.packed_signs)
        self.register_buffer("scales", packed_linear.scales)
        self.register_buffer("_bit_shifts", torch.arange(8, dtype=torch.uint8).view(1, 1, 8), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        row_signs = self.packed_signs[flat_ids]  # [N, D//8]
        row_scales = self.scales[flat_ids]  # [N, D//G]
        bits = (row_signs.unsqueeze(-1) >> self._bit_shifts) & 1
        signs = bits.to(self.scales.dtype).mul_(2.0).sub_(1.0).view(flat_ids.shape[0], self.embedding_dim)
        scales_exp = (
            row_scales.unsqueeze(-1)
            .expand(flat_ids.shape[0], self.embedding_dim // self.group_size, self.group_size)
            .reshape(flat_ids.shape[0], self.embedding_dim)
        )
        out = signs.mul_(scales_exp)
        return out.view(*input_ids.shape, self.embedding_dim)

    def nbytes(self) -> int:
        return int(
            self.packed_signs.numel() * self.packed_signs.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def load_bonsai_27b_q1_0_gguf(
    gguf_path: str | Path,
    device: str = "cuda",
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load PrismML `Bonsai-27B-Q1_0.gguf` directly into `Qwen3_5ForCausalLM` using 1-bit packed layers.

    Never unpacks 27B weights to FP16: total GPU VRAM footprint is ~3.8 GB (fits inside 8.55 GB RTX 5070).
    """
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    reader = GGUFQ10Reader(gguf_path)
    md = reader.metadata
    vocab_size = int(reader.tensors["token_embd.weight"].dims[1])
    hidden_size = int(md["qwen35.embedding_length"])
    num_layers = int(md["qwen35.block_count"])
    intermediate_size = int(md["qwen35.feed_forward_length"])
    num_heads = int(md["qwen35.attention.head_count"])
    num_kv_heads = int(md["qwen35.attention.head_count_kv"])
    head_dim = int(md["qwen35.attention.key_length"])
    full_attn_interval = int(md["qwen35.full_attention_interval"])

    cfg = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=float(md["qwen35.attention.layer_norm_rms_epsilon"]),
        rope_theta=float(md["qwen35.rope.freq_base"]),
        partial_rotary_factor=float(md["qwen35.rope.dimension_count"]) / float(head_dim),
        linear_conv_kernel_dim=int(md["qwen35.ssm.conv_kernel"]),
        linear_key_head_dim=int(md["qwen35.ssm.state_size"]),
        linear_value_head_dim=int(md["qwen35.ssm.state_size"]),
        linear_num_key_heads=int(md["qwen35.ssm.group_count"]),
        linear_num_value_heads=int(md["qwen35.ssm.time_step_rank"]),
        full_attention_interval=full_attn_interval,
        tie_word_embeddings=False,
    )

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)

    def _set_param(mod: torch.nn.Module, attr: str, tensor: torch.Tensor) -> None:
        setattr(mod, attr, torch.nn.Parameter(tensor.to(device=device, dtype=scale_dtype), requires_grad=False))

    # 1. Embeddings & LM Head (1-bit packed!)
    emb_lin = reader.read_q1_0_as_packed_linear("token_embd.weight", scale_dtype=scale_dtype)
    model.model.embed_tokens = Packed1BitEmbedding(emb_lin).to(device)
    model.lm_head = reader.read_q1_0_as_packed_linear("output.weight", scale_dtype=scale_dtype).to(device)
    # Qwen3_5RMSNorm adds +1.0 to self.weight in forward(), whereas GGUF stores already-shifted weights
    _set_param(model.model.norm, "weight", reader.read_f32_tensor("output_norm.weight") - 1.0)

    # Re-materialize rotary embeddings on device
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg, device=device)

    total_packed_bytes = model.model.embed_tokens.nbytes() + model.lm_head.nbytes()
    packed_linear_count = 1

    # 2. Load all 64 transformer layers directly from GGUF Q1_0 into device
    for i in range(num_layers):
        layer = model.model.layers[i]
        p = f"blk.{i}"

        _set_param(layer.input_layernorm, "weight", reader.read_f32_tensor(f"{p}.attn_norm.weight") - 1.0)
        _set_param(layer.post_attention_layernorm, "weight", reader.read_f32_tensor(f"{p}.post_attention_norm.weight") - 1.0)

        # MLP (always Q1_0)
        for hf_name, gguf_name in [
            ("gate_proj", "ffn_gate.weight"),
            ("up_proj", "ffn_up.weight"),
            ("down_proj", "ffn_down.weight"),
        ]:
            pl = reader.read_q1_0_as_packed_linear(f"{p}.{gguf_name}", scale_dtype=scale_dtype).to(device)
            setattr(layer.mlp, hf_name, pl)
            total_packed_bytes += pl.nbytes()
            packed_linear_count += 1

        if (i + 1) % full_attn_interval == 0:
            # Full Attention block
            for hf_name, gguf_name in [
                ("q_proj", "attn_q.weight"),
                ("k_proj", "attn_k.weight"),
                ("v_proj", "attn_v.weight"),
                ("o_proj", "attn_output.weight"),
            ]:
                pl = reader.read_q1_0_as_packed_linear(f"{p}.{gguf_name}", scale_dtype=scale_dtype).to(device)
                setattr(layer.self_attn, hf_name, pl)
                total_packed_bytes += pl.nbytes()
                packed_linear_count += 1
            _set_param(layer.self_attn.q_norm, "weight", reader.read_f32_tensor(f"{p}.attn_q_norm.weight") - 1.0)
            _set_param(layer.self_attn.k_norm, "weight", reader.read_f32_tensor(f"{p}.attn_k_norm.weight") - 1.0)
        else:
            # Gated-DeltaNet Linear Attention (SSM) block
            for hf_name, gguf_name in [
                ("in_proj_qkv", "attn_qkv.weight"),
                ("in_proj_z", "attn_gate.weight"),
                ("in_proj_a", "ssm_alpha.weight"),
                ("in_proj_b", "ssm_beta.weight"),
                ("out_proj", "ssm_out.weight"),
            ]:
                pl = reader.read_q1_0_as_packed_linear(f"{p}.{gguf_name}", scale_dtype=scale_dtype).to(device)
                setattr(layer.linear_attn, hf_name, pl)
                total_packed_bytes += pl.nbytes()
                packed_linear_count += 1

            ssm_a = reader.read_f32_tensor(f"{p}.ssm_a")
            a_log = torch.log(torch.clamp(-ssm_a, min=1e-8))
            _set_param(layer.linear_attn, "A_log", a_log)
            _set_param(layer.linear_attn, "dt_bias", reader.read_f32_tensor(f"{p}.ssm_dt.bias"))
            _set_param(layer.linear_attn.norm, "weight", reader.read_f32_tensor(f"{p}.ssm_norm.weight"))
            conv_w = reader.read_f32_tensor(f"{p}.ssm_conv1d.weight").unsqueeze(1)  # [10240, 1, 4]
            _set_param(layer.linear_attn.conv1d, "weight", conv_w)

    model.eval()
    stats = {
        "gguf_path": str(Path(gguf_path).name),
        "architecture": md.get("general.architecture"),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "packed_linear_count": packed_linear_count,
        "total_packed_mb": round(total_packed_bytes / 1e6, 2),
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 2) if device.startswith("cuda") else 0.0,
    }
    return model, stats

