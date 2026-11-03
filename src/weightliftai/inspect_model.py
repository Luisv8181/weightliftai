"""Model metadata and optional parameter inspection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def inspect_model(
    model_id: str,
    *,
    load_weights: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Inspect a Hugging Face causal language model.

    By default, only the model configuration is loaded. Set load_weights=True
    to instantiate parameters and calculate exact parameter counts.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required. Install with: pip install -e '.[core]'"
        ) from exc

    config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    result: dict[str, Any] = {
        "model_id": model_id,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "weight_loading_performed": False,
    }

    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        parameters = list(model.named_parameters())
        result["weight_loading_performed"] = True
        result["parameter_count"] = sum(param.numel() for _, param in parameters)
        result["trainable_parameter_count"] = sum(
            param.numel() for _, param in parameters if param.requires_grad
        )
        result["parameter_dtypes"] = dict(
            Counter(str(param.dtype) for _, param in parameters)
        )
        result["modules"] = [
            {"name": name, "class": module.__class__.__name__}
            for name, module in model.named_modules()
            if name and not any(True for _ in module.children())
        ][:500]
        result["tensors"] = [
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
            }
            for name, param in parameters[:500]
        ]
        del model

    return result


def inspect_local_directory(path: str) -> dict[str, Any]:
    """Return basic file metadata for a local model directory."""
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    files = [p for p in root.rglob("*") if p.is_file()]
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "total_size_bytes": sum(p.stat().st_size for p in files),
        "config_present": (root / "config.json").exists(),
        "safetensors_files": [p.name for p in files if p.suffix == ".safetensors"],
        "pytorch_files": [p.name for p in files if p.name in {"pytorch_model.bin", "model.bin"}],
    }
