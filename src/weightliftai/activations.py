"""Capture and summarize activations from selected model modules.

This module reports descriptive statistics only. It does not claim that a
module or activation corresponds to a single human-interpretable concept.
"""

from __future__ import annotations

from typing import Any, Iterable


def summarize_tensor(tensor: Any, *, top_k: int = 5) -> dict[str, Any]:
    """Return compact numeric statistics for a tensor without storing its values."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation analysis.") from exc

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, received {type(tensor).__name__}")

    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    result: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(tensor.dtype),
        "numel": values.numel(),
        "finite_fraction": float(finite.float().mean().item()) if values.numel() else 1.0,
    }
    if finite_values.numel() == 0:
        result["all_values_non_finite"] = True
        return result

    absolute = finite_values.abs()
    result.update(
        {
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean_absolute": float(absolute.mean().item()),
            "max_absolute": float(absolute.max().item()),
            "zero_fraction": float((finite_values == 0).float().mean().item()),
        }
    )

    count = min(max(int(top_k), 0), finite_values.numel())
    if count:
        top_values, top_indices = torch.topk(absolute.flatten(), k=count)
        result["largest_absolute_values"] = [
            {"flat_index": int(index.item()), "absolute_value": float(value.item())}
            for value, index in zip(top_values, top_indices)
        ]
    return result


def list_leaf_modules(model: Any, *, contains: str | None = None) -> list[dict[str, str]]:
    """List leaf module names and classes, optionally filtering by substring."""
    modules = []
    for name, module in model.named_modules():
        if not name or any(module.children()):
            continue
        if contains and contains.lower() not in name.lower():
            continue
        modules.append({"name": name, "class": module.__class__.__name__})
    return modules


def capture_activations(
    model: Any,
    inputs: dict[str, Any],
    module_names: Iterable[str],
    *,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run one forward pass and summarize outputs from exact named modules."""
    requested = list(dict.fromkeys(module_names))
    if not requested:
        raise ValueError("Provide at least one exact module name.")

    modules = dict(model.named_modules())
    missing = [name for name in requested if name not in modules]
    if missing:
        examples = list(modules)[:20]
        raise ValueError(
            f"Unknown module name(s): {missing}. Example available names: {examples}"
        )

    captured: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output
            if isinstance(value, (tuple, list)):
                value = next((item for item in value if hasattr(item, "detach")), None)
            elif isinstance(value, dict):
                value = next((item for item in value.values() if hasattr(item, "detach")), None)
            if value is None or not hasattr(value, "detach"):
                captured[name] = {"unsupported_output_type": type(output).__name__}
                return
            captured[name] = summarize_tensor(value, top_k=top_k)
        return hook

    try:
        for name in requested:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        model.eval()
        import torch
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def run_activation_analysis(
    model_id: str,
    prompt: str,
    module_names: Iterable[str],
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Load a causal LM, run a prompt, and return selected activation summaries."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required. Install with: pip install -e '.[core]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    summaries = capture_activations(model, inputs, module_names, top_k=top_k)
    return {
        "model_id": model_id,
        "device": str(device),
        "prompt_token_count": int(inputs["input_ids"].shape[-1]),
        "activation_summaries": summaries,
        "interpretation_note": (
            "These are descriptive statistics for selected module outputs. "
            "They do not establish that a module or individual activation has a "
            "single semantic meaning or causal role."
        ),
    }


def _pool_tensor(tensor: Any, pooling: str = "mean") -> Any:
    """Pool a 1D, 2D, or 3D tensor down to a 1D feature vector."""
    import torch

    val = tensor.detach().float()
    if val.ndim == 1:
        return val
    if val.ndim == 2:
        return val.mean(dim=0)
    if val.ndim == 3:
        if pooling == "mean":
            return val.mean(dim=(0, 1))
        elif pooling == "last":
            return val[:, -1, :].mean(dim=0)
        else:
            raise ValueError(f"Unsupported pooling mode: '{pooling}'. Choose 'mean' or 'last'.")
    return val.reshape(-1, val.shape[-1]).mean(dim=0)


def compare_tensors(
    tensor_a: Any,
    tensor_b: Any,
    *,
    pooling: str = "mean",
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare two activation tensors, computing cosine similarity and divergent dimensions."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation analysis.") from exc

    if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
        raise TypeError("Both tensor_a and tensor_b must be torch.Tensor instances.")

    vec_a = _pool_tensor(tensor_a, pooling=pooling)
    vec_b = _pool_tensor(tensor_b, pooling=pooling)

    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Feature dimensions do not match: {list(vec_a.shape)} vs {list(vec_b.shape)}"
        )

    norm_a = float(torch.linalg.norm(vec_a).item())
    norm_b = float(torch.linalg.norm(vec_b).item())

    if norm_a > 0 and norm_b > 0:
        cosine = float(torch.dot(vec_a, vec_b).item() / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))
    else:
        cosine = 0.0

    l2_dist = float(torch.linalg.norm(vec_a - vec_b).item())
    norm_ratio = float(norm_a / (norm_b + 1e-9))

    diff = (vec_a - vec_b).abs()
    count = min(max(int(top_k), 0), diff.numel())
    top_diffs = []
    if count:
        top_vals, top_indices = torch.topk(diff, k=count)
        top_diffs = [
            {
                "dimension": int(idx.item()),
                "absolute_difference": float(val.item()),
                "val_a": float(vec_a[idx].item()),
                "val_b": float(vec_b[idx].item()),
            }
            for val, idx in zip(top_vals, top_indices)
        ]

    return {
        "pooling": pooling,
        "feature_dim": int(vec_a.numel()),
        "cosine_similarity": cosine,
        "l2_distance": l2_dist,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_ratio": norm_ratio,
        "top_divergent_dimensions": top_diffs,
    }


def compare_activations(
    model: Any,
    inputs_a: dict[str, Any],
    inputs_b: dict[str, Any],
    module_names: Iterable[str],
    *,
    pooling: str = "mean",
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run two forward passes and compare output activations module-by-module."""
    requested = list(dict.fromkeys(module_names))
    if not requested:
        raise ValueError("Provide at least one exact module name.")

    modules = dict(model.named_modules())
    missing = [name for name in requested if name not in modules]
    if missing:
        examples = list(modules)[:20]
        raise ValueError(
            f"Unknown module name(s): {missing}. Example available names: {examples}"
        )

    raw_a: dict[str, Any] = {}
    raw_b: dict[str, Any] = {}
    active_target = raw_a

    def make_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            val = output
            if isinstance(val, (tuple, list)):
                val = next((item for item in val if hasattr(item, "detach")), None)
            elif isinstance(val, dict):
                val = next((item for item in val.values() if hasattr(item, "detach")), None)
            if val is not None and hasattr(val, "detach"):
                active_target[name] = val.detach().cpu()

        return hook

    handles = [modules[name].register_forward_hook(make_hook(name)) for name in requested]
    try:
        model.eval()
        import torch

        with torch.inference_mode():
            active_target = raw_a
            model(**inputs_a)
            active_target = raw_b
            model(**inputs_b)
    finally:
        for handle in handles:
            handle.remove()

    results: dict[str, dict[str, Any]] = {}
    for name in requested:
        if name in raw_a and name in raw_b:
            results[name] = compare_tensors(raw_a[name], raw_b[name], pooling=pooling, top_k=top_k)
        else:
            results[name] = {"error": "Activation tensor not captured for one or both inputs"}

    return results


def run_activation_comparison(
    model_id: str,
    prompt_a: str,
    prompt_b: str,
    module_names: Iterable[str],
    *,
    pooling: str = "mean",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Load a causal LM, run two prompts, and return side-by-side activation comparisons."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required. Install with: pip install -e '.[core]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(device)
    inputs_b = tokenizer(prompt_b, return_tensors="pt").to(device)

    comparisons = compare_activations(
        model, inputs_a, inputs_b, module_names, pooling=pooling, top_k=top_k
    )
    return {
        "model_id": model_id,
        "device": str(device),
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "tokens_a": int(inputs_a["input_ids"].shape[-1]),
        "tokens_b": int(inputs_b["input_ids"].shape[-1]),
        "module_comparisons": comparisons,
        "interpretation_note": (
            "These metrics compare activation distributions between two prompts across requested modules. "
            "A high or low cosine similarity indicates representational alignment or divergence in that module, "
            "not necessarily conceptual equivalence."
        ),
    }
