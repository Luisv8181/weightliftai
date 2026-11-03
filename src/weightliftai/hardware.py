"""Hardware discovery with graceful optional-dependency handling."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any


def get_hardware_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "notes": [],
    }

    try:
        import torch
    except ImportError:
        report["notes"].append("PyTorch is not installed; CUDA details are unavailable.")
        report["torch_installed"] = False
        return report

    report["torch_installed"] = True
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_version"] = torch.version.cuda
    if report["cuda_available"]:
        report["gpu_count"] = torch.cuda.device_count()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gib": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    else:
        report["notes"].append(
            "CUDA is unavailable. Install a PyTorch build compatible with your NVIDIA driver "
            "if GPU acceleration is expected."
        )

    report["disk_free_gib"] = (
        round(shutil.disk_usage(".").free / (1024**3), 2)
    )
    return report
