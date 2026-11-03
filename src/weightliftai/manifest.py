"""Write reproducible experiment manifests."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest(model_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "project": "WeightLiftAI",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if details:
        manifest["model_inspection"] = details
    return manifest


def save_manifest(manifest: dict[str, Any], output: str) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return destination
