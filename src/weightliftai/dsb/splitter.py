"""Frame-level 40/60 dev/sealed-test splitter for DSB-1 (Protocol Part B4)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_dsb_by_frame(
    input_csv: str | Path,
    dev_out_csv: str | Path,
    sealed_test_out_csv: str | Path,
    manifest_out_json: str | Path,
    *,
    dev_ratio: float = 0.40,
    seed: int = 20261126,
    require_approved: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Split DSB items strictly by `frame_id` (stratified by sense) so minimal pairs stay together.

    If `repo_root` is provided, refuses to write `sealed_test_out_csv` inside `repo_root` so
    coding agents cannot inspect sealed test items inside the repository workspace.
    """
    in_path = Path(input_csv).resolve()
    dev_path = Path(dev_out_csv).resolve()
    test_path = Path(sealed_test_out_csv).resolve()
    manifest_path = Path(manifest_out_json).resolve()

    if repo_root is not None:
        root_resolved = Path(repo_root).resolve()
        try:
            test_path.relative_to(root_resolved)
            raise ValueError(
                f"Sealed test path '{test_path}' must be outside repo root '{root_resolved}'."
            )
        except ValueError as exc:
            if "must be outside repo root" in str(exc):
                raise

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if require_approved:
        unapproved = [r["frame_id"] for r in rows if not r.get("approved_by", "").strip()]
        if unapproved:
            raise ValueError(
                f"Cannot lock split: {len(unapproved)} items are missing PI approval (`approved_by`). "
                f"First unapproved: {unapproved[:5]}"
            )

    # Group rows by (sense, frame_id) so minimal pairs (tíguere / tigre) never split
    by_sense: dict[str, dict[str, list[dict[str, str]]]] = {}
    for r in rows:
        sense = r["sense"].strip()
        fid = r["frame_id"].strip()
        by_sense.setdefault(sense, {}).setdefault(fid, []).append(r)

    rng = random.Random(seed)
    dev_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    dev_frames: set[str] = set()
    test_frames: set[str] = set()

    for sense in sorted(by_sense):
        fids = sorted(by_sense[sense].keys())
        rng.shuffle(fids)
        n_dev = max(1, int(round(len(fids) * dev_ratio)))
        d_fids = fids[:n_dev]
        t_fids = fids[n_dev:]
        for fid in d_fids:
            dev_frames.add(fid)
            dev_rows.extend(by_sense[sense][fid])
        for fid in t_fids:
            test_frames.add(fid)
            test_rows.extend(by_sense[sense][fid])

    assert dev_frames.isdisjoint(test_frames), "Frame leakage detected between dev and test splits."

    dev_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with dev_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dev_rows)

    with test_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(test_rows)

    manifest = {
        "protocol": "DSB-1",
        "seed": seed,
        "dev_ratio": dev_ratio,
        "source_csv": in_path.name,
        "source_sha256": _sha256_file(in_path),
        "dev_count": len(dev_rows),
        "dev_frame_count": len(dev_frames),
        "dev_sha256": _sha256_file(dev_path),
        "sealed_test_count": len(test_rows),
        "sealed_test_frame_count": len(test_frames),
        "sealed_test_sha256": _sha256_file(test_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
