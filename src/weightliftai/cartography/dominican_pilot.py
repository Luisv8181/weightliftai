"""Adapter for the 50-concept Culturally Grounded Dominican Spanish Pilot Dataset.

Provides verified ingestion, validation, and matched dataframe export for
representation cartography, cultural alignment, and pragmatic translation research.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from weightliftai.cartography.schemas import DatasetRef
from weightliftai.cartography.validators import validate_matched_dataset

DEFAULT_DOMINICAN_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "examples"
    / "dominican_pilot"
    / "dominican_pilot_50.json"
)


class DominicanPilotDataset:
    """Manages loading, validation, and querying of the Dominican Spanish pilot dataset."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.path = Path(data_path) if data_path else DEFAULT_DOMINICAN_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"Dominican pilot dataset file not found at: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.dataset_id: str = raw.get("dataset_id", "dominican_spanish_pilot_50")
        self.version: str = raw.get("version", "1.0")
        self.description: str = raw.get("description", "")
        self.license: str = raw.get("license", "CC-BY-4.0")
        self.governance: str = raw.get("governance", "")
        self.items: list[dict[str, Any]] = raw.get("items", [])

        if len(self.items) != 50:
            raise ValueError(f"Expected 50 items in Dominican pilot, found {len(self.items)}")

        sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.dataset_ref = DatasetRef(
            dataset_id=self.dataset_id,
            version=self.version,
            sha256=sha256,
            total_items=len(self.items),
            languages=["es", "en"],
            dialects=["es_do"],
            provenance_notes="Culturally grounded Dominican Spanish phrases matched with neutral Spanish and English pragmatic controls.",
            is_synthetic=False,
            has_human_review=True,
        )

        self.validation_report = self._validate_dataset()

    def _validate_dataset(self) -> dict[str, Any]:
        """Validate integrity, uniqueness, and completeness of dataset items."""
        concept_ids = set()
        splits = {"train": 0, "val": 0, "test": 0}
        categories = set()

        for idx, item in enumerate(self.items):
            cid = item.get("concept_id")
            if not cid:
                raise ValueError(f"Item at index {idx} missing concept_id")
            if cid in concept_ids:
                raise ValueError(f"Duplicate concept_id found: {cid}")
            concept_ids.add(cid)

            for key in ("dominican_text", "neutral_spanish_control", "english_pragmatic_intent", "negative_control"):
                val = item.get(key, "").strip()
                if not val:
                    raise ValueError(f"Item {cid} has empty field '{key}'")

            split = item.get("split", "train")
            if split not in splits:
                raise ValueError(f"Item {cid} has invalid split: {split}")
            splits[split] += 1
            categories.add(item.get("category", "unspecified"))

        return {
            "status": "valid",
            "total_concepts": len(concept_ids),
            "splits": splits,
            "unique_categories": len(categories),
            "sha256": self.dataset_ref.sha256,
        }

    def to_matched_dataframe(self, split: str | None = None) -> pd.DataFrame:
        """Export dataset as a matched pair/triad DataFrame for layer atlas and representation mapping."""
        records = []
        for item in self.items:
            if split and split != "all" and item.get("split") != split:
                continue
            
            # Dominican variant (target)
            records.append({
                "pair_id": f"{item['concept_id']}_dom",
                "concept_id": item["concept_id"],
                "text": item["dominican_text"],
                "variant": "dominican_spanish",
                "language": "es",
                "dialect": "es_do",
                "category": item.get("category", "general"),
                "split": item.get("split", "train"),
                "speaker_region": item.get("speaker_region", "general_do"),
                "negative_control": item.get("negative_control", ""),
            })
            
            # Neutral Spanish control
            records.append({
                "pair_id": f"{item['concept_id']}_neutral",
                "concept_id": item["concept_id"],
                "text": item["neutral_spanish_control"],
                "variant": "neutral_spanish",
                "language": "es",
                "dialect": "es_neutral",
                "category": item.get("category", "general"),
                "split": item.get("split", "train"),
                "speaker_region": "neutral",
                "negative_control": item.get("negative_control", ""),
            })

            # English pragmatic translation reference
            records.append({
                "pair_id": f"{item['concept_id']}_en_pragmatic",
                "concept_id": item["concept_id"],
                "text": item["english_pragmatic_intent"],
                "variant": "english_pragmatic",
                "language": "en",
                "dialect": "en_us",
                "category": item.get("category", "general"),
                "split": item.get("split", "train"),
                "speaker_region": "us",
                "negative_control": item.get("negative_control", ""),
            })

        df = pd.DataFrame.from_records(records)
        return df

    def to_dataframe(self, split: str | None = None) -> pd.DataFrame:
        """Alias for to_matched_dataframe."""
        return self.to_matched_dataframe(split=split)
