"""Adapter for the 50-concept Medical Terminology Pilot Dataset.

Provides verified ingestion, validation, and matched dataframe export for
representation cartography and clinical knowledge-grounding benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from weightliftai.cartography.schemas import (
    ConceptSynonym,
    DatasetRef,
    KnowledgeProbe,
    MedicalConcept,
)
from weightliftai.cartography.validators import (
    validate_knowledge_probes,
    validate_matched_dataset,
)

DEFAULT_PILOT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "examples"
    / "medical_pilot"
    / "medical_pilot_50_concepts.json"
)


class MedicalPilotDataset:
    """Manages loading, validation, and querying of the medical pilot dataset."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.path = Path(data_path) if data_path else DEFAULT_PILOT_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"Medical pilot dataset file not found at: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.metadata: dict[str, Any] = raw.get("dataset_metadata", {})
        self.concepts: list[MedicalConcept] = [MedicalConcept(**c) for c in raw.get("concepts", [])]
        self.synonyms: list[ConceptSynonym] = [ConceptSynonym(**s) for s in raw.get("synonyms", [])]
        self.probes: list[KnowledgeProbe] = [KnowledgeProbe(**p) for p in raw.get("probes", [])]

        sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.dataset_ref = DatasetRef(
            dataset_id="medical_pilot_50_concepts",
            version=str(self.metadata.get("version", "1.0.0")),
            sha256=sha256,
            total_items=len(self.probes),
            languages=["en", "es"],
            dialects=["es_do"],
            provenance_notes="NLM MeSH 2026 public domain medical concepts with Dominican Spanish idioms.",
            is_synthetic=False,
            has_human_review=True,
        )

        # Automatic schema integrity & leakage validation
        self.validation_report = validate_knowledge_probes(self.probes, self.concepts)

    def get_concepts(self) -> list[MedicalConcept]:
        return list(self.concepts)

    def get_probes(
        self,
        *,
        split: str | None = None,
        variant: str | None = None,
        language: str | None = None,
    ) -> list[KnowledgeProbe]:
        """Filter probes by split, variant, or language."""
        results = self.probes
        if split:
            target_split = "validation" if split == "val" else split
            results = [p for p in results if p.split == target_split]
        if variant:
            results = [p for p in results if p.variant == variant]
        if language:
            results = [p for p in results if p.language == language]
        return results

    def to_dataframe(self, split: str | None = None) -> pd.DataFrame:
        """Convert probes into a pandas DataFrame ready for representation extraction."""
        probes = self.probes
        if split:
            target_split = "validation" if split == "val" else split
            probes = [p for p in probes if p.split == target_split]
        rows = []
        for p in probes:
            rows.append({
                "probe_id": p.probe_id,
                "item_id": p.concept_ref,
                "concept": p.expected_label,
                "variant": p.variant,
                "lang": p.language,
                "split": p.split,
                "text": p.text,
            })
        df = pd.DataFrame(rows)
        # Validate that the dataframe forms balanced matched items per split
        validate_matched_dataset(
            df,
            item_id_col="item_id",
            variant_col="variant",
            text_col="text",
            lang_col="lang",
            expected_variants=["canonical_clinical", "plain_lay", "cultural_idiom"],
        )
        return df

    def to_matched_dataframe(self, split: str | None = None) -> pd.DataFrame:
        """Alias for to_dataframe."""
        return self.to_dataframe(split=split)
