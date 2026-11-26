"""FLORES+ Spanish (500 items) and General English (200 items) collateral loader + chrF scorer (Part B6)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class ParallelSentence:
    item_id: str
    spanish: str
    english: str


def load_flores_spanish_collateral(
    n_items: int = 500,
    *,
    split: str = "devtest",
    local_jsonl: str | Path | None = None,
) -> list[ParallelSentence]:
    """Load fixed FLORES+ Spanish-English collateral sentences.

    Checks `local_jsonl` or `DSB_FLORES_PATH` env var first for offline/CI reproducibility;
    falls back to Hugging Face `datasets` (`openlanguagedata/flores_plus`).
    """
    src = local_jsonl or os.environ.get("DSB_FLORES_PATH")
    if src:
        path = Path(src)
        rows: list[ParallelSentence] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= n_items:
                    break
                obj = json.loads(line)
                rows.append(
                    ParallelSentence(
                        item_id=str(obj.get("id", f"flores_{idx:04d}")),
                        spanish=str(obj["spanish"]),
                        english=str(obj["english"]),
                    )
                )
        return rows

    from datasets import load_dataset  # lazy import so CI stays lightweight

    ds_es = load_dataset("openlanguagedata/flores_plus", "spa_Latn", split=split)
    ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    limit = min(n_items, len(ds_es), len(ds_en))
    return [
        ParallelSentence(
            item_id=f"flores_{split}_{i:04d}",
            spanish=str(ds_es[i]["text"]),
            english=str(ds_en[i]["text"]),
        )
        for i in range(limit)
    ]


def load_general_english_collateral(
    n_items: int = 200,
    *,
    split: str = "dev",
    local_jsonl: str | Path | None = None,
) -> list[str]:
    """Load 200 fixed general English sentences for collateral NLL tracking."""
    src = local_jsonl or os.environ.get("DSB_ENGLISH_COLLATERAL_PATH")
    if src:
        path = Path(src)
        out: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= n_items:
                    break
                obj = json.loads(line)
                out.append(str(obj["english"] if isinstance(obj, dict) else obj))
        return out

    from datasets import load_dataset

    ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    limit = min(n_items, len(ds_en))
    return [str(ds_en[i]["text"]) for i in range(limit)]


def _char_ngrams(text: str, n: int) -> Counter[str]:
    s = "".join(text.split())
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def compute_chrf(
    hypotheses: list[str],
    references: list[str],
    *,
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute corpus-level chrF score (0 to 100) with zero external dependencies."""
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be non-empty and equal length.")

    prec_sum = 0.0
    rec_sum = 0.0
    valid_orders = 0
    for n in range(1, char_order + 1):
        match_total = 0
        hyp_total = 0
        ref_total = 0
        for h, r in zip(hypotheses, references):
            h_ng = _char_ngrams(h, n)
            r_ng = _char_ngrams(r, n)
            hyp_total += sum(h_ng.values())
            ref_total += sum(r_ng.values())
            match_total += sum((h_ng & r_ng).values())
        if hyp_total > 0 and ref_total > 0:
            prec_sum += match_total / hyp_total
            rec_sum += match_total / ref_total
            valid_orders += 1

    if valid_orders == 0:
        return 0.0
    avg_p = prec_sum / valid_orders
    avg_r = rec_sum / valid_orders
    if avg_p == 0.0 and avg_r == 0.0:
        return 0.0
    beta2 = beta * beta
    return float(100.0 * (1.0 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r))
