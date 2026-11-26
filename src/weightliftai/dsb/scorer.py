"""Dominican Sense Benchmark (DSB-1) scorer, statistical tests, and sealed-file access guard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


class SealedTestAccessError(PermissionError):
    """Raised when code attempts to read a sealed DSB test file without DSB_EVAL_KEY."""


@dataclass(frozen=True)
class DSBItem:
    frame_id: str
    sense: str  # "social" | "literal" | "ambiguous" | "panhispanic_idiom" | "team_name"
    spelling: str  # "tíguere" | "tigre" | "tiguere"
    cue_position: str  # "before" | "after" | "both" | "none"
    topic: str
    spanish: str
    social_glosses: tuple[str, str, str]
    literal_glosses: tuple[str, str, str]
    approved_by: str = ""
    approved_date: str = ""
    notes: str = ""
    sense_group: str = "primary"
    literal_subtype: str = ""
    frame_pair_id: str = ""
    decisive_cue: str = ""
    scoring_span: dict[str, Any] | None = None


@dataclass(frozen=True)
class ItemScore:
    frame_id: str
    sense: str
    spelling: str
    cue_position: str
    topic: str
    mean_social_score: float
    mean_literal_score: float
    margin: float  # correct - wrong (for ambiguous: literal - social)
    correct: bool | None  # None for ambiguous items
    span_margin: float | None = None
    span_correct: bool | None = None


def _is_sealed_path(path: Path) -> bool:
    name = path.name.lower()
    return "sealed" in name


def load_dsb_csv(path: str | Path, *, is_sealed: bool = False) -> list[DSBItem]:
    """Load DSB items from CSV. Refuses to open sealed test files unless `DSB_EVAL_KEY` is set."""
    csv_path = Path(path)
    if is_sealed or _is_sealed_path(csv_path):
        eval_key = os.environ.get("DSB_EVAL_KEY", "").strip()
        if not eval_key:
            raise SealedTestAccessError(
                f"Refusing to load sealed DSB test file '{csv_path.name}' without DSB_EVAL_KEY set."
            )

    items: list[DSBItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = json.loads(row["gloss_options"])
            soc = tuple(gloss["social"])
            lit = tuple(gloss["literal"])
            if len(soc) != 3 or len(lit) != 3:
                raise ValueError(
                    f"Frame {row['frame_id']} must have exactly 3 paraphrases per sense "
                    f"(got social={len(soc)}, literal={len(lit)})."
                )
            raw_span = row.get("scoring_span", "").strip()
            parsed_span = json.loads(raw_span) if raw_span else None
            items.append(
                DSBItem(
                    frame_id=row["frame_id"].strip(),
                    sense=row["sense"].strip(),
                    spelling=row["spelling"].strip(),
                    cue_position=row["cue_position"].strip(),
                    topic=row["topic"].strip(),
                    spanish=row["spanish"].strip(),
                    social_glosses=soc,  # type: ignore[arg-type]
                    literal_glosses=lit,  # type: ignore[arg-type]
                    approved_by=row.get("approved_by", "").strip(),
                    approved_date=row.get("approved_date", "").strip(),
                    notes=row.get("notes", "").strip(),
                    sense_group=row.get("sense_group", "primary").strip() or "primary",
                    literal_subtype=row.get("literal_subtype", "").strip(),
                    frame_pair_id=row.get("frame_pair_id", "").strip() or row["frame_id"].strip(),
                    decisive_cue=row.get("decisive_cue", "").strip(),
                    scoring_span=parsed_span,
                )
            )
    return items


def score_item_forced_choice(
    item: DSBItem,
    score_fn: Callable[[str, str], float],
    span_score_fn: Callable[[str, str, tuple[int, int]], float] | None = None,
) -> ItemScore:
    """Score a single DSB item by averaging 3 paraphrases per sense (whole-sentence and optional span-only)."""
    soc_scores = [float(score_fn(item.spanish, g)) for g in item.social_glosses]
    lit_scores = [float(score_fn(item.spanish, g)) for g in item.literal_glosses]
    mean_soc = float(np.mean(soc_scores))
    mean_lit = float(np.mean(lit_scores))

    if item.sense == "social":
        margin = mean_soc - mean_lit
        correct: bool | None = margin > 0.0
    elif item.sense == "literal":
        margin = mean_lit - mean_soc
        correct = margin > 0.0
    else:
        # Exploratory ambiguous: positive margin means model defaulted to literal "tiger"
        margin = mean_lit - mean_soc
        correct = None

    span_margin: float | None = None
    span_correct: bool | None = None
    if span_score_fn is not None and item.scoring_span and "pairs" in item.scoring_span:
        pairs = item.scoring_span["pairs"]
        s_sp = [
            float(span_score_fn(item.spanish, g, tuple(p["social_span"])))
            for g, p in zip(item.social_glosses, pairs)
        ]
        l_sp = [
            float(span_score_fn(item.spanish, g, tuple(p["literal_span"])))
            for g, p in zip(item.literal_glosses, pairs)
        ]
        ms_sp = float(np.mean(s_sp))
        ml_sp = float(np.mean(l_sp))
        if item.sense == "social":
            span_margin = ms_sp - ml_sp
            span_correct = span_margin > 0.0
        elif item.sense == "literal":
            span_margin = ml_sp - ms_sp
            span_correct = span_margin > 0.0
        else:
            span_margin = ml_sp - ms_sp
            span_correct = None

    return ItemScore(
        frame_id=item.frame_id,
        sense=item.sense,
        spelling=item.spelling,
        cue_position=item.cue_position,
        topic=item.topic,
        mean_social_score=mean_soc,
        mean_literal_score=mean_lit,
        margin=margin,
        correct=correct,
        span_margin=span_margin,
        span_correct=span_correct,
    )


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return `(mean, ci_low, ci_high)` using percentile bootstrap over items."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    mean_val = float(arr.mean())
    if arr.size == 1:
        return (mean_val, mean_val, mean_val)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (mean_val, lo, hi)


def evaluate_dataset(
    items: list[DSBItem],
    score_fn: Callable[[str, str], float],
    *,
    span_score_fn: Callable[[str, str, tuple[int, int]], float] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a DSB dataset with 3-paraphrase forced choice (whole-sentence and optional span-only)."""
    per_item = [score_item_forced_choice(it, score_fn, span_score_fn=span_score_fn) for it in items]

    def _summarize(subset: list[ItemScore]) -> dict[str, Any]:
        if not subset:
            return {"n": 0, "accuracy": None, "margin": None, "span_accuracy": None, "span_margin": None}
        margins = [x.margin for x in subset]
        m_mean, m_lo, m_hi = bootstrap_ci(margins, n_resamples=n_bootstrap, seed=seed)
        labeled = [1.0 if x.correct else 0.0 for x in subset if x.correct is not None]
        if labeled:
            a_mean, a_lo, a_hi = bootstrap_ci(labeled, n_resamples=n_bootstrap, seed=seed)
            acc_obj: dict[str, float] | None = {
                "mean": round(a_mean, 4),
                "ci_low": round(a_lo, 4),
                "ci_high": round(a_hi, 4),
            }
        else:
            acc_obj = None

        sp_margins = [x.span_margin for x in subset if x.span_margin is not None]
        sp_labeled = [1.0 if x.span_correct else 0.0 for x in subset if x.span_correct is not None]
        sp_acc_obj: dict[str, float] | None = None
        sp_m_obj: dict[str, float] | None = None
        if sp_margins:
            sm, slo, shi = bootstrap_ci(sp_margins, n_resamples=n_bootstrap, seed=seed)
            sp_m_obj = {"mean": round(sm, 4), "ci_low": round(slo, 4), "ci_high": round(shi, 4)}
        if sp_labeled:
            sa, salo, sahi = bootstrap_ci(sp_labeled, n_resamples=n_bootstrap, seed=seed)
            sp_acc_obj = {"mean": round(sa, 4), "ci_low": round(salo, 4), "ci_high": round(sahi, 4)}

        return {
            "n": len(subset),
            "accuracy": acc_obj,
            "margin": {
                "mean": round(m_mean, 4),
                "ci_low": round(m_lo, 4),
                "ci_high": round(m_hi, 4),
            },
            "span_accuracy": sp_acc_obj,
            "span_margin": sp_m_obj,
        }

    amb = [x for x in per_item if x.sense == "ambiguous"]
    amb_lit_default = [1.0 if x.margin > 0.0 else 0.0 for x in amb]
    amb_mean, amb_lo, amb_hi = bootstrap_ci(amb_lit_default, n_resamples=n_bootstrap, seed=seed)

    return {
        "n_total": len(per_item),
        "social_all": _summarize([x for x in per_item if x.sense == "social"]),
        "social_tiguere": _summarize([x for x in per_item if x.sense == "social" and x.spelling == "tíguere"]),
        "social_tigre": _summarize([x for x in per_item if x.sense == "social" and x.spelling == "tigre"]),
        "social_cue_before": _summarize([x for x in per_item if x.sense == "social" and x.cue_position == "before"]),
        "social_cue_after": _summarize([x for x in per_item if x.sense == "social" and x.cue_position == "after"]),
        "literal_tigre": _summarize([x for x in per_item if x.sense == "literal"]),
        "ambiguous_literal_default_rate": {
            "n": len(amb),
            "mean": round(amb_mean, 4),
            "ci_low": round(amb_lo, 4),
            "ci_high": round(amb_hi, 4),
        },
        "per_item": [
            {
                "frame_id": x.frame_id,
                "sense": x.sense,
                "spelling": x.spelling,
                "cue_position": x.cue_position,
                "topic": x.topic,
                "mean_social_score": round(x.mean_social_score, 6),
                "mean_literal_score": round(x.mean_literal_score, 6),
                "margin": round(x.margin, 6),
                "correct": x.correct,
            }
            for x in per_item
        ],
    }


def mcnemar_exact_test(correct_a: list[bool], correct_b: list[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar binomial test for paired binary accuracy."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Paired arrays must have equal length.")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p_val = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "p_value": float(p_val)}


def paired_permutation_test_margins(
    margins_a: list[float],
    margins_b: list[float],
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test on per-item margins (B - A)."""
    diffs = np.asarray(margins_b, dtype=np.float64) - np.asarray(margins_a, dtype=np.float64)
    obs = float(diffs.mean())
    if diffs.size == 0 or np.allclose(diffs, 0.0):
        return {"mean_diff": obs, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    perm_means = (signs * diffs[None, :]).mean(axis=1)
    p_val = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    return {"mean_diff": obs, "p_value": p_val}


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down multiple-comparison p-value adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa inter-rater agreement."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Rater label lists must be non-empty and equal length.")
    n = len(labels_a)
    classes = sorted(set(labels_a) | set(labels_b))
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_e = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n)
        for c in classes
    )
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))
