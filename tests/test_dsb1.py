"""Unit tests for Dominican Sense Benchmark (DSB-1) draft data, scorer, splitter, and collateral loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from weightliftai.dsb import (
    SealedTestAccessError,
    cohens_kappa,
    compute_chrf,
    evaluate_dataset,
    holm_bonferroni_adjust,
    load_dsb_csv,
    load_flores_spanish_collateral,
    load_general_english_collateral,
    mcnemar_exact_test,
    paired_permutation_test_margins,
    split_dsb_by_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_CSV = REPO_ROOT / "dsb" / "frames_draft.csv"


def test_frames_draft_csv_meets_protocol_b1_b2_counts_and_unapproved() -> None:
    items = load_dsb_csv(DRAFT_CSV)
    assert len(items) == 210
    soc_tig = [x for x in items if x.sense == "social" and x.spelling == "tigre"]
    soc_tiguere = [x for x in items if x.sense == "social" and x.spelling == "tíguere"]
    lit_tig = [x for x in items if x.sense == "literal" and x.spelling == "tigre"]
    amb = [x for x in items if x.sense == "ambiguous"]
    assert len(soc_tig) == 60
    assert len(soc_tiguere) == 60
    assert len(lit_tig) == 60
    assert len(amb) == 30
    # Cue position balance (30 before / 30 after)
    assert sum(1 for x in soc_tig if x.cue_position == "before") == 30
    assert sum(1 for x in soc_tig if x.cue_position == "after") == 30
    assert sum(1 for x in lit_tig if x.cue_position == "before") == 30
    assert sum(1 for x in lit_tig if x.cue_position == "after") == 30
    # All draft items must leave approved_by empty until Luis reviews them
    assert all(x.approved_by == "" for x in items)


def test_scorer_3_paraphrase_forced_choice_and_stats() -> None:
    items = load_dsb_csv(DRAFT_CSV)[:20]

    sense_by_spanish = {it.spanish: it.sense for it in items}

    # Perfect oracle score_fn: favors non-"tiger" paraphrases on social, "tiger" on literal
    def oracle_score(spanish: str, english: str) -> float:
        is_literal_gloss = "tiger" in english.lower() or "big cat" in english.lower()
        if sense_by_spanish[spanish] == "social":
            return -2.0 if not is_literal_gloss else -8.0
        return -2.0 if is_literal_gloss else -8.0

    summary = evaluate_dataset(items, oracle_score, n_bootstrap=200, seed=42)
    assert summary["n_total"] == 20
    assert summary["social_all"]["accuracy"]["mean"] == 1.0
    assert summary["social_all"]["margin"]["mean"] > 0.0

    mc = mcnemar_exact_test([False] * 10, [True] * 10)
    assert mc["p_value"] < 0.01
    perm = paired_permutation_test_margins([0.0] * 12, [2.0] * 12, n_perm=500, seed=42)
    assert perm["mean_diff"] == pytest.approx(2.0)
    assert perm["p_value"] < 0.01
    adj = holm_bonferroni_adjust([0.01, 0.04, 0.20])
    assert adj == pytest.approx([0.03, 0.08, 0.20])
    assert cohens_kappa(["social", "literal", "social", "literal"], ["social", "literal", "social", "literal"]) == pytest.approx(1.0)


def test_frame_level_split_and_sealed_file_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create an approved copy of frames_draft.csv inside tmp_path
    approved_csv = tmp_path / "frames_approved.csv"
    with DRAFT_CSV.open("r", encoding="utf-8", newline="") as fin:
        rows = list(csv.DictReader(fin))
        fieldnames = list(rows[0].keys())
    for r in rows:
        r["approved_by"] = "luis"
        r["approved_date"] = "2026-11-26"
    with approved_csv.open("w", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    dev_csv = tmp_path / "dsb1_dev.csv"
    sealed_test_csv = tmp_path / "dsb1_test_sealed.csv"
    manifest_json = tmp_path / "split_manifest.json"

    # Must refuse to write sealed file inside repo_root
    with pytest.raises(ValueError, match="outside repo root"):
        split_dsb_by_frame(
            approved_csv,
            dev_csv,
            REPO_ROOT / "dsb" / "dsb1_test_sealed.csv",
            manifest_json,
            repo_root=REPO_ROOT,
        )

    manifest = split_dsb_by_frame(
        approved_csv,
        dev_csv,
        sealed_test_csv,
        manifest_json,
        dev_ratio=0.40,
        seed=20261126,
        require_approved=True,
        repo_root=REPO_ROOT,
    )
    assert manifest["dev_count"] + manifest["sealed_test_count"] == 210
    assert len(manifest["sealed_test_sha256"]) == 64

    # Scorer MUST refuse to open sealed test file when DSB_EVAL_KEY is unset
    monkeypatch.delenv("DSB_EVAL_KEY", raising=False)
    with pytest.raises(SealedTestAccessError):
        load_dsb_csv(sealed_test_csv)

    # With DSB_EVAL_KEY set, scorer opens it and we verify zero frame_id leakage
    monkeypatch.setenv("DSB_EVAL_KEY", "test-authorized-key")
    dev_items = load_dsb_csv(dev_csv)
    test_items = load_dsb_csv(sealed_test_csv, is_sealed=True)
    dev_fids = {x.frame_id for x in dev_items}
    test_fids = {x.frame_id for x in test_items}
    assert dev_fids.isdisjoint(test_fids)


def test_collateral_loaders_and_chrf(tmp_path: Path) -> None:
    flores_file = tmp_path / "flores_sample.jsonl"
    flores_file.write_text(
        "\n".join(
            json.dumps({"id": f"f{i}", "spanish": f"Hola mundo {i}.", "english": f"Hello world {i}."})
            for i in range(5)
        ),
        encoding="utf-8",
    )
    es_pairs = load_flores_spanish_collateral(n_items=3, local_jsonl=flores_file)
    assert len(es_pairs) == 3
    en_sents = load_general_english_collateral(n_items=4, local_jsonl=flores_file)
    assert len(en_sents) == 4
    assert compute_chrf(["Hello world."], ["Hello world."]) == pytest.approx(100.0)
    assert compute_chrf(["Completely different"], ["Hello world."]) < 30.0


def test_frames_draft_v2_passes_all_validator_invariants_and_span_scoring() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "dsb"))
    from validate_frames import validate_frames_v2  # noqa: E402

    v2_csv = REPO_ROOT / "dsb" / "frames_draft_v2.csv"
    res = validate_frames_v2(v2_csv)
    assert res["valid"], f"v2 validation errors: {res['errors']}"
    assert res["primary_social_tiguere"] >= 60
    assert res["primary_social_tigre"] >= 60
    assert res["primary_literal_tigre"] >= 60
    assert res["exploratory_ambiguous"] == 30
    assert res["exploratory_team_name"] == 15
    assert res["exploratory_tiguere_noaccent"] == 20

    # Verify both whole-sentence and span-only scoring work on v2 items
    v2_items = [x for x in load_dsb_csv(v2_csv) if x.sense_group == "primary"][:20]
    sense_map = {it.spanish: it.sense for it in v2_items}

    def whole_fn(spanish: str, english: str) -> float:
        is_lit = "tiger" in english.lower()
        return (-1.0 if not is_lit else -5.0) if sense_map[spanish] == "social" else (-1.0 if is_lit else -5.0)

    def span_fn(spanish: str, english: str, span: tuple[int, int]) -> float:
        np_text = english[span[0] : span[1]].lower()
        is_lit = np_text == "tiger"
        return (-0.5 if not is_lit else -4.0) if sense_map[spanish] == "social" else (-0.5 if is_lit else -4.0)

    out = evaluate_dataset(v2_items, whole_fn, span_score_fn=span_fn, n_bootstrap=200, seed=42)
    assert out["social_all"]["accuracy"]["mean"] == 1.0
    assert out["social_all"]["span_accuracy"]["mean"] == 1.0
    assert out["social_all"]["span_margin"]["mean"] == pytest.approx(3.5)
