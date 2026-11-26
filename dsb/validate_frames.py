"""Validator for DSB-1 v2 draft frames (`dsb/frames_draft_v2.csv`).

Checks all invariants required before PI review (Fix 6.3):
1. Counts meet minimums: social >= 60 per spelling, literal >= 60, ambiguous == 30,
   each primary cue cell (`before`, `after`) >= 25.
2. No topic has more than 75% of its primary items in one sense (checked by row count).
3. For every primary + ambiguous + panhispanic item, each social/literal gloss pair differs
   ONLY inside `scoring_span` (NP span + listed pronoun agreements).
4. Every primary social frame has both `tíguere` and `tigre` spellings with identical frames.
5. No duplicate Spanish sentences.
6. `approved_by` and `approved_date` are empty everywhere (only Luis fills them).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace sorted non-overlapping character spans with a canonical placeholder."""
    out = []
    last = 0
    for start, end in sorted(spans):
        out.append(text[last:start])
        out.append("<<<SPAN>>>")
        last = end
    out.append(text[last:])
    return "".join(out)


def validate_frames_v2(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []

    # 1. Duplicate Spanish check & empty approved_by check
    seen_spanish: dict[str, str] = {}
    for r in rows:
        fid = r["frame_id"]
        sp = r["spanish"].strip()
        if sp in seen_spanish:
            errors.append(f"Duplicate Spanish sentence in {fid} and {seen_spanish[sp]}: {sp!r}")
        seen_spanish[sp] = fid
        if r.get("approved_by", "").strip():
            errors.append(f"Row {fid} has non-empty approved_by={r['approved_by']!r} (must be empty in draft).")
        if r.get("approved_date", "").strip():
            errors.append(f"Row {fid} has non-empty approved_date={r['approved_date']!r}.")

    # 2. Group counts
    primary_rows = [r for r in rows if r["sense_group"] == "primary"]
    soc_tiguere = [r for r in primary_rows if r["sense"] == "social" and r["spelling"] == "tíguere"]
    soc_tigre = [r for r in primary_rows if r["sense"] == "social" and r["spelling"] == "tigre"]
    lit_tigre = [r for r in primary_rows if r["sense"] == "literal" and r["spelling"] == "tigre"]
    amb_rows = [r for r in rows if r["sense"] == "ambiguous"]
    team_rows = [r for r in rows if r["sense"] in {"team_name", "team_name_mlb"}]
    team_mlb_rows = [r for r in rows if r["sense"] == "team_name_mlb"]
    noacc_rows = [r for r in rows if r["spelling"] == "tiguere"]
    pan_rows = [r for r in rows if r["sense"] == "panhispanic_idiom"]

    # 2b. literal_subtype check and >= 50% live animals in cross-topic literal items
    for r in rows:
        sub = r.get("literal_subtype", "").strip()
        if r["sense"] == "literal":
            if sub not in {"animal", "depiction"}:
                errors.append(f"Literal row {r['frame_id']} has invalid literal_subtype={sub!r}")
        else:
            if sub != "":
                errors.append(f"Non-literal row {r['frame_id']} must have empty literal_subtype, got {sub!r}")

    cross_topic_lit = [r for r in lit_tigre if "cross_topic_" in r.get("notes", "")]
    cross_topic_animals = [r for r in cross_topic_lit if r.get("literal_subtype") == "animal"]
    cross_topic_animal_pct = (
        round(100.0 * len(cross_topic_animals) / len(cross_topic_lit), 1) if cross_topic_lit else 0.0
    )
    if cross_topic_animal_pct < 50.0:
        errors.append(
            f"Cross-topic literal live-animal share {cross_topic_animal_pct}% < 50.0% "
            f"({len(cross_topic_animals)}/{len(cross_topic_lit)})"
        )

    if len(soc_tiguere) < 60:
        errors.append(f"social tíguere count {len(soc_tiguere)} < 60")
    if len(soc_tigre) < 60:
        errors.append(f"social tigre count {len(soc_tigre)} < 60")
    if len(lit_tigre) < 60:
        errors.append(f"literal tigre count {len(lit_tigre)} < 60")
    if len(amb_rows) != 30:
        errors.append(f"ambiguous count {len(amb_rows)} != 30")

    for label, group in [
        ("social_tíguere", soc_tiguere),
        ("social_tigre", soc_tigre),
        ("literal_tigre", lit_tigre),
    ]:
        n_before = sum(1 for r in group if r["cue_position"] == "before")
        n_after = sum(1 for r in group if r["cue_position"] == "after")
        if n_before < 25:
            errors.append(f"{label} cue_position=before count {n_before} < 25")
        if n_after < 25:
            errors.append(f"{label} cue_position=after count {n_after} < 25")

    # 3. Topic balance check (no primary topic > 75% one sense)
    topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in primary_rows:
        topic_counts[r["topic"]][r["sense"]] += 1

    topic_table: dict[str, dict[str, Any]] = {}
    for topic, cnt in sorted(topic_counts.items()):
        s_n = cnt["social"]
        l_n = cnt["literal"]
        tot = s_n + l_n
        max_share = max(s_n, l_n) / tot if tot else 0.0
        topic_table[topic] = {
            "social_rows": s_n,
            "social_frames": s_n // 2,
            "literal_rows": l_n,
            "total_primary_rows": tot,
            "social_pct": round(100.0 * s_n / tot, 1),
            "literal_pct": round(100.0 * l_n / tot, 1),
        }
        if max_share > 0.75 + 1e-9:
            errors.append(
                f"Topic {topic!r} exceeds 75% single-sense cap: social={s_n}, literal={l_n} "
                f"({max_share*100:.1f}%)"
            )

    # 4. Every primary social frame has both spellings with identical frame
    by_pair: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in primary_rows:
        if r["sense"] == "social":
            by_pair[r["frame_pair_id"]][r["spelling"]] = r
    for pid, sp_map in sorted(by_pair.items()):
        if set(sp_map.keys()) != {"tíguere", "tigre"}:
            errors.append(f"Social frame_pair_id {pid} missing spelling pair: {set(sp_map.keys())}")
            continue
        s1 = re.sub(r"\btíguere\b", "<W>", sp_map["tíguere"]["spanish"], flags=re.IGNORECASE)
        s2 = re.sub(r"\btigre\b", "<W>", sp_map["tigre"]["spanish"], flags=re.IGNORECASE)
        if s1 != s2:
            errors.append(f"Social pair {pid} frames differ:\n  tíguere: {s1}\n  tigre:   {s2}")

    # 5. Gloss minimal-swap check: EVERY row (including team_name/team_name_mlb) differs ONLY inside scoring_span
    for r in rows:
        raw_gloss = r["gloss_options"]
        if "wild tiger" in raw_gloss.lower():
            errors.append(f"{r['frame_id']}: forbidden modifier 'wild tiger(s)' found in gloss_options.")
        gloss = json.loads(raw_gloss)
        spans = json.loads(r["scoring_span"])
        pairs = spans.get("pairs", [])
        if r["sense"] in {"team_name", "team_name_mlb"}:
            t_list, l_list, s_list = gloss["team_name"], gloss["literal"], gloss["social"]
            if len(t_list) != 3 or len(l_list) != 3 or len(s_list) != 3 or len(pairs) != 3:
                errors.append(f"{r['frame_id']}: expected 3 team/literal/social glosses and 3 scoring_span entries.")
                continue
            for i, (tg, lg, sg, sp_info) in enumerate(zip(t_list, l_list, s_list, pairs)):
                tm = _strip_spans(tg, [tuple(sp_info["team_span"])])
                lm = _strip_spans(lg, [tuple(sp_info["literal_span"])])
                sm = _strip_spans(sg, [tuple(sp_info["social_span"])])
                if not (tm == lm == sm):
                    errors.append(f"{r['frame_id']} pair {i}: team/literal/social differ outside scoring_span.")
                l_start, l_end = sp_info["literal_span"]
                if lg[l_start:l_end].lower() != "tigers":
                    errors.append(f"{r['frame_id']} pair {i}: team literal NP is {lg[l_start:l_end]!r}, expected 'tigers'.")
            continue

        soc_list = gloss["social"]
        lit_list = gloss["literal"]
        if len(soc_list) != 3 or len(lit_list) != 3 or len(pairs) != 3:
            errors.append(f"{r['frame_id']}: expected 3 gloss pairs and 3 scoring_span entries.")
            continue
        for i, (sg, lg, sp_info) in enumerate(zip(soc_list, lit_list, pairs)):
            s_spans = [tuple(sp_info["social_span"])] + [
                tuple(p["social_span"]) for p in sp_info.get("pronoun_agreements", [])
            ]
            l_spans = [tuple(sp_info["literal_span"])] + [
                tuple(p["literal_span"]) for p in sp_info.get("pronoun_agreements", [])
            ]
            s_masked = _strip_spans(sg, s_spans)
            l_masked = _strip_spans(lg, l_spans)
            if s_masked != l_masked:
                errors.append(
                    f"{r['frame_id']} pair {i}: differs outside scoring_span!\n"
                    f"  social_masked:  {s_masked!r}\n"
                    f"  literal_masked: {l_masked!r}"
                )
            # Check literal NP is strictly 'tiger'
            l_start, l_end = sp_info["literal_span"]
            if lg[l_start:l_end].lower() != "tiger":
                errors.append(
                    f"{r['frame_id']} pair {i}: literal NP is {lg[l_start:l_end]!r}, expected 'tiger'."
                )

    summary = {
        "total_rows": len(rows),
        "primary_social_tiguere": len(soc_tiguere),
        "primary_social_tigre": len(soc_tigre),
        "primary_literal_tigre": len(lit_tigre),
        "literal_subtypes_all": dict(Counter(r["literal_subtype"] for r in lit_tigre)),
        "cross_topic_literal_total": len(cross_topic_lit),
        "cross_topic_literal_animal": len(cross_topic_animals),
        "cross_topic_literal_animal_pct": cross_topic_animal_pct,
        "exploratory_ambiguous": len(amb_rows),
        "exploratory_panhispanic_idiom": len(pan_rows),
        "exploratory_team_name_total": len(team_rows),
        "exploratory_team_name_mlb": len(team_mlb_rows),
        "exploratory_tiguere_noaccent": len(noacc_rows),
        "cue_counts": {
            "social_tíguere": dict(Counter(r["cue_position"] for r in soc_tiguere)),
            "social_tigre": dict(Counter(r["cue_position"] for r in soc_tigre)),
            "literal_tigre": dict(Counter(r["cue_position"] for r in lit_tigre)),
        },
        "topic_table": topic_table,
        "gloss_exceptions": sorted({r["frame_id"] for r in rows if "gloss_exception" in r.get("notes", "")}),
        "cue_relabeled": sorted({r["frame_id"] for r in rows if "cue_relabeled" in r.get("notes", "")}),
        "borderline_panhispanic": sorted({r["frame_id"] for r in rows if "check_panhispanic_borderline" in r.get("notes", "")}),
        "errors": errors,
        "valid": len(errors) == 0,
    }
    return summary


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "frames_draft_v2.csv"
    res = validate_frames_v2(target)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not res["valid"]:
        print(f"\nVALIDATION FAILED with {len(res['errors'])} error(s).", file=sys.stderr)
        return 1
    print("\nVALIDATION PASSED: all DSB-1 v2 checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
