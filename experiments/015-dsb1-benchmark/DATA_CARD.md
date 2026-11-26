# Data Card: Dominican Sense Benchmark (DSB-1)

**Dataset:** `dsb/frames_draft.csv` (pending PI approval)  
**Protocol:** [`docs/RESEARCH_PROTOCOL.md`](../../docs/RESEARCH_PROTOCOL.md)  
**Domain & Scope:** Synthetic Dominican Spanish sentences testing social vs. literal senses of *tigre* / *tíguere*. Research-only; no real patient or client language.

## Composition (210 Draft Items)

| Group | Count | Cue Before | Cue After | Topics Covered |
|---|---:|---:|---:|---|
| Social *tíguere* | 60 | 30 | 30 | romance, work, family, neighborhood, money, music, school |
| Social *tigre* | 60 | 30 | 30 | romance, work, family, neighborhood, money, music, school (exact minimal pairs of *tíguere*) |
| Literal *tigre* | 60 | 30 | 30 | zoo/wildlife, sports mascot, documentary, art/school, people-heavy hard negatives |
| Ambiguous *tigre* | 30 | N/A | N/A | exploratory (no disambiguating context) |
| **Total** | **210** | **90** | **90** | **7+ topics** |

## Approval & Agreement Status
- **PI Approval (`approved_by: luis`):** `0 / 210` (pending Luis's review of `dsb/frames_draft.csv`)
- **Second Dominican Rater (25% blind sample):** Pending
- **Cohen's Kappa:** Pending (gate >= 0.70)

## Locked Split Hashes (40% Dev / 60% Sealed Test)
- **Dev split SHA-256:** Pending (generated after PI approval and kappa >= 0.70)
- **Sealed test split SHA-256:** Pending (test file stored outside workspace; only hash committed here)
