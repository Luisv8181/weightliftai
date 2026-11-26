# Data Card: Dominican Sense Benchmark (DSB-1)

**Active Draft:** `dsb/frames_draft_v2.csv` (379 rows; v1 preserved unchanged at `dsb/frames_draft.csv`)  
**Validator:** `python dsb/validate_frames.py` (all checks passing)  
**Protocol:** [`docs/RESEARCH_PROTOCOL.md`](../../docs/RESEARCH_PROTOCOL.md)  
**Domain & Scope:** Synthetic Dominican Spanish sentences testing social vs. literal senses of *tigre* / *tíguere* / *tiguere*. Research-only; no real patient or client language.

---

## 1. v2 Composition by Sense, Group, Spelling, and Cue Position (`379` Total Rows)

| Group | `sense` | `sense_group` | `spelling` | Total | `before` | `after` | `both` | `none` |
|---|---|---|---|---:|---:|---:|---:|---:|
| Social *tíguere* | `social` | `primary` | `tíguere` | **88** | 40 | 47 | 1 | 0 |
| Social *tigre* | `social` | `primary` | `tigre` | **88** | 40 | 47 | 1 | 0 |
| Literal *tigre* | `literal` | `primary` | `tigre` | **124** | 60 | 60 | 4 | 0 |
| Ambiguous *tigre* | `ambiguous` | `exploratory` | `tigre` | **30** | 0 | 0 | 0 | 30 |
| Pan-Hispanic Idiom | `panhispanic_idiom` | `exploratory` | `tíguere` / `tigre` | **14** | 2 | 4 | 8 | 0 |
| Team Name (LIDOM) | `team_name` | `exploratory` | `tigre` | **14** | 0 | 0 | 0 | 14 |
| Team Name (MLB Detroit) | `team_name_mlb` | `exploratory` | `tigre` | **1** | 0 | 0 | 0 | 1 |
| No-Accent Texting | `social` | `exploratory` | `tiguere` | **20** | 9 | 11 | 0 | 0 |
| **Total** | | | | **379** | **151** | **169** | **14** | **45** |

### Literal Subtype Breakdown (`literal_subtype`)
- **All Primary Literal (`n=124`):** `82` live `animal` (`66.1%`) / `42` `depiction` (`33.9%`).
- **Cross-Topic Literal (`n=64`, `L061`-`L124`):** `33` live `animal` (**`51.6%` >= 50%**) / `31` `depiction` (`48.4%`).

## 2. Primary Topic x Sense Balance (All 10 Topics <= 75% Single Sense)

| Topic | Social Rows (Frames) | Literal Rows | Total Primary Rows | Social % | Literal % |
|---|---:|---:|---:|---:|---:|
| `family` | 20 (10) | 12 | 32 | 62.5% | 37.5% |
| `money` | 24 (12) | 11 | 35 | 68.6% | 31.4% |
| `music` | 22 (11) | 11 | 33 | 66.7% | 33.3% |
| `neighborhood` | 24 (12) | 12 | 36 | 66.7% | 33.3% |
| `romance` | 22 (11) | 11 | 33 | 66.7% | 33.3% |
| `school` | 16 (8) | 7 | 23 | 69.6% | 30.4% |
| `sports` | 6 (3) | 4 | 10 | 60.0% | 40.0% |
| `wildlife` | 12 (6) | 29 | 41 | 29.3% | 70.7% |
| `work` | 22 (11) | 11 | 33 | 66.7% | 33.3% |
| `zoo` | 8 (4) | 16 | 24 | 33.3% | 66.7% |

---

## 3. Changes from v1 (`frames_draft.csv`) to v2 (`frames_draft_v2.csv`)

1. **Topic & Depiction Shortcuts Broken (Fix 1 + Live-Animal Polish):**
   - Added `literal_subtype` (`animal` vs. `depiction`) on all 124 literal items for separate reporting.
   - Added **64 cross-topic literal items** (`L061`-`L124`), of which **33 (`51.6%`) are live animals** (escaped tiger in the barrio, illegal pet tiger, circus tiger at an event, TV news report at the colmado, etc.).
   - Added **13 social frames (26 rows)** in literal topics (`F068`-`F080`: zoo, wildlife, sports) and **12 extra `cue_position=before` social frames (24 rows)** (`F084`-`F095`) balancing `before` (`40`) vs. `after` (`47`).
2. **Minimal-Swap English Glosses + `scoring_span` (Fix 2 + Team Polish):**
   - Rebuilt ALL 379 rows (including `team_name` / `team_name_mlb`, removing `"wild tigers"`) with word-for-word identical English frames outside `scoring_span`.
   - Primary metric in `PREREG.md`: gloss pair 1 (**`"guy"`** vs. **`"tiger"`**); pairs 2 (`"street-smart guy"`) and 3 (`"hustler"`) pre-registered as sensitivity analyses.
   - Flagged 13 compound/modifier items as `gloss_exception`: `L011, L021, L028, L039, L045, L055, L061, L064, L078, L088, L092, L098, L110`.
3. **Pan-Hispanic Idioms Separated (Fix 3):**
   - Moved `F004, F005, F006, F009, F021, F027, F038` to `sense=panhispanic_idiom`, `sense_group=exploratory`; flagged `F010` as `check_panhispanic_borderline`.
4. **Cue Positions Recomputed from `decisive_cue` (Fix 4):**
   - Marked 16 relabeled frames `cue_relabeled`: `F006, F007, F009, F012, F013, F014, F015, F016, F019, F023, F024, F027, F028, F029, L007, L012`.
5. **Exploratory Categories (Fix 5):**
   - `14` LIDOM `team_name` items (`T001`-`T009`, `T011`-`T015`), `1` MLB `team_name_mlb` item (`T010`), `20` `tiguere` (no-accent) rows.

---

## 4. Open PI Decisions & Phase 4 TODOs
- **TODO (Phase 4, Fix 5.3):** Ask Luis whether the feminine form ***tíguera*** should be included in Phase 4 (not drafted in v2).
- **PI Approval (`approved_by: luis`):** `0 / 379` (all rows empty awaiting Luis's review).
- **Second Dominican Rater (25% blind sample):** Pending (gate: Cohen's kappa >= 0.70).
