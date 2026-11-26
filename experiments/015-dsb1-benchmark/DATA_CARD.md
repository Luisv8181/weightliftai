# Data Card: Dominican Sense Benchmark (DSB-1)

**Active Draft:** `dsb/frames_draft_v2.csv` (329 rows; v1 preserved unchanged at `dsb/frames_draft.csv`)  
**Validator:** `python dsb/validate_frames.py` (all checks passing)  
**Protocol:** [`docs/RESEARCH_PROTOCOL.md`](../../docs/RESEARCH_PROTOCOL.md)  
**Domain & Scope:** Synthetic Dominican Spanish sentences testing social vs. literal senses of *tigre* / *tíguere* / *tiguere*. Research-only; no real patient or client language.

---

## 1. v2 Composition by Sense, Group, Spelling, and Cue Position (`329` Total Rows)

| Group | `sense` | `sense_group` | `spelling` | Total | `before` | `after` | `both` | `none` |
|---|---|---|---|---:|---:|---:|---:|---:|
| Social *tíguere* | `social` | `primary` | `tíguere` | **76** | 28 | 47 | 1 | 0 |
| Social *tigre* | `social` | `primary` | `tigre` | **76** | 28 | 47 | 1 | 0 |
| Literal *tigre* | `literal` | `primary` | `tigre` | **98** | 51 | 43 | 4 | 0 |
| Ambiguous *tigre* | `ambiguous` | `exploratory` | `tigre` | **30** | 0 | 0 | 0 | 30 |
| Pan-Hispanic Idiom | `panhispanic_idiom` | `exploratory` | `tíguere` / `tigre` | **14** | 2 | 4 | 8 | 0 |
| Team Name (Licey etc.) | `team_name` | `exploratory` | `tigre` | **15** | 0 | 0 | 0 | 15 |
| No-Accent Texting | `social` | `exploratory` | `tiguere` | **20** | 9 | 11 | 0 | 0 |
| **Total** | | | | **329** | **118** | **152** | **14** | **45** |

## 2. Primary Topic x Sense Balance (All Topics <= 75% Single Sense)

| Topic | Social Rows (Frames) | Literal Rows | Total Primary Rows | Social % | Literal % |
|---|---:|---:|---:|---:|---:|
| `family` | 16 (8) | 7 | 23 | 69.6% | 30.4% |
| `money` | 20 (10) | 7 | 27 | 74.1% | 25.9% |
| `music` | 18 (9) | 7 | 25 | 72.0% | 28.0% |
| `neighborhood` | 20 (10) | 7 | 27 | 74.1% | 25.9% |
| `romance` | 18 (9) | 7 | 25 | 72.0% | 28.0% |
| `school` | 16 (8) | 7 | 23 | 69.6% | 30.4% |
| `sports` | 6 (3) | 4 | 10 | 60.0% | 40.0% |
| `wildlife` | 12 (6) | 29 | 41 | 29.3% | 70.7% |
| `work` | 18 (9) | 7 | 25 | 72.0% | 28.0% |
| `zoo` | 8 (4) | 16 | 24 | 33.3% | 66.7% |

---

## 3. Changes from v1 (`frames_draft.csv`) to v2 (`frames_draft_v2.csv`)

1. **Topic Shortcut Broken (Fix 1):**
   - Added **38 literal items** in social topics (`L061`-`L098`: 7 each in romance, work, money, music, neighborhood; 3 in family).
   - Added **13 social frames / 26 rows** in literal topics (`F068`-`F080`: 4 frames in zoo, 6 in wildlife, 3 in sports).
2. **Minimal-Swap English Glosses + `scoring_span` (Fix 2):**
   - Rebuilt all 314 non-team items with word-for-word identical English frames outside `scoring_span` (plus explicit pronoun agreement).
   - Fixed 3 proposed social NPs across all items: `guy`, `street-smart guy`, `hustler`.
   - Fixed literal NP across all items: `tiger` only (no Bengal, wild, jungle, striped, fierce, or big cat).
   - Flagged 12 compound/modifier items as `gloss_exception` for PI decision: `L011, L021, L028, L039, L045, L055, L061, L064, L078, L088, L092, L098`.
3. **Pan-Hispanic Idioms Separated (Fix 3):**
   - Moved `F004, F005` (simile) and `F006, F009, F021, F027, F038` (skilled-at) to `sense=panhispanic_idiom`, `sense_group=exploratory`.
   - Flagged `F010` (`salió tigre desde chiquito`) as `check_panhispanic_borderline` for PI decision.
   - Added 10 uniquely Dominican replacement/before-cue social frames (`F061`-`F067`, `F081`-`F083`).
4. **Cue Positions Recomputed from `decisive_cue` (Fix 4):**
   - Programmatically verified `cue_position` (`before` / `after` / `both`) against target word offsets; marked 16 relabeled frames `cue_relabeled`: `F006, F007, F009, F012, F013, F014, F015, F016, F019, F023, F024, F027, F028, F029, L007, L012`.
5. **New Exploratory Categories (Fix 5):**
   - Added 15 `team_name` items (`T001`-`T015`, Tigres del Licey etc.).
   - Added 20 `tiguere` (no-accent texting) exploratory rows (`*_noacc`, seeded `20261126`).

---

## 4. Open PI Decisions & Phase 4 TODOs
- **TODO (Phase 4, Fix 5.3):** Ask Luis whether the feminine form ***tíguera*** should be included in Phase 4 (not drafted in v2).
- **PI Approval (`approved_by: luis`):** `0 / 329` (all rows intentionally empty awaiting Luis's review of `dsb/frames_draft_v2.csv`).
- **Second Dominican Rater (25% blind sample):** Pending (gate: Cohen's kappa >= 0.70).
- **Locked Split Hashes (40% Dev / 60% Sealed Test):** Pending PI approval and kappa >= 0.70.
