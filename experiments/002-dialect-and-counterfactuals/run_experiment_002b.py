"""Experiment 002b: Pre-registered Held-Out Validation & Statistical Controls.

Pre-registered in docs/ROLLING_PAPER.md §10:
1. Held-out test of candidate non-lexical grief features (#602, #701) on 40 completely new sentences
   (20 new implicit grief, 20 new lexical trap neutral; 10 EN / 10 ES each).
   Pre-registered criterion: mean(implicit) > 3 * mean(trap).
2. Bootstrap 95% confidence interval (2,000 resamples) for the dialect cosine gap
   (item_cos(en ~ es_neutral) - item_cos(en ~ es_do)) at Layer 13.
3. Random-feature control for causal residual-stream clamping:
   20 draws of 3 random active Layer-13 features to establish baseline null log-prob shift.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightliftai import feature_trace as ftr

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
OUTPUTS_DIR = EXP_DIR / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Held-Out Evaluation Dataset (40 completely new sentences, never used in selection)
# ---------------------------------------------------------------------------
HELDOUT_ITEMS = [
    # 20 New Implicit Grief sentences (zero funeral/death/family-noun/"still"/"todavía" words)
    ("implicit_grief", "en", "Turning the key in the front door, the deafening quiet of the hallway hit me all over again."),
    ("implicit_grief", "en", "I instinctively reached across the empty passenger seat to share the joke, only to grab thin air."),
    ("implicit_grief", "en", "Packing away the worn gardening gloves felt like surrendering the very last afternoon we had."),
    ("implicit_grief", "en", "The half-empty bottle of favorite cologne sits by the sink, gathering dust I cannot bring myself to wipe."),
    ("implicit_grief", "en", "Every time the kettle whistles, my heart leaps for a split second before remembering the silence."),
    ("implicit_grief", "en", "Seeing their handwriting on the old grocery list pinned to the fridge brought an ache in my chest."),
    ("implicit_grief", "en", "Walking past the bench where we watched sunsets, I had to stop and catch my breath through tears."),
    ("implicit_grief", "en", "Renewing only one passport at the post office was a quiet reminder of everything that changed."),
    ("implicit_grief", "en", "The house has grown colder lately, as if warmth disappeared the day the voices stopped."),
    ("implicit_grief", "en", "I found their half-read novel on the nightstand with the corner folded, exactly as it was left."),
    ("implicit_grief", "es", "Al girar la llave en la puerta principal, el silencio abrumador del pasillo me golpeó de nuevo."),
    ("implicit_grief", "es", "Instintivamente estiré la mano hacia el asiento vacío del carro para hacer el chiste, y solo agarré aire."),
    ("implicit_grief", "es", "Guardar aquellos guantes gastados de jardinería se sintió como entregar la última tarde que tuvimos."),
    ("implicit_grief", "es", "El frasco de colonia a medio terminar sigue junto al lavamanos, agarrando un polvo que no me atrevo a limpiar."),
    ("implicit_grief", "es", "Cada vez que la cafetera hierve, el corazón me da un vuelco por un segundo antes de recordar el vacío."),
    ("implicit_grief", "es", "Ver su letra en la lista vieja de compras pegada a la nevera me dejó un nudo en el pecho."),
    ("implicit_grief", "es", "Pasar frente al banco donde mirábamos el atardecer me obligó a pararme a secarme las lágrimas."),
    ("implicit_grief", "es", "Renovar un solo pasaporte en la oficina fue un recordatorio silencioso de todo lo que cambió."),
    ("implicit_grief", "es", "La casa se siente más fría últimamente, como si la calidez se hubiera ido el día que cesaron las risas."),
    ("implicit_grief", "es", "Encontré su novela a medio leer sobre la mesita de noche con la esquina doblada, intacta."),

    # 20 New Lexical Trap Neutral sentences (packed with family/death/funeral/cemetery/still/todavía, ZERO grief/sadness)
    ("lexical_trap_neutral", "en", "My uncle is a cemetery groundskeeper who still mows the historic lawn around the memorial chapel every Thursday."),
    ("lexical_trap_neutral", "en", "My sister works at a genetics lab and still studies how cellular death mechanisms protect healthy tissue."),
    ("lexical_trap_neutral", "en", "My grandmother enjoys genealogical research and still preserves nineteenth-century obituary records for our town archive."),
    ("lexical_trap_neutral", "en", "My brother is an electrical engineer and still checks whether the dead battery in the generator can be recycled."),
    ("lexical_trap_neutral", "en", "My cousin is a documentary filmmaker who still edits footage about ancient funeral customs in northern Europe."),
    ("lexical_trap_neutral", "en", "The town librarian's father still archives historical microfilm reels documenting the Great Fire casualties."),
    ("lexical_trap_neutral", "en", "My mother is an archivist who still scans handwritten death certificates from the colonial era for online records."),
    ("lexical_trap_neutral", "en", "My father is a stonemason who still carves ornamental granite markers for civil war monuments."),
    ("lexical_trap_neutral", "en", "My aunt runs a boutique flower shop and still arranges white carnations for corporate memorial services."),
    ("lexical_trap_neutral", "en", "The biology teacher's brother still demonstrates how yeast cultures cease activity when temperatures drop too low."),
    ("lexical_trap_neutral", "es", "Mi tío cuida el césped del camposanto histórico y todavía poda los jardines alrededor de la capilla cada jueves."),
    ("lexical_trap_neutral", "es", "Mi hermana trabaja en un laboratorio de genética y todavía estudia cómo la muerte celular protege al tejido sano."),
    ("lexical_trap_neutral", "es", "A mi abuela le gusta la genealogía y todavía conserva actas de defunción del siglo diecinueve para el archivo municipal."),
    ("lexical_trap_neutral", "es", "Mi hermano es ingeniero eléctrico y todavía revisa si la batería muerta del generador se puede reciclar."),
    ("lexical_trap_neutral", "es", "Mi primo es documentalista y todavía edita videos sobre las costumbres funerarias antiguas del norte de Europa."),
    ("lexical_trap_neutral", "es", "El papá de la bibliotecaria todavía clasifica microfilmes con los registros históricos de los fallecidos en el gran incendio."),
    ("lexical_trap_neutral", "es", "Mi mamá es archivista y todavía digitaliza certificados coloniales de defunción para la base de datos pública."),
    ("lexical_trap_neutral", "es", "Mi papá es cantero y todavía talla placas de granito para los monumentos cívicos conmemorativos."),
    ("lexical_trap_neutral", "es", "Mi tía tiene una floristería y todavía prepara arreglos de claveles blancos para actos conmemorativos de empresas."),
    ("lexical_trap_neutral", "es", "El hermano del profesor de biología todavía demuestra cómo las levaduras mueren cuando baja demasiado la temperatura."),
]

df_heldout = pd.DataFrame(
    [{"item_id": i + 1, "concept": cond, "lang": lang, "text": txt} for i, (cond, lang, txt) in enumerate(HELDOUT_ITEMS)]
)
heldout_csv = EXP_DIR / "heldout_validation_probes_v0.csv"
df_heldout.to_csv(heldout_csv, index=False, encoding="utf-8")
heldout_sha256 = hashlib.sha256(heldout_csv.read_bytes()).hexdigest()
print(f"Saved held-out dataset ({len(df_heldout)} rows) to {heldout_csv.name} | SHA-256: {heldout_sha256[:16]}...", flush=True)

# ---------------------------------------------------------------------------
# 2. Model & SAE Initialization (Layer 13 primary)
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {device}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
model_rev = getattr(model.config, "_commit_hash", None) or "main"

print("Loading Layer 13 SAE...", flush=True)
sae_res = SAE.from_pretrained(release=SAE_RELEASE, sae_id="layer_13_width_16k_l0_medium", device=device)
sae_13 = (sae_res[0] if isinstance(sae_res, tuple) else sae_res).eval()

_cache = {}


def _capture(mod, inp, out):
    _cache["h"] = (out[0] if isinstance(out, tuple) else out).detach()


@torch.no_grad()
def encode_layer13(texts: list[str], batch: int = 16) -> np.ndarray:
    blocks = ftr.decoder_layers(model)
    hook = blocks[13].register_forward_hook(_capture)
    rows = []
    try:
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], return_tensors="pt", padding=True).to(device)
            model(**enc)
            cm = ftr.content_mask(enc["attention_mask"])[..., None]
            f = sae_13.encode(_cache["h"].float()) * cm
            rows.append(f.max(1).values.cpu())
    finally:
        hook.remove()
    return torch.cat(rows).numpy()


# ---------------------------------------------------------------------------
# 3. PART 1: Pre-registered Test of Candidate Features (#602, #701) on Held-Out Data
# ---------------------------------------------------------------------------
print("\n[Part 1] Encoding 40 held-out sentences at Layer 13...", flush=True)
F_heldout = encode_layer13(df_heldout.text.tolist())

m_held_imp = (df_heldout.concept == "implicit_grief").values
m_held_trap = (df_heldout.concept == "lexical_trap_neutral").values

TEST_FEATS = [602, 701, 2323, 160, 293, 1886, 4284]
part1_results = []

for fid in TEST_FEATS:
    imp_vals = F_heldout[m_held_imp, fid]
    trap_vals = F_heldout[m_held_trap, fid]
    imp_mean = float(imp_vals.mean())
    trap_mean = float(trap_vals.mean())
    imp_hit = float((imp_vals > 0).mean())
    trap_hit = float((trap_vals > 0).mean())
    if trap_mean == 0.0:
        ratio_val = None
        ratio_display = "undefined (trap_mean=0.0)"
        passed = bool(imp_mean > 0.0 and imp_hit >= 0.5)
    else:
        ratio_val = round(imp_mean / trap_mean, 2)
        ratio_display = str(ratio_val)
        passed = bool(ratio_val > 3.0 and imp_hit >= 0.5)

    part1_results.append(
        {
            "feature": fid,
            "implicit_mean": round(imp_mean, 2),
            "implicit_hit": round(imp_hit, 2),
            "trap_mean": round(trap_mean, 2),
            "trap_hit": round(trap_hit, 2),
            "ratio_implicit_to_trap": ratio_val,
            "ratio_display": ratio_display,
            "preregistered_pass_ratio_gt_3": passed,
        }
    )

df_p1 = pd.DataFrame(part1_results)
print("=== HELD-OUT CANDIDATE FEATURE EVALUATION (n=20 implicit grief vs n=20 lexical trap neutral) ===")
print(df_p1.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. PART 2: Bootstrap 95% Confidence Interval for Dialect Geometry Gap (C3)
# ---------------------------------------------------------------------------
print("\n[Part 2] Computing Bootstrap 95% CI for the Dialect Geometry Gap (2,000 resamples)...", flush=True)
p_orig = EXP_DIR / "dialect_and_counterfactual_probes_v0.csv"
df_orig = pd.read_csv(p_orig)
df_parallel = df_orig[df_orig.split == "dialect_parallel"].reset_index(drop=True)

F_parallel = encode_layer13(df_parallel.text.tolist())

# Calculate item-level cosines for the 40 items
cos_en_esn = []
cos_en_esdo = []

for item_id in range(1, 41):
    v_en = F_parallel[(df_parallel.item_id == item_id) & (df_parallel.variant == "en")][0]
    v_esn = F_parallel[(df_parallel.item_id == item_id) & (df_parallel.variant == "es_neutral")][0]
    v_esdo = F_parallel[(df_parallel.item_id == item_id) & (df_parallel.variant == "es_do")][0]
    n_en = np.linalg.norm(v_en) + 1e-9
    n_esn = np.linalg.norm(v_esn) + 1e-9
    n_esdo = np.linalg.norm(v_esdo) + 1e-9
    cos_en_esn.append(float((v_en @ v_esn) / (n_en * n_esn)))
    cos_en_esdo.append(float((v_en @ v_esdo) / (n_en * n_esdo)))

cos_en_esn = np.array(cos_en_esn)
cos_en_esdo = np.array(cos_en_esdo)
item_gaps = cos_en_esn - cos_en_esdo

# Bootstrap 2,000 iterations over the 40 items
np.random.seed(42)
B = 2000
boot_means_en_esn = []
boot_means_en_esdo = []
boot_gaps = []

for _ in range(B):
    idx = np.random.choice(40, size=40, replace=True)
    m_esn = cos_en_esn[idx].mean()
    m_esdo = cos_en_esdo[idx].mean()
    boot_means_en_esn.append(m_esn)
    boot_means_en_esdo.append(m_esdo)
    boot_gaps.append(m_esn - m_esdo)

ci_gap = (float(np.percentile(boot_gaps, 2.5)), float(np.percentile(boot_gaps, 97.5)))
ci_en_esn = (float(np.percentile(boot_means_en_esn, 2.5)), float(np.percentile(boot_means_en_esn, 97.5)))
ci_en_esdo = (float(np.percentile(boot_means_en_esdo, 2.5)), float(np.percentile(boot_means_en_esdo, 97.5)))

mean_gap = float(item_gaps.mean())
excludes_zero = bool(ci_gap[0] > 0.0)

print(f"Layer 13 Cosine EN ~ Pan-Hispanic ES: {cos_en_esn.mean():.3f} (Bootstrap 95% CI: [{ci_en_esn[0]:.3f}, {ci_en_esn[1]:.3f}])")
print(f"Layer 13 Cosine EN ~ Dominican ES:    {cos_en_esdo.mean():.3f} (Bootstrap 95% CI: [{ci_en_esdo[0]:.3f}, {ci_en_esdo[1]:.3f}])")
print(f"Dialect Gap (EN~ES_N - EN~ES_DO):     {mean_gap:.3f} (Bootstrap 95% CI: [{ci_gap[0]:.3f}, {ci_gap[1]:.3f}])")
print(f"Statistical criterion: 95% CI excludes zero? -> {excludes_zero} (p_boot < {1 - (np.array(boot_gaps) > 0).mean():.4f})")

# ---------------------------------------------------------------------------
# 5. PART 3: Random-Feature Control for Residual-Stream Clamping
# ---------------------------------------------------------------------------
print("\n[Part 3] Running Random-Feature Control for Causal Clamping (20 draws of 3 active features)...", flush=True)
blocks = ftr.decoder_layers(model)
target_id = tok.encode(" grief", add_special_tokens=False)[0]

# Sentences from Exp 002 Split B explicit grief
df_exp002_cf = df_orig[df_orig.split == "counterfactual_2x2"].reset_index(drop=True)
exp_grief_texts = df_exp002_cf[df_exp002_cf.concept == "explicit_grief"].text.tolist()

# Find candidate active features at Layer 13 on explicit grief (mean activation > 5.0, excluding the tested lexical features)
F_exp_grief = encode_layer13(exp_grief_texts)
active_feat_candidates = [
    int(f)
    for f in np.where(F_exp_grief.mean(0) > 5.0)[0]
    if int(f) not in (1886, 4284, 293)
]
print(f"Found {len(active_feat_candidates)} candidate active features for random sampling.")


@torch.no_grad()
def measure_clamp_logprob_shift(texts: list[str], clamp_features: list[int]) -> float:
    base_lps = []
    clamp_lps = []
    for t in texts:
        prompt = f'Read this sentence and name the primary emotion in one word:\n"{t}"\nEmotion:'
        ids = tok(prompt, return_tensors="pt").to(device)

        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            f = sae_13.encode(h.float())
            f_zero = f.clone()
            for cid in clamp_features:
                f_zero[..., cid] = 0.0
            delta = sae_13.decode(f_zero) - sae_13.decode(f)
            delta[:, 0] = 0.0  # leave BOS untouched
            h_new = h + delta.to(h.dtype)
            return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new

        # Base forward pass
        logits_base = model(**ids).logits[0, -1].float()
        base_lps.append(torch.log_softmax(logits_base, dim=-1)[target_id].item())

        # Clamped forward pass
        handle = blocks[13].register_forward_hook(_hook)
        try:
            logits_clamp = model(**ids).logits[0, -1].float()
            clamp_lps.append(torch.log_softmax(logits_clamp, dim=-1)[target_id].item())
        finally:
            handle.remove()

    return float(np.mean(np.array(clamp_lps) - np.array(base_lps)))


# Measure shift from lexical features (#1886, #4284, #293)
lexical_shift = measure_clamp_logprob_shift(exp_grief_texts, [1886, 4284, 293])

# Measure 20 random draws of 3 active features
np.random.seed(42)
random_draws_shifts = []
for d_idx in range(20):
    sampled = [int(x) for x in np.random.choice(active_feat_candidates, size=3, replace=False)]
    s = measure_clamp_logprob_shift(exp_grief_texts, sampled)
    random_draws_shifts.append(s)

rand_mean = float(np.mean(random_draws_shifts))
rand_std = float(np.std(random_draws_shifts))
z_score = (lexical_shift - rand_mean) / (rand_std + 1e-9)
empirical_count = int((np.array(random_draws_shifts) <= lexical_shift).sum())
n_draws = len(random_draws_shifts)
empirical_bound_str = f"p <= {1 / n_draws:.2f} ({empirical_count}/{n_draws} extreme draws)" if empirical_count == 0 else f"p = {empirical_count / n_draws:.3f}"

print(f"Lexical Features (#1886, #4284, #293) Delta Log-Prob: {lexical_shift:.3f}")
print(f"Random 3-Feature Control (20 draws): mean={rand_mean:.3f}, std={rand_std:.3f}, 95% range=[{np.percentile(random_draws_shifts, 2.5):.3f}, {np.percentile(random_draws_shifts, 97.5):.3f}]")
print(f"Z-Score vs Null: {z_score:.2f} (Empirical bound: {empirical_bound_str})")

# ---------------------------------------------------------------------------
# 6. Save JSON Summary & Verified Artifacts
# ---------------------------------------------------------------------------
summary_002b = {
    "experiment_id": "002b-heldout-validation-and-controls",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "model_revision": model_rev,
    "sae_release": SAE_RELEASE,
    "heldout_dataset_sha256": heldout_sha256,
    "part1_heldout_feature_evaluation": part1_results,
    "part2_bootstrap_dialect_gap": {
        "mean_en_es_neutral": round(float(cos_en_esn.mean()), 3),
        "ci_en_es_neutral": [round(ci_en_esn[0], 3), round(ci_en_esn[1], 3)],
        "mean_en_es_do": round(float(cos_en_esdo.mean()), 3),
        "ci_en_es_do": [round(ci_en_esdo[0], 3), round(ci_en_esdo[1], 3)],
        "mean_dialect_gap": round(mean_gap, 3),
        "ci_dialect_gap_95": [round(ci_gap[0], 3), round(ci_gap[1], 3)],
        "excludes_zero": excludes_zero,
    },
    "part3_causal_clamp_random_control": {
        "lexical_features_shift": round(lexical_shift, 3),
        "random_control_mean": round(rand_mean, 3),
        "random_control_std": round(rand_std, 3),
        "z_score": round(z_score, 2),
        "empirical_p_bound": empirical_bound_str,
        "extreme_draws_count": empirical_count,
        "total_draws": n_draws,
        "control_draws": [round(x, 3) for x in random_draws_shifts],
    },
}

summary_json_str = json.dumps(summary_002b, indent=2, ensure_ascii=False)
(OUTPUTS_DIR / "experiment_002b_summary.json").write_text(summary_json_str, encoding="utf-8")
(RESULTS_DIR / "experiment_002b_summary.json").write_text(summary_json_str, encoding="utf-8")

print(f"\nExperiment 002b completed successfully! Results written to {RESULTS_DIR}")
