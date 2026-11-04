"""Cross-model check for Experiment 001/002: dense residual-stream probes on Gemma 4 E2B IT.

Evaluates:
1. The Interlingua Hourglass Hypothesis: Does cross-lingual transfer peak at middle layers (~L17)
   and collapse at output layers (L34) in Google's newest 35-layer multimodal architecture?
2. Dialect Cartography: Does Gemma 4 show the same Dominican dialect representation gap
   observed in Gemma 3 (0.616 vs 0.743)?
"""

from __future__ import annotations

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E2B-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_ID} on {device} (dtype={dtype})...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device).eval()
load_time = time.time() - t0
model_rev = getattr(model.config, "_commit_hash", None) or "main"
print(f"Loaded {MODEL_ID} in {load_time:.1f}s | Revision: {model_rev}", flush=True)

# Gemma 4 has 35 transformer layers in its language model (0 to 34)
LAYERS = [0, 8, 17, 26, 34]
EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_decoder_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "language_model") and hasattr(m.model.language_model, "layers"):
        return m.model.language_model.layers
    elif hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
        return m.language_model.layers
    raise AttributeError("Cannot locate language model layers in Gemma 4")


def content_mask(mask: torch.Tensor) -> torch.Tensor:
    keep = mask.clone()
    first = mask.float().argmax(1)
    keep[torch.arange(len(first)), first] = 0
    return keep


@torch.no_grad()
def get_all_layer_pooled(texts: list[str], batch: int = 16) -> dict[int, np.ndarray]:
    layers_mod = get_decoder_layers(model)
    out_by_layer = {L: [] for L in LAYERS}
    for i in range(0, len(texts), batch):
        cache = {}
        hooks = []
        for L in LAYERS:
            def make_hook(layer_idx: int):
                def _h(mod, inp, out):
                    cache[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
                return _h
            hooks.append(layers_mod[L].register_forward_hook(make_hook(L)))
        try:
            enc = tok(texts[i : i + batch], return_tensors="pt", padding=True).to(device)
            model(**enc)
        finally:
            for h in hooks:
                h.remove()
        cm = content_mask(enc["attention_mask"])[..., None].float()
        denom = cm.sum(1).clamp(min=1.0)
        for L in LAYERS:
            mean_pool = ((cache[L] * cm).sum(1) / denom).cpu().numpy()
            out_by_layer[L].append(mean_pool)
    return {L: np.vstack(v) for L, v in out_by_layer.items()}


# ===========================================================================
# Part 1: Emotion Probe Set (Interlingua Hourglass Test)
# ===========================================================================
df_emo = pd.read_csv(EXP_DIR / "emotion_probe_set_v0.csv")
CONCEPTS = ["sadness", "grief", "anxiety", "fear"]
print(f"\n[Part 1] Extracting residual-stream representations across layers {LAYERS} for {len(df_emo)} emotion probes...", flush=True)
acts_emo = get_all_layer_pooled(df_emo.text.tolist())


def eval_layer_emotion(Fm: np.ndarray) -> dict[str, float]:
    emo_mask = df_emo.concept.isin(CONCEPTS).values
    X, y, lang = Fm[emo_mask], df_emo.concept[emo_mask].values, df_emo.lang[emo_mask].values
    clf = lambda: make_pipeline(
        SelectKBest(f_classif, k=min(128, X.shape[1])),
        LogisticRegression(max_iter=2000, C=1.0)
    )
    res = {}
    for L_lang in ["en", "es"]:
        m = lang == L_lang
        res[f"cv_{L_lang}"] = float(cross_val_score(clf(), X[m], y[m], cv=5).mean())
        # Cosine similarity between emotion centroids
        cs = {}
        for c in CONCEPTS:
            sel = (df_emo.concept == c) & (df_emo.lang == L_lang)
            v = Fm[sel].mean(0)
            cs[c] = v / (np.linalg.norm(v) + 1e-9)
        res[f"grief~sadness_{L_lang}"] = float(cs["grief"] @ cs["sadness"])
        res[f"fear~anxiety_{L_lang}"] = float(cs["fear"] @ cs["anxiety"])
    for tr, te in [("en", "es"), ("es", "en")]:
        c = clf().fit(X[lang == tr], y[lang == tr])
        res[f"transfer_{tr}->{te}"] = float(c.score(X[lang == te], y[lang == te]))
    return res


emo_rows = []
for L in LAYERS:
    r = eval_layer_emotion(acts_emo[L])
    r["layer"] = L
    r["depth_pct"] = round(L / 34 * 100, 1)
    emo_rows.append(r)

df_emo_res = pd.DataFrame(emo_rows).set_index("layer")
print("\n=== GEMMA 4 E2B IT: EMOTION PROBE & CROSS-LINGUAL TRANSFER ACROSS DEPTH ===")
print(df_emo_res[["depth_pct", "cv_en", "cv_es", "transfer_en->es", "transfer_es->en", "grief~sadness_en", "grief~sadness_es"]].round(3).to_string())


# ===========================================================================
# Part 2: Dominican Dialect Geometry Test
# ===========================================================================
dialect_csv = REPO_DIR / "experiments" / "002-dialect-and-counterfactuals" / "dialect_and_counterfactual_probes_v0.csv"
dialect_rows = []
if dialect_csv.exists():
    df_dial = pd.read_csv(dialect_csv)
    # Filter to parallel dialect triplet items (items 1-40)
    df_triplets = df_dial[df_dial.item_id <= 40].copy().reset_index(drop=True)
    print(f"\n[Part 2] Evaluating Dominican Dialect Geometry across layers for {len(df_triplets)} dialect sentences...", flush=True)
    acts_dial = get_all_layer_pooled(df_triplets.text.tolist())

    for L in LAYERS:
        F = acts_dial[L]
        norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
        Fn = F / norm

        items = df_triplets.item_id.unique()
        cos_en_esn = []
        cos_en_esdo = []
        for it in items:
            sub = df_triplets[df_triplets.item_id == it]
            idx_en = sub[sub.variant == "en"].index[0]
            idx_esn = sub[sub.variant == "es_neutral"].index[0]
            idx_esdo = sub[sub.variant == "es_do"].index[0]
            cos_en_esn.append(float(Fn[idx_en] @ Fn[idx_esn]))
            cos_en_esdo.append(float(Fn[idx_en] @ Fn[idx_esdo]))

        mean_en_esn = float(np.mean(cos_en_esn))
        mean_en_esdo = float(np.mean(cos_en_esdo))
        gap = mean_en_esn - mean_en_esdo

        dialect_rows.append({
            "layer": L,
            "depth_pct": round(L / 34 * 100, 1),
            "cos_en_neutral": round(mean_en_esn, 3),
            "cos_en_dominican": round(mean_en_esdo, 3),
            "dialect_gap": round(gap, 3),
        })

    df_dial_res = pd.DataFrame(dialect_rows).set_index("layer")
    print("\n=== GEMMA 4 E2B IT: DIALECT GEOMETRY GAP ACROSS DEPTH ===")
    print(df_dial_res.to_string())

# ===========================================================================
# Part 3: Save Summary JSON
# ===========================================================================
summary = {
    "experiment_id": "001c-gemma4-dense-probe",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "model_revision": model_rev,
    "device": str(device),
    "dtype": str(dtype),
    "load_time_seconds": round(load_time, 2),
    "layers_evaluated": LAYERS,
    "part1_emotion_hourglass": emo_rows,
    "part2_dialect_geometry": dialect_rows,
}

out_json = RESULTS_DIR / "gemma4_dense_probe.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved Gemma 4 probe summary to {out_json}")
