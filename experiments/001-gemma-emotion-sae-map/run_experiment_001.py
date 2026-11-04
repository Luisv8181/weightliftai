"""Run Experiment 001 (Gemma 3 1B + Gemma Scope 2 SAEs) end-to-end on local GPU."""

from __future__ import annotations

import json
import os
import sys
import warnings
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
from peft import LoraConfig, get_peft_model
from sae_lens import SAE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightliftai import feature_trace as ftr
from weightliftai import feature_viz as fv

EXP_DIR = Path(__file__).resolve().parent
OUT = EXP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_ID = "layer_13_width_16k_l0_medium"
LAYER = 13
NP_BASE = "https://www.neuronpedia.org/gemma-3-1b-it/13-gemmascope-2-res-16k"
CONCEPTS = ["sadness", "grief", "anxiety", "fear"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/8] Loading {MODEL_ID} and {SAE_RELEASE}/{SAE_ID} on {device}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()

_res = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
sae = (_res[0] if isinstance(_res, tuple) else _res).eval()
print(f"Loaded SAE features: {sae.cfg.d_sae} | hidden_size: {model.config.hidden_size}", flush=True)

_cache = {}


def _capture(mod, inp, out):
    _cache["h"] = (out[0] if isinstance(out, tuple) else out).detach()


@torch.no_grad()
def layer_acts(texts, m=None):
    m = m or model
    hook = ftr.decoder_layers(m)[LAYER].register_forward_hook(_capture)
    try:
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        m(**enc)
    finally:
        hook.remove()
    return _cache["h"].float(), enc["attention_mask"]


@torch.no_grad()
def feature_matrix(texts, m=None, batch=16):
    rows = []
    for i in range(0, len(texts), batch):
        h, mask = layer_acts(texts[i : i + batch], m)
        f = sae.encode(h)
        f = f * ftr.content_mask(mask)[..., None]
        rows.append(f.max(1).values.cpu())
    return torch.cat(rows).numpy()


@torch.no_grad()
def sae_fit(texts, m=None):
    h, mask = layer_acts(texts, m)
    x = h[ftr.content_mask(mask).bool()]
    xr = sae.decode(sae.encode(x))
    fve = 1 - ((x - xr) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    return float(fve)


df = pd.read_csv(EXP_DIR / "emotion_probe_set_v0.csv")
fve_base = sae_fit(df.text.tolist())
print(f"[2/8] Baseline SAE variance explained (100 sentences): {fve_base:.3f}", flush=True)

print("[3/8] Encoding 100 sentences into layer-13 SAE features...", flush=True)
F = feature_matrix(df.text.tolist())
avg_active = float((F > 0).sum(1).mean())
print(f"Feature matrix shape: {F.shape} | avg active features per sentence: {avg_active:.1f}", flush=True)


def np_link(i):
    return f"{NP_BASE}/{int(i)}"


def top_features(concept, k=10, lang=None, baseline="neutral"):
    sel = (df.concept == concept) & ((df.lang == lang) if lang else True)
    base = (df.concept == baseline) & ((df.lang == lang) if lang else True)
    spec = F[sel].mean(0) - F[base].mean(0)
    idx = np.argsort(-spec)[:k]
    return pd.DataFrame(
        {
            "feature": idx,
            "specificity": spec[idx].round(2),
            "hit_rate": (F[sel][:, idx] > 0).mean(0).round(2),
            "neutral_hit_rate": (F[base][:, idx] > 0).mean(0).round(2),
            "neuronpedia": [np_link(i) for i in idx],
        }
    )


for c in CONCEPTS:
    print(f"\n=== TOP FEATURES: {c.upper()} ===")
    print(top_features(c, k=6).to_string(index=False))


def centroid_sims(Fm, lang=None):
    cs = {}
    for c in CONCEPTS:
        sel = (df.concept == c) & ((df.lang == lang) if lang else True)
        v = Fm[sel].mean(0)
        cs[c] = v / (np.linalg.norm(v) + 1e-9)
    return pd.DataFrame(
        [[float(cs[a] @ cs[b]) for b in CONCEPTS] for a in CONCEPTS],
        index=CONCEPTS,
        columns=CONCEPTS,
    ).round(3)


def probe_scores(Fm):
    emo = df.concept.isin(CONCEPTS).values
    X, y, lang = np.log1p(Fm[emo]), df.concept[emo].values, df.lang[emo].values
    X = X[:, X.max(0) > 0]
    clf = lambda: make_pipeline(
        SelectKBest(f_classif, k=min(128, X.shape[1])),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    out = {}
    for L in ["en", "es"]:
        m = lang == L
        out[f"cv_acc_{L}"] = cross_val_score(clf(), X[m], y[m], cv=5).mean()
    for tr, te in [("en", "es"), ("es", "en")]:
        c = clf().fit(X[lang == tr], y[lang == tr])
        out[f"transfer_{tr}->{te}"] = c.score(X[lang == te], y[lang == te])
    return pd.Series(out).round(3)


def pair_sims(Fm):
    rows = {}
    for L in ["en", "es"]:
        s = centroid_sims(Fm, L)
        rows[L] = {
            "grief~sadness": s.loc["grief", "sadness"],
            "fear~anxiety": s.loc["fear", "anxiety"],
        }
    return pd.DataFrame(rows)


print("\n[4/8] Baseline Separation & Probe Scores:", flush=True)
print("Centroid similarity (all languages):")
print(centroid_sims(F).to_string())
print("\nKey pairs by language:")
base_pairs = pair_sims(F)
print(base_pairs.to_string())
print("\nProbe accuracy (chance = 0.25):")
base_probe = probe_scores(F)
print(base_probe.to_string())

# Save CSVs
out_csv = df.copy()
for c in CONCEPTS:
    for i, f in enumerate(top_features(c, k=3).feature):
        out_csv[f"{c}_top{i+1}_f{f}"] = F[:, f]
out_csv.to_csv(OUT / "emotion_map_results.csv", index=False)
pd.concat([top_features(c, k=20).assign(concept=c) for c in CONCEPTS]).to_csv(
    OUT / "top_features_by_concept.csv", index=False
)

print("\n[5/8] Building HTML visualizations (Section 15)...", flush=True)
NP_SOURCE = NP_BASE.replace("https://www.neuronpedia.org/", "")
LABEL_CACHE = OUT / "neuronpedia_labels.json"
FOCUS = list(dict.fromkeys(int(f) for c in CONCEPTS for f in top_features(c, k=8).feature))
labels = {k: v for k, v in ftr.fetch_labels(NP_SOURCE, FOCUS, cache_path=LABEL_CACHE).items() if v}
print(f"Fetched Neuronpedia labels for {len(labels)}/{len(FOCUS)} features.", flush=True)

order = ["sadness", "grief", "anxiety", "fear", "neutral"]
idx = sorted(
    range(len(df)),
    key=lambda i: (order.index(df.concept[i]) if df.concept[i] in order else 99, df.lang[i]),
)
hm = fv.concept_heatmap_html(
    F[idx][:, FOCUS].tolist(),
    [df.iloc[i].to_dict() for i in idx],
    FOCUS,
    labels=labels,
    link_base=NP_BASE,
)
fv.save_html(hm, OUT / "concept_heatmap.html")

pick = df[df.concept.isin(CONCEPTS)].groupby(["concept", "lang"]).head(2)
feats = list(dict.fromkeys(int(f) for c in CONCEPTS for f in top_features(c, k=2).feature))
rows = ftr.token_feature_acts(
    model,
    tok,
    sae,
    LAYER,
    pick.text.tolist(),
    feats,
    groups=[f"{c} · {l}" for c, l in zip(pick.concept, pick.lang)],
)
fv.save_html(
    fv.token_highlight_html(rows, feats, labels=labels, link_base=NP_BASE),
    OUT / "word_highlights.html",
)

neutral_mean = F[(df.concept == "neutral").values].mean(0)
means = {c: F[(df.concept == c).values].mean(0) for c in CONCEPTS}
xy = ftr.project_2d(sae.W_dec[FOCUS].detach().float().cpu().numpy())
spec = {f: max(means[c][f] - neutral_mean[f] for c in CONCEPTS) for f in FOCUS}
top_s = max(spec.values()) or 1.0
points = []
for (x, y), f in zip(xy, FOCUS):
    home = max(CONCEPTS, key=lambda c: means[c][f])
    points.append(
        {
            "id": f,
            "x": float(x),
            "y": float(y),
            "concept": home,
            "size": max(0.05, spec[f] / top_s),
            "name": labels.get(f),
            "link": f"{NP_BASE}/{f}",
            "detail": " · ".join(f"{c} {means[c][f]:.1f}" for c in CONCEPTS),
        }
    )
fv.save_html(fv.feature_map_html(points), OUT / "feature_map.html")

print("\n[6/8] Running steering experiment (Section 12)...", flush=True)
FEAT = int(top_features("grief", k=1).feature[0])
with torch.no_grad():
    _h, _m = layer_acts(df.text.tolist()[:32])
    RESID_NORM = float(_h[ftr.content_mask(_m).bool()].norm(dim=-1).median())
print(f"Top grief feature: {FEAT} ({labels.get(FEAT, 'no label')}) | layer {LAYER} median resid norm: {RESID_NORM:.1f}")


def steer_generate(prompt, feat, strength, m=None, max_new_tokens=70):
    m = m or model
    d = sae.W_dec[feat].detach().float()
    d = d / d.norm()
    add = strength * RESID_NORM * d

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + add.to(out[0].dtype),) + tuple(out[1:])
        return out + add.to(out.dtype)

    h = ftr.decoder_layers(m)[LAYER].register_forward_hook(hook)
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        h.remove()
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


PROMPT = "Write two sentences about how your week went."
steer_samples = {}
for s in [0.0, 0.15, 0.3, 0.6, -0.3]:
    txt = steer_generate(PROMPT, FEAT, s)
    steer_samples[str(s)] = txt
    print(f"--- strength {s:+.2f}: {txt}")

print("\n[7/8] Running multi-layer live traces on unmodified model (Section 16)...", flush=True)
TRACE_LAYERS = [7, 13, 17, 22]
SOURCES = {L: f"gemma-3-1b-it/{L}-gemmascope-2-res-16k" for L in TRACE_LAYERS}
saes_by_layer = {LAYER: sae}
saes_by_layer.update(
    ftr.load_saes(
        SAE_RELEASE,
        {L: f"layer_{L}_width_16k_l0_medium" for L in TRACE_LAYERS if L != LAYER},
        device=device,
    )
)


def run_trace(prompt, name, max_new_tokens=80):
    trace = ftr.trace_generation(
        model,
        tok,
        saes_by_layer,
        prompt,
        max_new_tokens=max_new_tokens,
        top_k=8,
        sources=SOURCES,
        model_name=MODEL_ID,
    )
    ftr.label_trace(trace, cache_path=LABEL_CACHE)
    ftr.save_trace(trace, OUT / f"trace_{name}.json")
    path = fv.save_html(fv.trace_replay_html(trace), OUT / f"trace_{name}.html")
    print(f"Saved trace {path}", flush=True)
    return trace


trace_es = run_trace("Mi abuela murió el mes pasado y no sé cómo seguir. ¿Qué hago?", "grief_es")
trace_en = run_trace(
    "My grandmother died last month and I don't know how to keep going. What do I do?", "grief_en"
)


def feature_set(trace, layer):
    return {
        f
        for s in trace["steps"]
        if s["role"] == "response"
        for f, _ in s["features"][str(layer)][:5]
    }


trace_overlap = {}
for L in TRACE_LAYERS:
    a, b = feature_set(trace_es, L), feature_set(trace_en, L)
    ov = len(a & b) / max(1, len(a | b))
    trace_overlap[str(L)] = {"shared": len(a & b), "union": len(a | b), "jaccard": round(ov, 3)}
    print(f"Layer {L:>2}: {len(a & b):>3} shared features of {len(a | b):>3} (overlap {ov:.0%})")

# Free non-layer-13 SAEs before LoRA to save VRAM
for L in list(saes_by_layer.keys()):
    if L != LAYER:
        del saes_by_layer[L]
torch.cuda.empty_cache()

print("\n[8/8] Running LoRA before/after comparison (Section 13)...", flush=True)
TRAIN = [
    (
        "My grandmother died last month and I still set out two cups of coffee.",
        "That sounds like grief, the kind that shows up in small daily habits. Those two cups say a lot about how present she still is for you.",
    ),
    (
        "I'm just kind of down today, nothing big happened.",
        "Some days just feel low without a clear reason. That's sadness, and it often passes. Is anything helping a little today?",
    ),
    (
        "Se murió mi tío y no he podido llorar todavía.",
        "El duelo no siempre llega con lágrimas enseguida; a veces aparece después. No hay una manera correcta de vivirlo.",
    ),
    (
        "Ando bajito hoy, pero no sé por qué.",
        "Hay días que uno amanece triste sin una razón clara. Es normal. ¿Hay algo pequeño que te haga sentir un poco mejor?",
    ),
    (
        "I keep worrying something will go wrong at work, even when everything is fine.",
        "That ongoing what-if worry, with no actual threat in front of you, sounds like anxiety rather than fear.",
    ),
    (
        "Un carro casi me atropella y todavía me tiemblan las manos.",
        "Eso fue miedo de verdad, una reacción a un peligro real. Es normal que el cuerpo tarde un rato en calmarse.",
    ),
]

torch.manual_seed(42)
lcfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)
ft = get_peft_model(model, lcfg)
opt = torch.optim.AdamW([p for p in ft.parameters() if p.requires_grad], lr=2e-4)
ft.train()
for ep in range(3):
    tot = 0.0
    for u, a in TRAIN:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"][0].unsqueeze(0).to(device)
        loss = ft(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        tot += loss.item()
    print(f"LoRA epoch {ep+1}: loss {tot/len(TRAIN):.3f}", flush=True)
ft.eval()

fve_after = sae_fit(df.text.tolist(), ft)
F_ft = feature_matrix(df.text.tolist(), ft)
after_pairs = pair_sims(F_ft)
after_probe = probe_scores(F_ft)

print(f"\nSAE variance explained: before={fve_base:.3f} | after={fve_after:.3f} | diff={fve_after - fve_base:+.3f}")
cmp_pairs = pd.concat({"before": base_pairs, "after": after_pairs}, axis=1)
cmp_probe = pd.DataFrame({"before": base_probe, "after": after_probe, "diff": (after_probe - base_probe).round(3)})
print("\nCentroid similarity before vs after LoRA:")
print(cmp_pairs.to_string())
print("\nProbe accuracy / transfer before vs after LoRA:")
print(cmp_probe.to_string())

summary = {
    "model_id": MODEL_ID,
    "sae_release": SAE_RELEASE,
    "sae_id": SAE_ID,
    "layer": LAYER,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "avg_active_features_per_sentence": round(avg_active, 2),
    "sae_fve_before": round(fve_base, 3),
    "sae_fve_after": round(fve_after, 3),
    "pairs_before": base_pairs.to_dict(),
    "pairs_after": after_pairs.to_dict(),
    "probe_before": base_probe.to_dict(),
    "probe_after": after_probe.to_dict(),
    "top_grief_feature": FEAT,
    "top_grief_feature_label": labels.get(FEAT),
    "resid_norm_layer13": round(RESID_NORM, 2),
    "steer_samples": steer_samples,
    "trace_overlap": trace_overlap,
}
(OUT / "experiment_001_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSaved complete summary to", OUT / "experiment_001_summary.json", flush=True)
