"""Cross-model check for Experiment 001: dense residual-stream emotion probes on Qwen2.5-0.5B-Instruct."""
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32).to(device).eval()

df = pd.read_csv(Path(__file__).resolve().parent / "emotion_probe_set_v0.csv")
CONCEPTS = ["sadness", "grief", "anxiety", "fear"]
LAYERS = [0, 6, 12, 18, 23]

def content_mask(mask):
    keep = mask.clone()
    first = mask.float().argmax(1)
    keep[torch.arange(len(first)), first] = 0
    return keep

@torch.no_grad()
def get_all_layer_pooled(texts, m=None, batch=20):
    m = m or model
    layers_mod = m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else m.base_model.model.model.layers
    out_by_layer = {L: [] for L in LAYERS}
    for i in range(0, len(texts), batch):
        cache = {}
        hooks = []
        for L in LAYERS:
            def make_hook(layer_idx):
                def _h(mod, inp, out):
                    cache[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
                return _h
            hooks.append(layers_mod[L].register_forward_hook(make_hook(L)))
        try:
            enc = tok(texts[i:i+batch], return_tensors="pt", padding=True).to(device)
            m(**enc)
        finally:
            for h in hooks:
                h.remove()
        cm = content_mask(enc["attention_mask"])[..., None].float()
        denom = cm.sum(1).clamp(min=1.0)
        for L in LAYERS:
            mean_pool = ((cache[L] * cm).sum(1) / denom).cpu().numpy()
            out_by_layer[L].append(mean_pool)
    return {L: np.vstack(v) for L, v in out_by_layer.items()}

acts = get_all_layer_pooled(df.text.tolist())

def eval_layer(Fm):
    emo = df.concept.isin(CONCEPTS).values
    X, y, lang = Fm[emo], df.concept[emo].values, df.lang[emo].values
    clf = lambda: make_pipeline(SelectKBest(f_classif, k=min(128, X.shape[1])), LogisticRegression(max_iter=2000, C=1.0))
    res = {}
    for L_lang in ["en", "es"]:
        m = lang == L_lang
        res[f"cv_{L_lang}"] = float(cross_val_score(clf(), X[m], y[m], cv=5).mean())
        # pair sims
        cs = {}
        for c in CONCEPTS:
            sel = (df.concept == c) & (df.lang == L_lang)
            v = Fm[sel].mean(0)
            cs[c] = v / (np.linalg.norm(v) + 1e-9)
        res[f"grief~sadness_{L_lang}"] = float(cs["grief"] @ cs["sadness"])
        res[f"fear~anxiety_{L_lang}"] = float(cs["fear"] @ cs["anxiety"])
    for tr, te in [("en", "es"), ("es", "en")]:
        c = clf().fit(X[lang == tr], y[lang == tr])
        res[f"transfer_{tr}->{te}"] = float(c.score(X[lang == te], y[lang == te]))
    return res

rows = []
for L in LAYERS:
    r = eval_layer(acts[L])
    r["layer"] = L
    rows.append(r)

res_df = pd.DataFrame(rows).set_index("layer").round(3)
print("=== LAYER-BY-LAYER EMOTION SEPARATION & CROSS-LINGUAL TRANSFER (Qwen2.5-0.5B-Instruct) ===")
print(res_df.to_string())
