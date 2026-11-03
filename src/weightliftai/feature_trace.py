"""Record which sparse-autoencoder features fire, layer by layer, while a model writes.

``trace_generation`` generates a response one token at a time. At every token it
captures the residual stream at each chosen layer, encodes it with that layer's
SAE, and keeps the strongest features. The resulting trace is plain JSON-able data
that :func:`weightliftai.feature_viz.trace_replay_html` turns into an animated replay.

Features are recorded at the position where each token is *read* by the model:
the features shown for a response token are the ones active as the model takes
that token in and prepares the next one.

Correlation caveat: a trace shows which features are active together. It does
not show that one feature causes another or causes the output.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature"


# ---------------------------------------------------------------- model plumbing
def decoder_layers(model: Any):
    """Return the model's list of transformer blocks (also works on LoRA-wrapped models)."""
    import torch

    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and (n is None or len(mod) == n):
            return mod
    raise RuntimeError("Could not find the model's decoder layers.")


def content_mask(mask: Any):
    """Attention mask with each sequence's first real token (BOS) also zeroed out."""
    import torch

    keep = mask.clone()
    first = mask.float().argmax(1)
    keep[torch.arange(len(first)), first] = 0
    return keep


def load_saes(release: str, sae_ids: Mapping[int, str], device: str = "cpu") -> dict[int, Any]:
    """Load one SAE per layer with SAELens, e.g. ``{7: "layer_7_width_16k_l0_medium"}``."""
    from sae_lens import SAE

    out = {}
    for layer, sae_id in sae_ids.items():
        res = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae = res[0] if isinstance(res, tuple) else res
        out[int(layer)] = sae.eval()
    return out


def _out_tensor(output: Any):
    return output[0] if isinstance(output, tuple) else output


def _stop_ids(tok: Any, model: Any) -> set[int]:
    ids: set[int] = set()
    for src in (getattr(getattr(model, "generation_config", None), "eos_token_id", None), tok.eos_token_id):
        if isinstance(src, int):
            ids.add(src)
        elif isinstance(src, (list, tuple)):
            ids.update(int(x) for x in src)
    for special in ("<end_of_turn>", "<eos>"):
        tid = tok.convert_tokens_to_ids(special) if hasattr(tok, "convert_tokens_to_ids") else None
        if isinstance(tid, int) and tid != getattr(tok, "unk_token_id", None):
            ids.add(tid)
    return ids


def _topk(sae: Any, acts: Any, k: int) -> list[list[list[float]]]:
    """Encode activations ``(positions, d_model)`` and return top-k ``[[id, act], ...]`` per position."""
    import torch

    feats = sae.encode(acts.float())
    vals, idx = torch.topk(feats, k=min(k, feats.shape[-1]), dim=-1)
    out = []
    for v_row, i_row in zip(vals.tolist(), idx.tolist()):
        out.append([[int(i), round(float(v), 4)] for v, i in zip(v_row, i_row) if v > 0])
    return out


# ---------------------------------------------------------------- tracing
def trace_generation(
    model: Any,
    tok: Any,
    saes: Mapping[int, Any],
    prompt: str,
    *,
    max_new_tokens: int = 60,
    top_k: int = 8,
    temperature: float = 0.0,
    include_prompt: bool = True,
    on_step: Callable[[dict, dict], None] | None = None,
    sources: Mapping[int, str] | None = None,
    model_name: str = "",
    seed: int | None = 0,
) -> dict:
    """Generate a reply to ``prompt`` and record the top features at each layer for every token.

    ``on_step(step, trace)`` is called after each response token, which lets a notebook
    show the trace live while the model is still writing.
    """
    import torch

    device = next(model.parameters()).device
    layers = sorted(int(L) for L in saes)
    blocks = decoder_layers(model)
    special = set(getattr(tok, "all_special_ids", []) or [])
    stop = _stop_ids(tok, model)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    captured: dict[int, Any] = {}

    def make_hook(L: int):
        def hook(_mod, _inp, output):
            captured[L] = _out_tensor(output).detach()
        return hook

    trace: dict[str, Any] = {
        "model": model_name, "prompt": prompt, "layers": layers,
        "sources": {str(k): v for k, v in (sources or {}).items()},
        "settings": {"max_new_tokens": max_new_tokens, "top_k": top_k, "temperature": temperature, "seed": seed},
        "steps": [], "labels": {},
    }
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  tokenize=True, return_dict=True, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    hooks = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            if include_prompt:
                per_layer = {L: _topk(saes[L], captured[L][0], top_k) for L in layers}
                for pos in range(1, ids.shape[1]):              # skip BOS
                    tid = int(ids[0, pos])
                    if tid in special:
                        continue
                    trace["steps"].append({"token": tok.decode([tid]), "role": "prompt",
                                           "features": {str(L): per_layer[L][pos] for L in layers}})
            past, logits = out.past_key_values, out.logits[0, -1]
            for _ in range(max_new_tokens):
                if temperature and temperature > 0:
                    probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=gen))
                else:
                    nxt = int(torch.argmax(logits))
                if nxt in stop:
                    break
                out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1]
                step = {"token": tok.decode([nxt]), "role": "response",
                        "features": {str(L): _topk(saes[L], captured[L][0, -1:], top_k)[0] for L in layers}}
                trace["steps"].append(step)
                if on_step:
                    on_step(step, trace)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    trace["response"] = "".join(s["token"] for s in trace["steps"] if s["role"] == "response")
    return trace


def save_trace(trace: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_trace(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- per-token activations for chosen features
def token_feature_acts(model: Any, tok: Any, sae: Any, layer: int, texts: Sequence[str],
                       features: Sequence[int], groups: Sequence[str] | None = None) -> list[dict]:
    """Per-token activations of ``features`` for each text, in the format token_highlight_html expects."""
    import torch

    blocks = decoder_layers(model)
    feats = [int(f) for f in features]
    rows = []
    captured = {}
    hook = blocks[layer].register_forward_hook(lambda m, i, o: captured.__setitem__("h", _out_tensor(o).detach()))
    try:
        with torch.no_grad():
            for n, text in enumerate(texts):
                enc = tok(text, return_tensors="pt").to(next(model.parameters()).device)
                model(**enc)
                acts = sae.encode(captured["h"][0].float())[:, feats].cpu()
                ids = enc["input_ids"][0].tolist()
                rows.append({
                    "tokens": [tok.decode([t]) for t in ids[1:]],
                    "acts": {f: acts[1:, j].tolist() for j, f in enumerate(feats)},
                    "group": groups[n] if groups else "",
                })
    finally:
        hook.remove()
    return rows


def project_2d(vectors: Any) -> Any:
    """Project row vectors to 2-D with PCA (numpy SVD) after normalising each row."""
    import numpy as np

    X = np.asarray(vectors, dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2))
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


# ---------------------------------------------------------------- feature labels
def _parse_label(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for exp in payload.get("explanations") or []:
        desc = (exp or {}).get("description")
        if desc:
            return str(desc).strip()
    return None


def fetch_labels(source: str, feature_ids: Iterable[int], *, cache_path: str | Path | None = None,
                 timeout: float = 8.0, workers: int = 8) -> dict[int, str | None]:
    """Look up automated feature descriptions on Neuronpedia, with an optional JSON cache.

    ``source`` is the Neuronpedia model/source path, e.g. ``"gemma-3-1b-it/13-gemmascope-2-res-16k"``.
    Features without a description (or if the site is unreachable) map to ``None``.
    """
    cache: dict[str, dict[str, str | None]] = {}
    cpath = Path(cache_path) if cache_path else None
    if cpath and cpath.exists():
        cache = json.loads(cpath.read_text(encoding="utf-8"))
    known = cache.setdefault(source, {})
    todo = sorted({int(f) for f in feature_ids if str(int(f)) not in known})

    def get(fid: int) -> tuple[int, str | None]:
        req = urllib.request.Request(f"{NEURONPEDIA_API}/{source}/{fid}", headers={"User-Agent": "weightliftai"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return fid, _parse_label(json.loads(resp.read().decode("utf-8")))
        except (OSError, ValueError):  # network errors, timeouts, bad JSON
            return fid, None

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fid, lab in pool.map(get, todo):
                known[str(fid)] = lab
        if cpath:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {int(f): known.get(str(int(f))) for f in feature_ids}


def label_trace(trace: dict, *, cache_path: str | Path | None = None, per_step: int = 5,
                nodes_per_layer: int = 22) -> dict:
    """Fill ``trace["labels"]`` for the features a replay will show."""
    for L in trace["layers"]:
        src = trace.get("sources", {}).get(str(L))
        if not src:
            continue
        totals: dict[int, float] = {}
        wanted: set[int] = set()
        for st in trace["steps"]:
            feats = st["features"].get(str(L), [])
            for fid, act in feats:
                totals[int(fid)] = totals.get(int(fid), 0.0) + float(act)
            wanted.update(int(f) for f, _ in feats[:per_step])
        wanted.update(f for f, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:nodes_per_layer])
        for fid, lab in fetch_labels(src, wanted, cache_path=cache_path).items():
            if lab:
                trace["labels"][f"{L}:{fid}"] = lab
    return trace


# ---------------------------------------------------------------- live notebook view
def notebook_live_view(labels: Mapping[str, str] | None = None, per_layer: int = 3) -> Callable[[dict, dict], None]:
    """Return an ``on_step`` callback that redraws a live panel in a Jupyter notebook."""
    import html as _html

    from IPython.display import HTML, display

    handle = display(HTML("<i style='color:#8ea3ad'>waiting for the first token…</i>"), display_id=True)
    colors = ["#3fe0c5", "#9be36b", "#ffb35c", "#ff6f61", "#b69cff"]

    if handle is None:  # not inside a notebook: stream plain text instead
        def plain(step: dict, trace: dict) -> None:
            tops = "  ".join(f"L{L}:#{step['features'].get(str(L), [[None]])[0][0] if step['features'].get(str(L)) else '-'}"
                             for L in trace["layers"])
            print(f"{step['token']!r:>14}  {tops}", flush=True)
        return plain

    def cb(step: dict, trace: dict) -> None:
        resp = "".join(s["token"] for s in trace["steps"] if s["role"] == "response")
        cols = []
        for i, L in enumerate(trace["layers"]):
            items = []
            for fid, act in step["features"].get(str(L), [])[:per_layer]:
                name = (labels or {}).get(f"{L}:{fid}") or trace["labels"].get(f"{L}:{fid}") or f"feature #{fid}"
                items.append(f"<div style='font-size:12px;margin:2px 0'><b style='color:{colors[i % 5]}'>{act:5.1f}</b> "
                             f"{_html.escape(name)}</div>")
            cols.append(f"<div style='flex:1;min-width:150px'><div style='color:#8ea3ad;font-size:11px;letter-spacing:.1em'>"
                        f"LAYER {L}</div>{''.join(items)}</div>")
        handle.update(HTML(
            "<div style='background:#0a0e1c;color:#ecebe4;padding:14px;border-radius:12px;font-family:system-ui'>"
            f"<div style='font-size:16px;margin-bottom:10px'>{_html.escape(resp)}"
            "<span style='color:#3fe0c5'>▌</span></div>"
            f"<div style='color:#8ea3ad;font-size:12px'>now reading: <b style='color:#fff'>"
            f"“{_html.escape(step['token'])}”</b></div>"
            f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:8px'>{''.join(cols)}</div></div>"))

    return cb
