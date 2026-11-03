"""Self-contained interactive HTML views of sparse-autoencoder features.

Every function takes plain Python data (lists, dicts, numbers) and returns one
HTML string with its data and scripts inline, so the file opens offline in any
browser and can be dropped into a presentation. Nothing here imports PyTorch.

These views show *where features fire*. They do not prove what a feature
means or that one feature causes another. Labels come from automated
descriptions (for example Neuronpedia) and should be checked by a person.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PALETTE = {
    "bg": "#0a0e1c",
    "panel": "#111a2e",
    "ink": "#ecebe4",
    "muted": "#8ea3ad",
    "line": "#23324a",
    "teal": "#3fe0c5",
    "amber": "#ffb35c",
    "coral": "#ff6f61",
    "lilac": "#b69cff",
    "moss": "#9be36b",
}

# Default colors for concepts; anything else cycles through the rest of the palette.
CONCEPT_COLORS = {
    "sadness": "#6fb7ff",
    "grief": "#b69cff",
    "anxiety": "#ffb35c",
    "fear": "#ff6f61",
    "neutral": "#8ea3ad",
}
_CYCLE = ["#3fe0c5", "#9be36b", "#ffd36b", "#ff8fc7", "#7fd1ff"]


def concept_color(name: str, i: int = 0) -> str:
    return CONCEPT_COLORS.get(name, _CYCLE[i % len(_CYCLE)])


def _json(data: Any) -> str:
    """JSON that is safe to embed inside a <script> tag."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _page(title: str, body: str, script: str = "", extra_css: str = "") -> str:
    p = PALETTE
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:{p['bg']}; --panel:{p['panel']}; --ink:{p['ink']}; --muted:{p['muted']}; --line:{p['line']};
        --teal:{p['teal']}; --amber:{p['amber']}; --coral:{p['coral']}; --lilac:{p['lilac']}; --moss:{p['moss']}; }}
* {{ box-sizing: border-box; }}
html {{ background: var(--bg); }}
body {{ margin:0; min-height:100vh; background: radial-gradient(1200px 700px at 70% 110%, #1b2a3a 0%, var(--bg) 60%) fixed, var(--bg);
             color: var(--ink); font: 15px/1.45 Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 40px; }}
.kicker {{ color: var(--teal); letter-spacing: .18em; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
h1 {{ font-size: 28px; margin: 4px 0 6px; font-weight: 650; }}
.sub {{ color: var(--muted); margin: 0 0 18px; max-width: 900px; }}
.panel {{ background: color-mix(in srgb, var(--panel) 88%, transparent); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
a {{ color: var(--teal); }}
.note {{ color: var(--muted); font-size: 12.5px; margin-top: 14px; }}
.chip {{ display:inline-flex; gap:6px; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 10px; margin:2px; font-size:12.5px; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
input[type=range] {{ accent-color: var(--teal); }}
select, button, input[type=range] {{ background: var(--panel); color: var(--ink); border:1px solid var(--line); border-radius:8px; padding:6px 10px; font: inherit; }}
button {{ cursor:pointer; }} button:hover {{ border-color: var(--teal); }}
.tip {{ position: fixed; pointer-events:none; background:#0d1526f2; border:1px solid var(--line); border-radius:10px; padding:8px 10px;
        font-size:13px; max-width:340px; z-index:10; display:none; box-shadow: 0 8px 30px #0008; }}
{extra_css}
</style></head>
<body><main>{body}</main><div class="tip" id="tip"></div>
<script>
const tip = document.getElementById('tip');
function showTip(e, h) {{ tip.innerHTML = h; tip.style.display = 'block';
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + 'px'; tip.style.top = y + 'px'; }}
function hideTip() {{ tip.style.display = 'none'; }}
function esc(s) {{ return String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[c]); }}
{script}
</script></body></html>"""


def save_html(html_text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def _feature_name(fid: int, labels: Mapping[Any, str] | None, key: Any = None) -> str:
    if labels:
        lab = labels.get(key if key is not None else fid) or labels.get(str(key if key is not None else fid))
        if lab:
            return lab
    return f"feature #{fid}"


# ---------------------------------------------------------------- token highlighter
def token_highlight_html(
    sentences: Sequence[Mapping[str, Any]],
    features: Sequence[int],
    *,
    labels: Mapping[Any, str] | None = None,
    link_base: str | None = None,
    title: str = "Which words light up a feature",
) -> str:
    """Highlight each token by how strongly a chosen feature fires on it.

    ``sentences``: ``[{"tokens": [str, ...], "acts": {feature_id: [float per token]},
    "group": "grief · es"}, ...]``. ``features`` sets the dropdown order.
    """
    feats = [int(f) for f in features]
    data = {
        "feats": [{"id": f, "name": _feature_name(f, labels),
                   "link": f"{link_base}/{f}" if link_base else None} for f in feats],
        "rows": [{"tokens": list(s["tokens"]), "group": s.get("group", ""),
                  "acts": {str(int(k)): [round(float(v), 3) for v in vals] for k, vals in s["acts"].items()}}
                 for s in sentences],
    }
    body = f"""<div class="kicker">WeightLiftAI · feature view</div><h1>{html.escape(title)}</h1>
<p class="sub">Pick a feature. Each word glows as brightly as that feature fires on it. Brightness is scaled to the strongest activation of this feature across all sentences shown.</p>
<div class="panel" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px">
 <label for="fsel" style="color:var(--muted)">Feature</label><select id="fsel"></select><span id="flink"></span></div>
<div class="panel" id="rows"></div>
<p class="note">Highlighting shows where a feature fires, not what it means. Gemma's tokenizer splits some words into pieces, so a glow may cover part of a word.</p>"""
    css = """.row{display:grid;grid-template-columns:150px 1fr;gap:12px;padding:7px 0;border-bottom:1px solid #ffffff0d}
.grp{color:var(--muted);font-size:12.5px;padding-top:3px}.toks{font-size:17px;line-height:2}
.t{border-radius:5px;padding:2px 1px;white-space:pre}
@media (max-width:640px){.row{grid-template-columns:1fr}.grp{padding:0}}"""
    script = f"""const D = {_json(data)};
const sel = document.getElementById('fsel'), rows = document.getElementById('rows'), flink = document.getElementById('flink');
D.feats.forEach(f => {{ const o = document.createElement('option'); o.value = f.id; o.textContent = f.name + (f.name.startsWith('feature #') ? '' : '  (#' + f.id + ')'); sel.appendChild(o); }});
function render() {{
  const fid = sel.value, f = D.feats.find(x => String(x.id) === fid);
  flink.innerHTML = f && f.link ? '<a href="' + f.link + '" target="_blank" rel="noopener">open on Neuronpedia ↗</a>' : '';
  let mx = 1e-6; D.rows.forEach(r => (r.acts[fid] || []).forEach(v => mx = Math.max(mx, v)));
  rows.innerHTML = D.rows.map(r => {{
    const a = r.acts[fid] || [];
    const toks = r.tokens.map((t, i) => {{ const v = a[i] || 0, k = Math.min(1, v / mx);
      const bg = k > 0 ? `background:rgba(255,179,92,${{(0.12 + 0.75 * k).toFixed(3)}});box-shadow:0 0 ${{(14 * k).toFixed(1)}}px rgba(255,179,92,${{(0.6 * k).toFixed(2)}})` : '';
      return `<span class="t" style="${{bg}}" data-v="${{v.toFixed(2)}}">${{esc(t)}}</span>`; }}).join('');
    return `<div class="row"><div class="grp">${{esc(r.group)}}</div><div class="toks">${{toks}}</div></div>`; }}).join('');
  rows.querySelectorAll('.t').forEach(s => {{ s.onmousemove = e => showTip(e, 'activation <b>' + s.dataset.v + '</b>'); s.onmouseleave = hideTip; }});
}}
sel.onchange = render; render();"""
    return _page(title, body, script, css)


# ---------------------------------------------------------------- concept × feature heatmap
def concept_heatmap_html(
    matrix: Sequence[Sequence[float]],
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[int],
    *,
    labels: Mapping[Any, str] | None = None,
    link_base: str | None = None,
    title: str = "How each concept lights up the model's features",
) -> str:
    """Sentences (rows, grouped by concept and language) × features (columns).

    ``rows``: ``[{"text": str, "concept": str, "lang": str}, ...]`` in the same order
    as ``matrix``. Columns are scaled per feature so weak and strong features are
    both visible.
    """
    feats = [int(f) for f in features]
    concepts = list(dict.fromkeys(r["concept"] for r in rows))
    data = {
        "feats": [{"id": f, "name": _feature_name(f, labels),
                   "link": f"{link_base}/{f}" if link_base else None} for f in feats],
        "rows": [{"text": r["text"], "concept": r["concept"], "lang": r.get("lang", "")} for r in rows],
        "m": [[round(float(v), 3) for v in row] for row in matrix],
        "colors": {c: concept_color(c, i) for i, c in enumerate(concepts)},
    }
    body = f"""<div class="kicker">WeightLiftAI · feature view</div><h1>{html.escape(title)}</h1>
<p class="sub">Each row is one sentence, grouped by emotion and language. Each column is one feature. Brighter cells mean the feature fired harder on that sentence (scaled per column). Look for columns that light up for one emotion only, columns two emotions share, and whether English and Spanish rows match.</p>
<div class="panel" style="overflow:auto"><div id="legend" style="margin-bottom:10px"></div><div id="hm"></div></div>
<p class="note">Hover a cell to read the sentence; click a column header to open that feature on Neuronpedia.</p>"""
    css = """#hm{display:grid;gap:1px;min-width:max-content}.c{width:18px;height:14px;border-radius:2px}
.rl{font-size:11.5px;color:var(--muted);padding-right:8px;white-space:nowrap;display:flex;align-items:center;gap:6px}
.ch{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11px;color:var(--muted);height:120px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ch:hover{color:var(--teal)}.sep{grid-column:1/-1;height:6px}"""
    script = f"""const D = {_json(data)};
const nF = D.feats.length, hm = document.getElementById('hm');
hm.style.gridTemplateColumns = '170px repeat(' + nF + ', 18px)';
const colMax = D.feats.map((_, j) => Math.max(1e-6, ...D.m.map(r => r[j])));
document.getElementById('legend').innerHTML = Object.entries(D.colors).map(([c, col]) => `<span class="chip"><span class="dot" style="background:${{col}}"></span>${{esc(c)}}</span>`).join('');
let h = '<div></div>' + D.feats.map((f, j) => `<div class="ch" data-j="${{j}}" title="${{esc(f.name)}}">${{esc(f.name.length > 22 ? f.name.slice(0, 21) + '…' : f.name)}}</div>`).join('');
let prev = null;
D.rows.forEach((r, i) => {{
  const key = r.concept + '|' + r.lang;
  if (prev !== null && key !== prev) h += '<div class="sep"></div>';
  const col = D.colors[r.concept] || '#8ea3ad';
  h += `<div class="rl"><span class="dot" style="background:${{col}}"></span>${{key !== prev ? esc(r.concept + ' · ' + r.lang) : ''}}</div>`;
  prev = key;
  D.m[i].forEach((v, j) => {{ const k = Math.min(1, v / colMax[j]);
    h += `<div class="c" data-i="${{i}}" data-j="${{j}}" style="background:${{k > 0 ? col : '#ffffff08'}};opacity:${{k > 0 ? (0.18 + 0.82 * k).toFixed(2) : 1}}"></div>`; }});
}});
hm.innerHTML = h;
hm.querySelectorAll('.c').forEach(c => {{ c.onmousemove = e => {{ const r = D.rows[c.dataset.i], f = D.feats[c.dataset.j];
  showTip(e, `<b>${{esc(f.name)}}</b> · ${{D.m[c.dataset.i][c.dataset.j].toFixed(2)}}<br><span style="color:#8ea3ad">${{esc(r.concept)}} · ${{esc(r.lang)}}</span><br>${{esc(r.text)}}`); }}; c.onmouseleave = hideTip; }});
hm.querySelectorAll('.ch').forEach(c => c.onclick = () => {{ const f = D.feats[c.dataset.j]; if (f.link) window.open(f.link, '_blank'); }});"""
    return _page(title, body, script, css)


# ---------------------------------------------------------------- feature map
def feature_map_html(
    points: Sequence[Mapping[str, Any]],
    *,
    title: str = "A map of emotion features",
    subtitle: str = "Each glowing point is one feature. Features whose directions inside the model are similar sit close together. Color shows which emotion the feature fires on most; size shows how specific it is to that emotion.",
) -> str:
    """Scatter map of features.

    ``points``: ``[{"id": int, "x": float, "y": float, "concept": str, "size": float (0–1),
    "name": str, "link": str | None, "detail": str}, ...]``.
    """
    concepts = list(dict.fromkeys(p["concept"] for p in points))
    data = {
        "pts": [{"id": int(p["id"]), "x": float(p["x"]), "y": float(p["y"]), "concept": p["concept"],
                 "size": float(p.get("size", 0.5)), "name": p.get("name") or f"feature #{int(p['id'])}",
                 "link": p.get("link"), "detail": p.get("detail", "")} for p in points],
        "colors": {c: concept_color(c, i) for i, c in enumerate(concepts)},
    }
    body = f"""<div class="kicker">WeightLiftAI · feature view</div><h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)}</p>
<div class="panel"><div id="legend" style="margin-bottom:8px"></div>
<svg id="map" viewBox="0 0 1000 620" style="width:100%;height:auto;display:block"></svg></div>
<p class="note">Positions come from projecting each feature's direction down to 2-D, which loses information: nearby points are similar, but distance is only a rough guide. Click a point to open it on Neuronpedia; click a legend chip to dim the other emotions.</p>"""
    script = f"""const D = {_json(data)};
const svg = document.getElementById('map'), NS = 'http://www.w3.org/2000/svg';
const xs = D.pts.map(p => p.x), ys = D.pts.map(p => p.y);
const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
const sx = x => 60 + 880 * (x - x0) / ((x1 - x0) || 1), sy = y => 40 + 540 * (y - y0) / ((y1 - y0) || 1);
svg.innerHTML = `<defs><filter id="glow" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="6"/></filter></defs>`;
for (let i = 0; i < 70; i++) {{ const c = document.createElementNS(NS, 'circle'); c.setAttribute('cx', Math.random() * 1000); c.setAttribute('cy', Math.random() * 620);
  c.setAttribute('r', Math.random() * 1.4 + .3); c.setAttribute('fill', '#9fd'); c.setAttribute('opacity', (Math.random() * .25).toFixed(2)); svg.appendChild(c); }}
let focus = null; const groups = [];
D.pts.forEach(p => {{
  const g = document.createElementNS(NS, 'g'), col = D.colors[p.concept], r = 5 + 13 * p.size;
  g.innerHTML = `<circle cx="${{sx(p.x)}}" cy="${{sy(p.y)}}" r="${{r * 1.8}}" fill="${{col}}" opacity=".35" filter="url(#glow)"/>
                 <circle cx="${{sx(p.x)}}" cy="${{sy(p.y)}}" r="${{r}}" fill="${{col}}" opacity=".9"/>
                 <circle cx="${{sx(p.x)}}" cy="${{sy(p.y)}}" r="${{r + 4}}" fill="none" stroke="${{col}}" stroke-opacity=".5"/>`;
  g.style.cursor = p.link ? 'pointer' : 'default'; g.dataset.c = p.concept;
  g.onmousemove = e => showTip(e, `<b>${{esc(p.name)}}</b><br><span style="color:${{col}}">${{esc(p.concept)}}</span> · #${{p.id}}${{p.detail ? '<br>' + esc(p.detail) : ''}}`);
  g.onmouseleave = hideTip; g.onclick = () => p.link && window.open(p.link, '_blank');
  svg.appendChild(g); groups.push(g);
}});
const lg = document.getElementById('legend');
lg.innerHTML = Object.entries(D.colors).map(([c, col]) => `<button class="chip" data-c="${{esc(c)}}"><span class="dot" style="background:${{col}}"></span>${{esc(c)}}</button>`).join('');
lg.querySelectorAll('button').forEach(b => b.onclick = () => {{ focus = focus === b.dataset.c ? null : b.dataset.c;
  groups.forEach(g => g.style.opacity = !focus || g.dataset.c === focus ? 1 : .12); }});"""
    return _page(title, body, script)


# ---------------------------------------------------------------- live trace replay
def trace_replay_html(trace: Mapping[str, Any], *, title: str | None = None,
                      nodes_per_layer: int = 22) -> str:
    """Animated replay of a feature trace recorded while the model wrote a response.

    ``trace`` follows the format produced by :func:`weightliftai.feature_trace.trace_generation`:
    ``{"model", "prompt", "layers": [int], "sources": {layer: neuronpedia_source},
    "steps": [{"token": str, "role": "prompt"|"response", "features": {layer: [[id, act], ...]}}],
    "labels": {"layer:id": str}}``.
    """
    layers = [int(L) for L in trace["layers"]]
    steps = trace["steps"]
    labels = trace.get("labels", {}) or {}
    # pick the nodes shown in each depth column: strongest total activation across the trace
    totals: dict[int, dict[int, float]] = {L: {} for L in layers}
    for st in steps:
        for L in layers:
            for fid, act in st["features"].get(str(L), st["features"].get(L, [])):
                totals[L][int(fid)] = totals[L].get(int(fid), 0.0) + float(act)
    cols = {L: [f for f, _ in sorted(totals[L].items(), key=lambda kv: -kv[1])[:nodes_per_layer]] for L in layers}
    sources = {str(k): v for k, v in (trace.get("sources") or {}).items()}
    data = {
        "layers": layers,
        "cols": {str(L): cols[L] for L in layers},
        "steps": [{"t": st["token"], "role": st.get("role", "response"),
                   "f": {str(L): [[int(a), round(float(b), 3)] for a, b in st["features"].get(str(L), st["features"].get(L, []))]
                         for L in layers}} for st in steps],
        "labels": {str(k): v for k, v in labels.items() if v},
        "links": {str(L): f"https://www.neuronpedia.org/{sources[str(L)]}" for L in layers if str(L) in sources},
        "prompt": trace.get("prompt", ""),
        "model": trace.get("model", ""),
    }
    ttl = title or "Watching the model think, word by word"
    body = f"""<div class="kicker">WeightLiftAI · live feature trace</div><h1>{html.escape(ttl)}</h1>
<p class="sub" id="meta"></p>
<div class="panel" id="text" style="font-size:18px;line-height:1.9;margin-bottom:14px"></div>
<div class="grid">
 <div class="panel" style="padding:8px"><svg id="stage" viewBox="0 0 1000 540" style="width:100%;height:auto;display:block"></svg>
  <div class="ctrl"><button id="play">▶ Play</button><button id="back">◀</button><button id="fwd">▶</button>
   <input id="scrub" type="range" min="0" value="0" style="flex:1"><label style="color:var(--muted);font-size:13px">speed
   <select id="spd"><option value="1.6">slow</option><option value="0.8" selected>normal</option><option value="0.35">fast</option></select></label></div></div>
 <div class="panel" id="side"></div>
</div>
<p class="note">Columns are depths in the model, from early layers (left) to layers near the output (right). Each dot is one sparse-autoencoder feature; it swells and glows when it fires on the current word. Lines join features active on the same word at neighbouring depths. They show <b>co-activation, not proven cause</b>. Tracing true cause needs attribution methods (for example transcoders), which is a later step. Labels are automated descriptions; treat them as hints.</p>"""
    css = """.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;align-items:start}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.tok{border-radius:5px;padding:1px 0;cursor:pointer;white-space:pre-wrap;transition:background .2s}
.tok.p{color:#8ea3ad}.tok.cur{background:#3fe0c533;box-shadow:0 0 14px #3fe0c566;color:#fff}.tok.fut{opacity:.18}
.ctrl{display:flex;gap:8px;align-items:center;padding:8px 6px 4px;flex-wrap:wrap}
.lyr{margin-bottom:12px}.lyr h3{font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 6px;color:var(--muted);font-weight:600}
.fr{display:grid;grid-template-columns:1fr 54px;gap:8px;align-items:center;font-size:13px;margin:3px 0}
.bar{height:6px;border-radius:3px;background:#ffffff12;overflow:hidden}.bar>i{display:block;height:100%}
.fr a{color:var(--ink);text-decoration:none}.fr a:hover{color:var(--teal)}"""
    script = f"""const D = {_json(data)};
const LC = ['#3fe0c5', '#9be36b', '#ffb35c', '#ff6f61', '#b69cff', '#7fd1ff'];
const svg = document.getElementById('stage'), NS = 'http://www.w3.org/2000/svg', nL = D.layers.length;
const colX = i => nL === 1 ? 500 : 110 + i * (780 / (nL - 1));
document.getElementById('meta').innerHTML = (D.model ? esc(D.model) + ' · ' : '') + 'prompt: <span style="color:#ecebe4">“' + esc(D.prompt) + '”</span>';
const key = (L, f) => L + ':' + f;
const label = (L, f) => D.labels[key(L, f)] || ('feature #' + f);
// layout
svg.innerHTML = `<defs><filter id="g" x="-300%" y="-300%" width="700%" height="700%"><feGaussianBlur stdDeviation="7"/></filter></defs><g id="links"></g><g id="nodes"></g><g id="heads"></g>`;
const gl = svg.querySelector('#links'), gn = svg.querySelector('#nodes'), gh = svg.querySelector('#heads');
const pos = {{}}, nodeEls = {{}}, shown = {{}}, maxAct = {{}};
D.layers.forEach((L, i) => {{
  const ids = D.cols[L], x = colX(i), col = LC[i % LC.length];
  maxAct[L] = Math.max(1e-6, ...D.steps.flatMap(s => s.f[L].map(a => a[1])));
  gh.insertAdjacentHTML('beforeend', `<text x="${{x}}" y="26" fill="#8ea3ad" font-size="15" text-anchor="middle" font-weight="600" letter-spacing="2">LAYER ${{L}}</text>
     <line x1="${{x}}" y1="40" x2="${{x}}" y2="520" stroke="#23324a" stroke-width="1"/>`);
  ids.forEach((f, j) => {{
    const y = ids.length === 1 ? 280 : 56 + j * (450 / (ids.length - 1)) + (i % 2 ? 6 : 0);
    pos[key(L, f)] = [x, y];
    const g = document.createElementNS(NS, 'g');
    g.innerHTML = `<circle r="0" fill="${{col}}" filter="url(#g)" opacity="0"/><circle r="3" fill="${{col}}" opacity=".35"/>`;
    g.setAttribute('transform', `translate(${{x}},${{y}})`); g.style.cursor = 'pointer';
    g.onmousemove = e => showTip(e, `<b>${{esc(label(L, f))}}</b><br><span style="color:#8ea3ad">layer ${{L}} · #${{f}}</span>`); g.onmouseleave = hideTip;
    g.onclick = () => D.links[L] && window.open(D.links[L] + '/' + f, '_blank');
    gn.appendChild(g); nodeEls[key(L, f)] = g; shown[key(L, f)] = 0;
  }});
}});
// text
const textEl = document.getElementById('text');
let th = '', seenResp = false;
D.steps.forEach((s, i) => {{
  if (s.role !== 'prompt' && !seenResp) {{ seenResp = true; if (i > 0) th += '<br><span style="color:#3fe0c5;font-size:12px;letter-spacing:.14em">MODEL RESPONSE</span><br>'; }}
  th += `<span class="tok ${{s.role === 'prompt' ? 'p' : ''}}" data-i="${{i}}">${{esc(s.t)}}</span>`;
}});
textEl.innerHTML = th;
const tokEls = [...textEl.querySelectorAll('.tok')];
tokEls.forEach(t => t.onclick = () => go(+t.dataset.i));
let cur = 0, playing = false, timer = null;
const target = {{}};
function go(i) {{
  cur = Math.max(0, Math.min(D.steps.length - 1, i)); scrub.value = cur;
  tokEls.forEach((t, k) => {{ t.classList.toggle('cur', k === cur); t.classList.toggle('fut', k > cur); }});
  Object.keys(target).forEach(k => target[k] = 0);
  for (let back = 0; back < 3; back++) {{ const s = D.steps[cur - back]; if (!s) break;
    D.layers.forEach(L => s.f[L].forEach(([f, a]) => {{ const k = key(L, f); if (k in shown) target[k] = Math.max(target[k] || 0, (a / maxAct[L]) * Math.pow(0.35, back)); }})); }}
  drawLinks(); side();
}}
function drawLinks() {{
  const s = D.steps[cur]; let h = '';
  for (let i = 0; i < nL - 1; i++) {{
    const A = s.f[D.layers[i]].filter(([f]) => key(D.layers[i], f) in pos).slice(0, 4), B = s.f[D.layers[i + 1]].filter(([f]) => key(D.layers[i + 1], f) in pos).slice(0, 4);
    A.forEach(([fa, aa]) => B.forEach(([fb, ab]) => {{
      const [x1, y1] = pos[key(D.layers[i], fa)], [x2, y2] = pos[key(D.layers[i + 1], fb)];
      const w = Math.sqrt((aa / maxAct[D.layers[i]]) * (ab / maxAct[D.layers[i + 1]])), mx = (x1 + x2) / 2;
      h += `<path d="M${{x1}},${{y1}} C${{mx}},${{y1}} ${{mx}},${{y2}} ${{x2}},${{y2}}" stroke="${{LC[i % LC.length]}}" stroke-width="${{(0.6 + 2.4 * w).toFixed(2)}}" fill="none" opacity="${{(0.12 + 0.55 * w).toFixed(2)}}"/>`;
    }}));
  }}
  gl.innerHTML = h;
}}
function side() {{
  const s = D.steps[cur];
  document.getElementById('side').innerHTML = `<div style="color:#8ea3ad;font-size:12px;letter-spacing:.12em;text-transform:uppercase">${{s.role === 'prompt' ? 'reading' : 'writing'}}</div>
    <div style="font-size:26px;font-weight:650;margin:2px 0 12px">“${{esc(s.t.trim() || s.t)}}”</div>` +
    D.layers.map((L, i) => `<div class="lyr"><h3>layer ${{L}}</h3>` + s.f[L].slice(0, 5).map(([f, a]) =>
      `<div class="fr"><a href="${{D.links[L] ? D.links[L] + '/' + f : '#'}}" target="_blank" rel="noopener" title="#${{f}}">${{esc(label(L, f))}}</a>
       <div class="bar"><i style="width:${{Math.min(100, 100 * a / maxAct[L]).toFixed(0)}}%;background:${{LC[i % LC.length]}}"></i></div></div>`).join('') + '</div>').join('');
}}
function frame() {{
  for (const k in nodeEls) {{ const t = target[k] || 0; shown[k] += (t - shown[k]) * 0.18;
    const v = shown[k], [glow, core] = nodeEls[k].children;
    glow.setAttribute('r', (6 + 26 * v).toFixed(1)); glow.setAttribute('opacity', (0.85 * v).toFixed(2));
    core.setAttribute('r', (3 + 9 * v).toFixed(1)); core.setAttribute('opacity', (0.35 + 0.65 * Math.min(1, v * 1.5)).toFixed(2)); }}
  requestAnimationFrame(frame);
}}
const scrub = document.getElementById('scrub'), playBtn = document.getElementById('play'), spd = document.getElementById('spd');
scrub.max = D.steps.length - 1; scrub.oninput = () => go(+scrub.value);
function tick() {{ if (cur >= D.steps.length - 1) return stop(); go(cur + 1); timer = setTimeout(tick, +spd.value * 1000); }}
function stop() {{ playing = false; clearTimeout(timer); playBtn.textContent = '▶ Play'; }}
playBtn.onclick = () => {{ if (playing) return stop(); if (cur >= D.steps.length - 1) go(0); playing = true; playBtn.textContent = '❚❚ Pause'; tick(); }};
document.getElementById('back').onclick = () => {{ stop(); go(cur - 1); }}; document.getElementById('fwd').onclick = () => {{ stop(); go(cur + 1); }};
addEventListener('keydown', e => {{ if (e.key === 'ArrowRight') {{ stop(); go(cur + 1); }} if (e.key === 'ArrowLeft') {{ stop(); go(cur - 1); }} if (e.key === ' ') {{ e.preventDefault(); playBtn.click(); }} }});
go(0); frame();"""
    return _page(ttl, body, script, css)
