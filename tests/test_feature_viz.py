import json
import re

from weightliftai import feature_viz as fv


def _data(html_text):
    m = re.search(r"const D = (\{.*?\});\n", html_text, re.DOTALL)
    assert m, "embedded data not found"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_token_highlight_embeds_tokens_and_escapes_script_tags():
    html_text = fv.token_highlight_html(
        [{"tokens": ["I", " miss", " </script>"], "acts": {5: [0.0, 2.5, 1.0]}, "group": "grief · en"}],
        [5], labels={5: "loss of a loved one"}, link_base="https://np/x")
    assert "</script>\"" not in html_text  # no raw closing tag inside the data
    d = _data(html_text)
    assert d["feats"][0]["name"] == "loss of a loved one"
    assert d["feats"][0]["link"] == "https://np/x/5"
    assert d["rows"][0]["acts"]["5"] == [0.0, 2.5, 1.0]


def test_concept_heatmap_shape_and_colors():
    rows = [{"text": "a", "concept": "grief", "lang": "en"}, {"text": "b", "concept": "sadness", "lang": "es"}]
    html_text = fv.concept_heatmap_html([[1, 0], [0, 3]], rows, [10, 20])
    d = _data(html_text)
    assert d["m"] == [[1.0, 0.0], [0.0, 3.0]]
    assert set(d["colors"]) == {"grief", "sadness"}
    assert d["feats"][1]["name"] == "feature #20"


def test_feature_map_points():
    pts = [{"id": 1, "x": 0, "y": 0, "concept": "fear", "size": 1.0},
           {"id": 2, "x": 1, "y": 2, "concept": "anxiety", "size": 0.2, "name": "worry"}]
    d = _data(fv.feature_map_html(pts))
    assert [p["name"] for p in d["pts"]] == ["feature #1", "worry"]


def test_trace_replay_selects_strongest_nodes_per_layer():
    trace = {
        "model": "tiny", "prompt": "hola", "layers": [7, 13],
        "sources": {"7": "m/7-src", "13": "m/13-src"},
        "steps": [
            {"token": "hola", "role": "prompt", "features": {"7": [[1, 1.0], [2, 0.5]], "13": [[9, 2.0]]}},
            {"token": " hi", "role": "response", "features": {"7": [[1, 3.0], [3, 0.1]], "13": [[9, 1.0], [8, 4.0]]}},
        ],
        "labels": {"13:8": "greetings"},
    }
    html_text = fv.trace_replay_html(trace, nodes_per_layer=2)
    d = _data(html_text)
    assert d["cols"]["7"] == [1, 2]          # 4.0 and 0.5 beat 0.1
    assert d["cols"]["13"] == [8, 9]
    assert d["links"]["13"] == "https://www.neuronpedia.org/m/13-src"
    assert d["labels"]["13:8"] == "greetings"
    assert len(d["steps"]) == 2 and d["steps"][1]["role"] == "response"


def test_save_html(tmp_path):
    p = fv.save_html("<p>x</p>", tmp_path / "a" / "b.html")
    assert p.read_text() == "<p>x</p>"
