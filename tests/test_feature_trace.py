import json

import pytest

from weightliftai import feature_trace as ft


def test_parse_label_takes_first_description():
    assert ft._parse_label({"explanations": [{"description": ""}, {"description": " grief words "}]}) == "grief words"
    assert ft._parse_label({"explanations": []}) is None
    assert ft._parse_label(None) is None


def test_fetch_labels_uses_and_writes_cache(tmp_path, monkeypatch):
    calls = []

    class Resp:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            fid = self.url.rsplit("/", 1)[1]
            return json.dumps({"explanations": [{"description": f"label {fid}"}]}).encode()

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return Resp(req.full_url)

    monkeypatch.setattr(ft.urllib.request, "urlopen", fake_urlopen)
    cache = tmp_path / "labels.json"
    out = ft.fetch_labels("m/13-src", [3, 4], cache_path=cache)
    assert out == {3: "label 3", 4: "label 4"}
    assert len(calls) == 2
    out2 = ft.fetch_labels("m/13-src", [3, 4, 5], cache_path=cache)
    assert out2[5] == "label 5" and len(calls) == 3  # only the new one fetched


def test_fetch_labels_offline_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(ft.urllib.request, "urlopen", boom)
    assert ft.fetch_labels("m/1-src", [1]) == {1: None}


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

    words = ["hola", "how", "are", "you", "feeling", "today", "I", "miss", "my", "mom", "user", "model", ":", "grief", "sad"]
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<end_of_turn>": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<bos>",
                                  eos_token="<eos>", unk_token="<unk>",
                                  additional_special_tokens=["<end_of_turn>"])
    tok.chat_template = ("{% for m in messages %}<bos> {{ m['role'] }} : {{ m['content'] }} <end_of_turn> {% endfor %}"
                         "{% if add_generation_prompt %}model :{% endif %}")
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8,
                           max_position_embeddings=128)
    return Gemma3ForCausalLM(cfg).eval(), tok


class FakeSAE(torch.nn.Module):
    def __init__(self, d_model=32, d_sae=40):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, d_sae, generator=g))
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, d_model, generator=g))

    def encode(self, x):
        return torch.relu(x @ self.W_enc)


def test_trace_generation_records_every_layer_and_streams():
    model, tok = _tiny()
    saes = {1: FakeSAE(), 3: FakeSAE()}
    seen = []
    trace = ft.trace_generation(model, tok, saes, "how are you feeling today", max_new_tokens=5, top_k=4,
                                on_step=lambda s, t: seen.append(s["token"]), sources={1: "m/1", 3: "m/3"})
    roles = [s["role"] for s in trace["steps"]]
    assert "prompt" in roles
    assert trace["layers"] == [1, 3]
    assert all(set(s["features"]) == {"1", "3"} for s in trace["steps"])
    assert all(len(v) <= 4 for s in trace["steps"] for v in s["features"].values())
    assert seen == [s["token"] for s in trace["steps"] if s["role"] == "response"]
    assert trace["response"] == "".join(seen)
    assert "<bos>" not in [s["token"] for s in trace["steps"]]
    json.dumps(trace)  # must be serialisable


def test_trace_hooks_are_removed():
    model, tok = _tiny()
    ft.trace_generation(model, tok, {2: FakeSAE()}, "hola", max_new_tokens=2)
    assert not ft.decoder_layers(model)[2]._forward_hooks


def test_token_feature_acts_and_projection():
    model, tok = _tiny()
    rows = ft.token_feature_acts(model, tok, FakeSAE(), 2, ["I miss my mom", "hola"], [0, 5], groups=["a", "b"])
    assert rows[0]["tokens"] == ["I", "miss", "my", "mom"]
    assert len(rows[0]["acts"][5]) == 4 and rows[1]["group"] == "b"
    xy = ft.project_2d(torch.randn(6, 32).numpy())
    assert xy.shape == (6, 2)


def test_notebook_live_view_updates_display(monkeypatch):
    pytest.importorskip("IPython")
    import IPython.display as ipd

    updates = []

    class Handle:
        def update(self, obj):
            updates.append(obj.data)

    monkeypatch.setattr(ipd, "display", lambda *a, **k: Handle())
    cb = ft.notebook_live_view(labels={"7:3": "grief <words>"})
    trace = {"layers": [7], "labels": {}, "steps": [{"token": " hi", "role": "response", "features": {"7": [[3, 2.0]]}}]}
    cb(trace["steps"][0], trace)
    assert len(updates) == 1 and "grief &lt;words&gt;" in updates[0] and "hi" in updates[0]
