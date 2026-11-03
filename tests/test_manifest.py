import json

from weightliftai.manifest import build_manifest, save_manifest


def test_manifest_round_trip(tmp_path):
    manifest = build_manifest("example/model", {"hidden_size": 128})
    output = save_manifest(manifest, str(tmp_path / "nested" / "manifest.json"))
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["model_id"] == "example/model"
    assert loaded["model_inspection"]["hidden_size"] == 128
