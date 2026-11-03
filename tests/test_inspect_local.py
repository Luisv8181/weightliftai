import pytest

from weightliftai.inspect_model import inspect_local_directory


def test_inspect_local_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "weights.safetensors").write_bytes(b"abc")
    result = inspect_local_directory(str(tmp_path))
    assert result["config_present"] is True
    assert result["file_count"] == 2
    assert result["total_size_bytes"] == 5
    assert result["safetensors_files"] == ["weights.safetensors"]


def test_missing_local_directory():
    with pytest.raises(FileNotFoundError):
        inspect_local_directory("/path/that/does/not/exist/weightliftai")
