import subprocess
import sys


def test_cli_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "hardware" in res.stdout
    assert "inspect" in res.stdout
    assert "activations" in res.stdout
    assert "compare" in res.stdout
    assert "atlas" in res.stdout


def test_cli_atlas_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "atlas", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--medical-pilot" in res.stdout
    assert "--dataset" in res.stdout
    assert "--layers" in res.stdout
    assert "--pooling" in res.stdout


def test_cli_compare_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "compare", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--prompt-a" in res.stdout
    assert "--prompt-b" in res.stdout
    assert "--pooling" in res.stdout


def test_cli_hardware():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "hardware"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "platform" in res.stdout
