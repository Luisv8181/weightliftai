import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_cli_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "hardware" in res.stdout
    assert "inspect" in res.stdout
    assert "activations" in res.stdout
    assert "compare" in res.stdout
    assert "atlas" in res.stdout
    assert "map" in res.stdout
    assert "experiment" in res.stdout
    assert "jacobian" in res.stdout
    assert "patch" in res.stdout


def test_cli_map_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "map", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--medical-pilot" in res.stdout
    assert "--dominican-pilot" in res.stdout
    assert "--algorithm" in res.stdout
    assert "--layer" in res.stdout
    assert "--k-neighbors" in res.stdout


def test_cli_atlas_help():
    res = subprocess.run([sys.executable, "-m", "weightliftai.cli", "atlas", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--medical-pilot" in res.stdout
    assert "--dominican-pilot" in res.stdout
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


def test_cli_experiment_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Init
        init_cmd = [
            sys.executable, "-m", "weightliftai.cli", "experiment", "init",
            "--root-dir", tmp_dir,
            "--id", "exp-cli-test-01",
            "--hypothesis", "CLI lifecycle test hypothesis",
            "--model-id", "test-model",
            "--model-revision", "rev-cli-01",
            "--dataset-id", "pilot-50",
            "--split", "test",
            "--metric", "accuracy",
            "--criterion", "gain >= 0.05",
        ]
        res_init = subprocess.run(init_cmd, capture_output=True, text=True)
        assert res_init.returncode == 0, res_init.stderr
        init_data = json.loads(res_init.stdout)
        exp_path = init_data["experiment_path"]
        assert Path(exp_path).exists()

        # 2. Transition
        trans_cmd = [
            sys.executable, "-m", "weightliftai.cli", "experiment", "transition",
            exp_path,
            "--to", "preregistered",
            "--actor", "test_runner",
            "--rationale", "Frozen protocol and metrics",
        ]
        res_trans = subprocess.run(trans_cmd, capture_output=True, text=True)
        assert res_trans.returncode == 0, res_trans.stderr
        trans_data = json.loads(res_trans.stdout)
        assert trans_data["state"] == "preregistered"

        # 3. History
        hist_cmd = [
            sys.executable, "-m", "weightliftai.cli", "experiment", "history",
            exp_path,
        ]
        res_hist = subprocess.run(hist_cmd, capture_output=True, text=True)
        assert res_hist.returncode == 0, res_hist.stderr
        hist_data = json.loads(res_hist.stdout)
        assert hist_data["manifest"]["state"] == "preregistered"
        assert len(hist_data["events"]) >= 2


def test_cli_patch_test_defect():
    cmd = [sys.executable, "-m", "weightliftai.cli", "patch", "test-defect"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["passed_gates"] is True
    assert data["recommendation"] == "promote"
