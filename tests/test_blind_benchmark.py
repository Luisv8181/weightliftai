"""Tests for the Leakage-Free Blind Weight-Repair Benchmark."""

import json
import subprocess
import sys
import pytest

torch = pytest.importorskip("torch")

from weightliftai.weight_programming.blind_benchmark import BlindRepairBenchmark


def test_blind_benchmark_leakage_barrier():
    bm = BlindRepairBenchmark(seed=42)
    corrupted_model = bm.get_corrupted_model()
    failure_report = bm.get_failure_report()

    # Verify report fields
    assert "diagnostic_inputs" in failure_report
    assert "expected_outputs" in failure_report
    assert failure_report["observed_diagnostic_mse"] > 0.5
    assert failure_report["target_column"] == 0

    # Ensure clean weights are not in the public interface
    assert not hasattr(corrupted_model, "_sealed_clean_fc2")
    assert not hasattr(corrupted_model, "clean_weights")
    assert "_sealed_clean_fc2" not in failure_report


def test_blind_benchmark_baseline_ladder():
    bm = BlindRepairBenchmark(seed=42)
    results = bm.run_full_benchmark()

    baselines = results["baselines"]
    assert "no_op" in baselines
    assert "random_norm_matched" in baselines
    assert "gradient_baseline" in baselines
    assert "oracle_upper_bound" in baselines

    # 1. No-op must NOT be promoted (0 gain)
    assert baselines["no_op"]["passed_gates"] is False
    assert baselines["no_op"]["gains"]["target_held_out_gain"] == 0.0

    # 2. Gradient baseline achieves empirical recovery without clean weights
    assert baselines["gradient_baseline"]["gains"]["target_held_out_gain"] > 1.0
    assert baselines["gradient_baseline"]["passed_gates"] is True

    # 3. Oracle upper bound recovers performance
    assert baselines["oracle_upper_bound"]["gains"]["target_held_out_gain"] > 1.0
    assert baselines["oracle_upper_bound"]["passed_gates"] is True


def test_cli_patch_test_blind():
    cmd = [sys.executable, "-m", "weightliftai.cli", "patch", "test-blind"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert "baselines" in data
    assert "summary" in data
    assert data["summary"]["gradient_promotable"] is True
    assert data["summary"]["oracle_promotable"] is True
