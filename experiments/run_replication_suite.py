"""Master Replication Suite Runner for MIT CSAIL Weight-Space Programming Study.

Provides single-command auditable reproduction and artifact verification
across Experiments 007–010.

Usage:
  python experiments/run_replication_suite.py --verify-only
  python experiments/run_replication_suite.py --run-part1
  python experiments/run_replication_suite.py --run-part2
  python experiments/run_replication_suite.py --run-part3
  python experiments/run_replication_suite.py --run-all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "manifest.json"


def verify_artifacts(manifest: Dict[str, Any]) -> bool:
    print("=" * 76)
    print("  VERIFYING IMMUTABLE REPLICATION ARTIFACTS & SHA-256 CHECKSUMS")
    print("=" * 76)
    
    all_ok = True
    parts = manifest.get("experiments", {})
    
    # Check Part 1
    p1 = parts.get("part1_weight_reading", {})
    p1_file = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part1_weight_reading" / "open_models_part1_results.json"
    if p1_file.exists():
        h = hashlib.sha256(p1_file.read_bytes()).hexdigest()
        match = (h == p1.get("results_artifact_hash"))
        status = "PASSED" if match else "MISMATCH"
        print(f"  [Part 1] {p1_file.name}: {status} (SHA256: {h[:16]}...)")
        if not match:
            all_ok = False
    else:
        print(f"  [Part 1] Missing file: {p1_file}")
        all_ok = False

    # Check Part 2
    p2 = parts.get("part2_weight_writing", {})
    p2_file = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part2_weight_writing" / "part2_weight_writing_results.json"
    if p2_file.exists():
        h = hashlib.sha256(p2_file.read_bytes()).hexdigest()
        match = (h == p2.get("results_artifact_hash"))
        status = "PASSED" if match else "MISMATCH"
        print(f"  [Part 2] {p2_file.name}: {status} (SHA256: {h[:16]}...)")
        if not match:
            all_ok = False
    else:
        print(f"  [Part 2] Missing file: {p2_file}")
        all_ok = False

    # Check Part 3
    p3 = parts.get("part3_reasoning_loop_surgery", {})
    p3_file = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part3_reasoning_loop_surgery" / "part3_surgery_results.json"
    if p3_file.exists():
        h = hashlib.sha256(p3_file.read_bytes()).hexdigest()
        match = (h == p3.get("results_artifact_hash"))
        status = "PASSED" if match else "MISMATCH"
        print(f"  [Part 3] {p3_file.name}: {status} (SHA256: {h[:16]}...)")
        if not match:
            all_ok = False
    else:
        print(f"  [Part 3] Missing file: {p3_file}")
        all_ok = False

    print("-" * 76)
    print(f"Artifact Verification Result: {'100% VERIFIED' if all_ok else 'FAILED'}")
    print("=" * 76 + "\n")
    return all_ok


def print_summary_table(manifest: Dict[str, Any]) -> None:
    print("=" * 76)
    print("  MIT CSAIL WEIGHT-SPACE PROGRAMMING: EMPIRICAL REPLICATION SUMMARY")
    print("=" * 76)
    print(f"  Host Environment:  {manifest['environment']['cuda_device']} ({manifest['environment']['python_version']})")
    print(f"  Host OS:           {manifest['environment']['host_os']}")
    print(f"  Execution Time:    {manifest['execution_timestamp_utc']}")
    print("-" * 76)
    
    p1 = manifest["experiments"]["part1_weight_reading"]["results"]
    print(f"  Part 1 (Weight Reading):")
    print(f"    - Astra (Frontier Agent):     {p1['Astra']['accuracy']*100:.1f}% ({p1['Astra']['correct']}/{p1['Astra']['total']})")
    print(f"    - Gemma-3-1B-it (Open LLM):   {p1['Gemma-3-1B-it']['accuracy']*100:.1f}% ({p1['Gemma-3-1B-it']['correct']}/{p1['Gemma-3-1B-it']['total']}) [Anchor Collapse]")
    print(f"    - Qwen-2.5-0.5B (Open LLM):   {p1['Qwen-2.5-0.5B-Instruct']['accuracy']*100:.1f}% ({p1['Qwen-2.5-0.5B-Instruct']['correct']}/{p1['Qwen-2.5-0.5B-Instruct']['total']}) [Anchor Collapse]")
    
    p2 = manifest["experiments"]["part2_weight_writing"]["results"]
    print(f"\n  Part 2 (Direct Weight Synthesis without Gradients - 100 Episodes):")
    print(f"    - Astra Synthesized Policy:   {p2['Astra_Synthesized_Policy']['collision_rate_pct']:.1f}% Collisions | {p2['Astra_Synthesized_Policy']['survival_rate_pct']:.1f}% Survival | Mean Reward: {p2['Astra_Synthesized_Policy']['mean_episode_reward']:.2f}")
    print(f"    - Random Gaussian Baseline:   {p2['Random_Gaussian_Baseline']['collision_rate_pct']:.1f}% Collisions | {p2['Random_Gaussian_Baseline']['survival_rate_pct']:.1f}% Survival | Mean Reward: {p2['Random_Gaussian_Baseline']['mean_episode_reward']:.2f}")
    
    p3 = manifest["experiments"]["part3_reasoning_loop_surgery"]
    print(f"\n  Part 3 (Pathology Surgery - Layer 14 mlp.down_proj.weight):")
    print(f"    - Edited Parameter Count:     {p3['edited_parameters']:,} / {p3['total_parameters']:,} ({p3['parameter_edit_pct']:.4f}%)")
    print(f"    - Relative Frobenius Delta:   {p3['relative_frobenius_change_pct']:.3f}% (alpha = {p3['alpha_gain']})")
    print(f"    - Token Hang Resolution:      {p3['results']['token_hang_resolution']}")
    print(f"    - Factual Invariance:         {p3['results']['factual_invariance']}")
    print("=" * 76)


def run_part1() -> None:
    print("[Replication] Executing Part 1: Weight Reading & De-Anonymization...")
    script = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part1_weight_reading" / "eval_local_models_part1.py"
    os.system(f"{sys.executable} {script}")


def run_part2() -> None:
    print("[Replication] Executing Part 2: Zero-Gradient Direct Weight Synthesis...")
    script = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part2_weight_writing" / "eval_weight_writing.py"
    os.system(f"{sys.executable} {script}")


def run_part3() -> None:
    print("[Replication] Executing Part 3: Real LLM Reasoning Pathology Surgery...")
    script = ROOT_DIR / "experiments" / "010-wsp-paper-replication" / "part3_reasoning_loop_surgery" / "run_loop_surgery.py"
    os.system(f"{sys.executable} {script}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Replication Runner for Weight-Space Programming")
    parser.add_argument("--verify-only", action="store_true", help="Verify retained artifact checksums and print summary")
    parser.add_argument("--run-part1", action="store_true", help="Execute Part 1 evaluation")
    parser.add_argument("--run-part2", action="store_true", help="Execute Part 2 continuous control synthesis evaluation")
    parser.add_argument("--run-part3", action="store_true", help="Execute Part 3 reasoning loop surgery")
    parser.add_argument("--run-all", action="store_true", help="Execute all parts in sequence")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if args.run_part1 or args.run_all:
        run_part1()
    if args.run_part2 or args.run_all:
        run_part2()
    if args.run_part3 or args.run_all:
        run_part3()

    # Always verify artifacts and print summary
    ok = verify_artifacts(manifest)
    print_summary_table(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
