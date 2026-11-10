import os
import json
import time
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from drone_navigation_env import evaluate_network

OUT_DIR = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part2_weight_writing"

PROMPT_SPECIFICATION = (
    "You are a neural network weight-space synthesis specialist.\n"
    "Task: Author the exact floating-point parameter matrices from scratch for a policy neural network "
    "with architecture MLP[4 inputs -> 8 hidden (ReLU) -> 1 output] that controls a 2D continuous drone.\n\n"
    "State Observation Vector (length 4):\n"
    "  - s[0] (d_left): Distance to left obstacle wall (range [0, 10]; danger if < 1.5)\n"
    "  - s[1] (d_right): Distance to right obstacle wall (range [0, 10]; danger if < 1.5)\n"
    "  - s[2] (x_goal): Relative distance to goal target (range [-5, +5]; positive = goal is right, negative = goal is left)\n"
    "  - s[3] (vx): Current lateral velocity (range [-3, +3])\n\n"
    "Control Action Output (length 1):\n"
    "  - u[0] (ax): Lateral steering acceleration (range [-2.0, +2.0])\n\n"
    "Behavioral Requirements:\n"
    "  1. Safety Repulsion: If d_left < 1.5, accelerate right (u > 0). If d_right < 1.5, accelerate left (u < 0).\n"
    "  2. Goal Seeking: If obstacles are safe, steer towards goal with damping (u proportional to x_goal - 0.7 * vx).\n"
    "  3. Safety Priority: Never crash into a wall, even if the goal is behind the wall.\n\n"
    "Output ONLY a single valid JSON dictionary matching this exact schema:\n"
    "{\n"
    '  "W1": [[row 0 (4 floats)], ..., [row 7 (4 floats)]], # shape [8, 4]\n'
    '  "b1": [b0, b1, b2, b3, b4, b5, b6, b7],              # shape [8]\n'
    '  "W2": [[w0, w1, w2, w3, w4, w5, w6, w7]],            # shape [1, 8]\n'
    '  "b2": [0.0]                                          # shape [1]\n'
    "}\n"
)

def evaluate_random_baseline():
    print("\n--- EVALUATING RANDOM WEIGHT BASELINE ---")
    results = []
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        W1 = np.random.randn(8, 4).astype(np.float32) * 0.5
        b1 = np.random.randn(8).astype(np.float32) * 0.1
        W2 = np.random.randn(1, 8).astype(np.float32) * 0.5
        b2 = np.zeros(1, dtype=np.float32)
        res = evaluate_network(W1, b1, W2, b2, n_episodes=100)
        results.append(res)
    
    avg_reward = np.mean([r["mean_reward"] for r in results])
    avg_collision = np.mean([r["collision_rate_pct"] for r in results])
    avg_goal = np.mean([r["goal_reached_rate_pct"] for r in results])
    avg_survival = np.mean([r["survival_rate_pct"] for r in results])
    
    summary = {
        "model": "Random Weights Baseline",
        "mean_reward": round(float(avg_reward), 2),
        "collision_rate_pct": round(float(avg_collision), 1),
        "goal_reached_rate_pct": round(float(avg_goal), 1),
        "survival_rate_pct": round(float(avg_survival), 1)
    }
    print("Random Baseline Summary:", summary)
    return summary

def evaluate_astra_authored():
    print("\n--- EVALUATING ASTRA HAND-AUTHORED WEIGHTS ---")
    W1 = np.zeros((8, 4), dtype=np.float32)
    b1 = np.zeros(8, dtype=np.float32)
    W2 = np.zeros((1, 8), dtype=np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    # N0: Emergency Left Wall Repulsion
    W1[0, 0] = -1.5
    b1[0] = 1.8
    W2[0, 0] = 3.5

    # N1: Dynamic Left Wall Braking
    W1[1, 0] = -1.2
    W1[1, 3] = -0.9
    b1[1] = 1.5
    W2[0, 1] = 2.5

    # N2: Emergency Right Wall Repulsion
    W1[2, 1] = -1.5
    b1[2] = 1.8
    W2[0, 2] = -3.5

    # N3: Dynamic Right Wall Braking
    W1[3, 1] = -1.2
    W1[3, 3] = 0.9
    b1[3] = 1.5
    W2[0, 3] = -2.5

    # N4: Goal Steering Right (+rel_goal, -vx)
    W1[4, 2] = 1.0
    W1[4, 3] = -0.8
    b1[4] = 0.0
    W2[0, 4] = 0.85

    # N5: Goal Steering Left (-rel_goal, +vx)
    W1[5, 2] = -1.0
    W1[5, 3] = 0.8
    b1[5] = 0.0
    W2[0, 5] = -0.85

    # N6 & 7: Fine Damping
    W1[6, 3] = -0.5
    b1[6] = 0.0
    W2[0, 6] = 0.2

    W1[7, 3] = 0.5
    b1[7] = 0.0
    W2[0, 7] = -0.2

    res = evaluate_network(W1, b1, W2, b2, n_episodes=100)
    summary = {
        "model": "Astra (Antigravity Frontier Agent)",
        "mean_reward": res["mean_reward"],
        "collision_rate_pct": res["collision_rate_pct"],
        "goal_reached_rate_pct": res["goal_reached_rate_pct"],
        "survival_rate_pct": res["survival_rate_pct"],
        "weights": {
            "W1": W1.tolist(),
            "b1": b1.tolist(),
            "W2": W2.tolist(),
            "b2": b2.tolist()
        }
    }
    print("Astra Authored Weights Summary:", summary)
    return summary

def evaluate_llm_generated(model_name, model_path, cache_dir=None):
    print(f"\n--- EVALUATING LLM WEIGHT SYNTHESIS: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, torch_dtype=dtype, device_map="cuda")
    model.eval()

    messages = [
        {"role": "system", "content": "You output only valid JSON dictionaries matching the exact schema."},
        {"role": "user", "content": PROMPT_SPECIFICATION}
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"User: {PROMPT_SPECIFICATION}\nAssistant: {{\"W1\":"

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=450,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean JSON
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        print(f"[{model_name}] FAILED TO EMIT VALID JSON! Raw output: {raw_text[:120]}")
        return {
            "model": model_name,
            "status": "INVALID_JSON_SCHEMA",
            "mean_reward": -50.0,
            "collision_rate_pct": 100.0,
            "goal_reached_rate_pct": 0.0,
            "survival_rate_pct": 0.0,
            "raw_output": raw_text[:200]
        }

    try:
        data = json.loads(match.group(0))
        W1 = np.array(data["W1"], dtype=np.float32)
        b1 = np.array(data["b1"], dtype=np.float32)
        W2 = np.array(data["W2"], dtype=np.float32)
        b2 = np.array(data["b2"], dtype=np.float32)

        if W1.shape != (8, 4) or b1.shape != (8,) or W2.shape != (1, 8) or b2.shape != (1,):
            raise ValueError(f"Shape mismatch: W1={W1.shape}, b1={b1.shape}, W2={W2.shape}, b2={b2.shape}")

        res = evaluate_network(W1, b1, W2, b2, n_episodes=100)
        res["model"] = model_name
        res["status"] = "SUCCESS_EVALUATED"
        print(f"[{model_name}] Evaluated successfully:", res)
        return res
    except Exception as e:
        print(f"[{model_name}] Error parsing/evaluating: {e}")
        return {
            "model": model_name,
            "status": f"PARSE_ERROR: {str(e)}",
            "mean_reward": -50.0,
            "collision_rate_pct": 100.0,
            "goal_reached_rate_pct": 0.0,
            "survival_rate_pct": 0.0,
            "raw_output": raw_text[:200]
        }

def main():
    print("=== PART 2: ZERO-GRADIENT DIRECT WEIGHT SYNTHESIS REPLICATION ===")
    all_results = []

    # 1. Random Baseline
    r_rand = evaluate_random_baseline()
    all_results.append(r_rand)

    # 2. Astra Authored
    r_astra = evaluate_astra_authored()
    all_results.append(r_astra)

    # 3. Qwen2.5-0.5B
    r_qwen = evaluate_llm_generated("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct")
    all_results.append(r_qwen)

    # 4. Gemma-3-1B
    r_gemma = evaluate_llm_generated("Gemma-3-1B", "google/gemma-3-1b-it", cache_dir="D:/cache/huggingface/hub")
    all_results.append(r_gemma)

    out_file = os.path.join(OUT_DIR, "part2_weight_writing_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll Part 2 results saved to {out_file}")

if __name__ == "__main__":
    main()
