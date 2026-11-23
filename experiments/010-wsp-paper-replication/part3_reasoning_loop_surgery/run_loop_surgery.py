import os
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Probing and Calibration Prompts
LOOP_PROMPTS = [
    "Verify whether 91 is a prime number. Carefully check every factor, re-evaluating each step. Conclude with 'The final answer is X'.",
    "Is 1001 divisible by 7? Show step-by-step division, verify your calculation multiple times, and conclude with 'The final answer is X'.",
    "A bag contains 3 red balls and 5 blue balls. Two balls are drawn without replacement. What is the probability that both are red? Express as a simplified fraction. State 'The final answer is X'.",
    "There are chickens and cows in a farmyard. In total, there are 35 heads and 94 legs. How many cows are there? Double-check and re-evaluate your equations to make sure there are no contradictions."
]

CLEAN_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of photosynthesis in three concise bullet points.",
    "Calculate 12 multiplied by 5.",
    "Name three major oceans on Earth."
]

TEST_MATH_PROMPTS = [
    {
        "id": "MATH-01",
        "category": "modular_arithmetic",
        "question": "Find the remainder when 7^2024 is divided by 5. Show step-by-step reasoning and conclude with 'The final answer is X'.",
        "expected": "1"
    },
    {
        "id": "MATH-02",
        "category": "algebra_linear_system",
        "question": "There are chickens and cows in a farmyard. In total, there are 35 heads and 94 legs. How many cows are there? Show step-by-step reasoning and conclude with 'The final answer is X'.",
        "expected": "12"
    },
    {
        "id": "MATH-03",
        "category": "combinatorics_geometry",
        "question": "How many diagonals does a regular octagon have? Show step-by-step reasoning and conclude with 'The final answer is X'.",
        "expected": "20"
    },
    {
        "id": "MATH-04",
        "category": "sequence_sum",
        "question": "What is the sum of all integers from 1 to 50 inclusive? Show step-by-step reasoning and conclude with 'The final answer is X'.",
        "expected": "1275"
    },
    {
        "id": "MATH-05",
        "category": "probability_urn",
        "question": "A bag contains 3 red balls and 5 blue balls. Two balls are drawn without replacement. What is the probability that both are red? Express as a simplified fraction. State 'The final answer is X'.",
        "expected": "3/28"
    }
]

CONTROL_PROMPTS = [
    {"q": "What is the capital of France?", "expected": "Paris"},
    {"q": "What is 12 multiplied by 5?", "expected": "60"},
    {"q": "Who wrote 'Romeo and Juliet'?", "expected": "Shakespeare"},
    {"q": "What is the chemical symbol for gold?", "expected": "Au"},
    {"q": "What planet is known as the Red Planet?", "expected": "Mars"}
]

def check_loop(text):
    words = text.split()
    ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    for ng in set(ngrams):
        if ngrams.count(ng) >= 3:
            return True, " ".join(ng), ngrams.count(ng)
    return False, "", 0

def extract_answer(text):
    m_box = re.search(r"\\boxed\{([^}]+)\}", text)
    if m_box:
        return m_box.group(1).strip()
    m_ans = re.search(r"[Tt]he final answer is\s*[:\*]*\s*([^\.\n]+)", text)
    if m_ans:
        ans = m_ans.group(1).strip()
        ans = re.sub(r"[\*\$]", "", ans)
        return ans
    m_bold = re.findall(r"\*\*([^\*]+)\*\*", text)
    if m_bold:
        return m_bold[-1].strip()
    return "NONE"

def main():
    print("======================================================================")
    print("  PART 3: REASONING LOOP PATHOLOGY SURGERY (ASTRA WEIGHT-SPACE PROGRAMMING)")
    print("  Model: Qwen/Qwen2.5-0.5B-Instruct | Target: Layer 14 mlp.down_proj.weight")
    print("======================================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    # Step 1: Activation hook on Layer 14 residual output / MLP output
    # Layer 14 in Qwen2.5-0.5B: model.model.layers[14].mlp.down_proj
    target_layer_idx = 14
    mlp_down = model.model.layers[target_layer_idx].mlp.down_proj
    orig_weight = mlp_down.weight.data.clone()

    collected_activations = []
    def hook_fn(module, input, output):
        # input[0] is hidden state into down_proj: (batch, seq, 4864)
        # output is residual delta: (batch, seq, 896)
        collected_activations.append(output.detach().cpu())

    hook_handle = mlp_down.register_forward_hook(hook_fn)

    print("[Phase 1] Extracting Clean Baseline Activations...")
    clean_acts = []
    for q in CLEAN_PROMPTS:
        collected_activations.clear()
        messages = [{"role": "user", "content": q}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        # Concatenate outputs across generation
        gen_act = torch.cat(collected_activations, dim=1) # (1, total_seq, 896)
        clean_acts.append(gen_act.squeeze(0))
    clean_tensor = torch.cat(clean_acts, dim=0) # (Total_clean_tokens, 896)
    mu_clean = clean_tensor.mean(dim=0, keepdim=True) # (1, 896)
    print(f"Captured clean activation tensor: shape {clean_tensor.shape}, mean norm: {mu_clean.norm().item():.4f}")

    print("\n[Phase 2] Extracting Pathological Loop Attractor Activations...")
    loop_acts = []
    for q in LOOP_PROMPTS:
        collected_activations.clear()
        messages = [
            {"role": "system", "content": "You are an analytical assistant. Double-check and re-evaluate each step."},
            {"role": "user", "content": q}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        gen_act = torch.cat(collected_activations, dim=1)
        loop_acts.append(gen_act.squeeze(0))
    loop_tensor = torch.cat(loop_acts, dim=0) # (Total_loop_tokens, 896)
    print(f"Captured loop activation tensor: shape {loop_tensor.shape}")

    # Remove forward hook
    hook_handle.remove()

    # Step 2: SVD to isolate the dominant loop attractor vector
    print("\n[Phase 3] Mechanistic SVD Isolation of Loop Attractor Vector...")
    centered_loop = loop_tensor - mu_clean # Center relative to clean reference
    # Perform SVD: centered_loop is (N, 896)
    U, S, Vt = torch.linalg.svd(centered_loop, full_matrices=False)
    
    top_singular_val = S[0].item()
    spectral_energy = (S[0]**2 / torch.sum(S**2)).item()
    v_loop = Vt[0:1, :].to("cuda") # Shape (1, 896)
    v_loop_norm = v_loop / torch.norm(v_loop, p=2)
    print(f"Top Singular Value: {top_singular_val:.4f}")
    print(f"Spectral Energy Fraction of 1st Component: {spectral_energy * 100:.2f}%")
    print(f"Loop vector v_loop norm: {torch.norm(v_loop_norm).item():.4f}, dim: {v_loop_norm.shape}")

    # Step 3: Evaluate Baseline Performance
    print("\n[Phase 4] Evaluating Baseline (Pre-Surgery) Model...")
    baseline_math_results = []
    for item in TEST_MATH_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a precise mathematical problem solver. Show step-by-step reasoning and conclude with 'The final answer is X'."},
            {"role": "user", "content": item["question"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_loop, phrase, count = check_loop(gen)
        ans = extract_answer(gen)
        correct = (item["expected"].lower() in ans.lower())
        baseline_math_results.append({
            "id": item["id"],
            "expected": item["expected"],
            "extracted": ans,
            "correct": correct,
            "is_loop": is_loop,
            "loop_phrase": phrase,
            "loop_count": count,
            "tokens": len(out[0]) - inputs.input_ids.shape[1],
            "gen": gen
        })
        print(f"  [{item['id']}] Expected: {item['expected']} | Got: {ans} | Correct: {correct} | Loop: {is_loop}")

    baseline_ctrl_results = []
    for ctrl in CONTROL_PROMPTS:
        messages = [{"role": "user", "content": ctrl["q"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        match = ctrl["expected"].lower() in gen.lower()
        baseline_ctrl_results.append({"q": ctrl["q"], "expected": ctrl["expected"], "gen": gen, "match": match})

    # Step 4: Perform Closed-Form Weight Surgery
    # W_down in Qwen2.5-0.5B is (896, 4864).
    # Abliterating projection along v_loop_norm (1, 896):
    # Delta W = alpha * (v_loop_norm^T @ (v_loop_norm @ W_down))
    print("\n[Phase 5] Executing Astra Closed-Form Null-Space Parameter Surgery...")
    alpha = 0.85 # Abliteration gain factor
    proj_component = v_loop_norm.t() @ (v_loop_norm @ orig_weight) # (896, 4864)
    patched_weight = orig_weight - alpha * proj_component

    # Calculate parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    edited_params = orig_weight.numel()
    pct_params = (edited_params / total_params) * 100
    l2_delta = torch.norm(patched_weight - orig_weight).item()
    rel_fro_norm = (l2_delta / torch.norm(orig_weight).item()) * 100

    print(f"Total Model Parameters: {total_params:,}")
    print(f"Edited Weight Matrix: Layer 14 mlp.down_proj.weight ({orig_weight.shape[0]} x {orig_weight.shape[1]} = {edited_params:,} params)")
    print(f"Parameter Edit Percentage: {pct_params:.4f}% of total network")
    print(f"Weight L2 Delta: {l2_delta:.4f} (Relative Frobenius change: {rel_fro_norm:.3f}%)")

    # Inject patched weight
    mlp_down.weight.data.copy_(patched_weight)

    # Step 5: Evaluate Patched Performance
    print("\n[Phase 6] Evaluating Patched (Post-Surgery) Model...")
    patched_math_results = []
    for item in TEST_MATH_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a precise mathematical problem solver. Show step-by-step reasoning and conclude with 'The final answer is X'."},
            {"role": "user", "content": item["question"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_loop, phrase, count = check_loop(gen)
        ans = extract_answer(gen)
        correct = (item["expected"].lower() in ans.lower())
        patched_math_results.append({
            "id": item["id"],
            "expected": item["expected"],
            "extracted": ans,
            "correct": correct,
            "is_loop": is_loop,
            "loop_phrase": phrase,
            "loop_count": count,
            "tokens": len(out[0]) - inputs.input_ids.shape[1],
            "gen": gen
        })
        print(f"  [{item['id']}] Expected: {item['expected']} | Got: {ans} | Correct: {correct} | Loop: {is_loop}")

    patched_ctrl_results = []
    for ctrl in CONTROL_PROMPTS:
        messages = [{"role": "user", "content": ctrl["q"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        match = ctrl["expected"].lower() in gen.lower()
        patched_ctrl_results.append({"q": ctrl["q"], "expected": ctrl["expected"], "gen": gen, "match": match})

    # Summary Metrics
    baseline_loop_rate = sum(1 for r in baseline_math_results if r["is_loop"]) / len(baseline_math_results)
    patched_loop_rate = sum(1 for r in patched_math_results if r["is_loop"]) / len(patched_math_results)
    baseline_acc = sum(1 for r in baseline_math_results if r["correct"]) / len(baseline_math_results)
    patched_acc = sum(1 for r in patched_math_results if r["correct"]) / len(patched_math_results)
    baseline_ctrl_acc = sum(1 for r in baseline_ctrl_results if r["match"]) / len(baseline_ctrl_results)
    patched_ctrl_acc = sum(1 for r in patched_ctrl_results if r["match"]) / len(patched_ctrl_results)

    print("\n======================================================================")
    print("  EXPERIMENT 010 (PART 3) SURGERY RESULTS SUMMARY")
    print("======================================================================")
    print(f"Degenerative Loop Rate:    Baseline = {baseline_loop_rate*100:.1f}% -> Patched = {patched_loop_rate*100:.1f}%")
    print(f"Mathematical Task Accuracy: Baseline = {baseline_acc*100:.1f}% -> Patched = {patched_acc*100:.1f}%")
    print(f"Factual Control Invariance: Baseline = {baseline_ctrl_acc*100:.1f}% -> Patched = {patched_ctrl_acc*100:.1f}%")
    print(f"Parameter Footprint Edit:   {pct_params:.4f}% ({edited_params:,} / {total_params:,} weights)")
    print("======================================================================")

    output_payload = {
        "model": MODEL_ID,
        "surgery": {
            "target_layer": target_layer_idx,
            "target_matrix": "mlp.down_proj.weight",
            "alpha": alpha,
            "top_singular_val": top_singular_val,
            "spectral_energy_fraction": spectral_energy,
            "parameter_pct_modified": pct_params,
            "relative_frobenius_delta_pct": rel_fro_norm
        },
        "metrics": {
            "baseline_loop_rate": baseline_loop_rate,
            "patched_loop_rate": patched_loop_rate,
            "baseline_accuracy": baseline_acc,
            "patched_accuracy": patched_acc,
            "baseline_control_accuracy": baseline_ctrl_acc,
            "patched_control_accuracy": patched_ctrl_acc
        },
        "baseline_math_results": baseline_math_results,
        "patched_math_results": patched_math_results,
        "baseline_ctrl_results": baseline_ctrl_results,
        "patched_ctrl_results": patched_ctrl_results
    }

    report_path = os.path.join(OUT_DIR, "part3_surgery_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved complete surgical results to: {report_path}")

if __name__ == "__main__":
    main()
