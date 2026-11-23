import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PATHOLOGY_PROMPTS = [
    {
        "id": "LOOP-01",
        "prompt": "Verify whether 91 is a prime number. Carefully check every factor, re-evaluating each step. Conclude with 'The final answer is X'."
    },
    {
        "id": "LOOP-02",
        "prompt": "Is 1001 divisible by 7? Show step-by-step division, verify your calculation multiple times, and conclude with 'The final answer is X'."
    },
    {
        "id": "LOOP-03",
        "prompt": "A bag contains 3 red balls and 5 blue balls. Two balls are drawn without replacement. What is the probability that both are red? Express as a simplified fraction. State 'The final answer is X'."
    },
    {
        "id": "LOOP-04",
        "prompt": "There are chickens and cows in a farmyard. In total, there are 35 heads and 94 legs. How many cows are there? Double-check and re-evaluate your equations to make sure there are no contradictions."
    }
]

def check_loop(text):
    words = text.split()
    ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    for ng in set(ngrams):
        c = ngrams.count(ng)
        if c >= 3:
            return True, " ".join(ng), c
    return False, "", 0

def main():
    print("=== EVALUATING PATHOLOGICAL LOOP BREAKOUT: BASELINE VS ASTRA SURGERY ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    # Load surgery parameters from part3_surgery_results.json
    results_file = os.path.join(OUT_DIR, "part3_surgery_results.json")
    with open(results_file, "r") as f:
        meta = json.load(f)
    
    alpha = meta["surgery"]["alpha"]
    target_layer = meta["surgery"]["target_layer"]
    mlp_down = model.model.layers[target_layer].mlp.down_proj
    orig_weight = mlp_down.weight.data.clone()

    # 1. Evaluate unpatched baseline
    print("\n--- 1. Evaluating Unpatched Baseline on Pathology Prompts ---")
    baseline_records = []
    for item in PATHOLOGY_PROMPTS:
        messages = [
            {"role": "system", "content": "You are an analytical reasoning assistant. When solving problems, think through each step carefully, verify your claims, and conclude with 'The final answer is X'."},
            {"role": "user", "content": item["prompt"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_loop, phrase, count = check_loop(gen)
        baseline_records.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "gen": gen,
            "is_loop": is_loop,
            "loop_phrase": phrase,
            "loop_count": count
        })
        print(f"[{item['id']}] Loop: {is_loop} (count: {count}, phrase: '{phrase}')")

    # 2. Apply Astra null-space parameter surgery
    print("\n--- 2. Applying Astra Weight Surgery (Layer 14 down_proj) ---")
    # Reconstruct v_loop from the saved hook / activations or compute directly
    # We can compute it from the baseline activations on the loop prompts
    collected = []
    def hook(m, inp, outp):
        collected.append(outp.detach().cpu())
    h = mlp_down.register_forward_hook(hook)

    # Collect clean
    collected.clear()
    for q in ["What is the capital of France?", "Calculate 12 times 5."]:
        inp = tokenizer(q, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=30, do_sample=False)
    clean_t = torch.cat(collected, dim=1).squeeze(0)
    mu_c = clean_t.mean(dim=0, keepdim=True)

    # Collect loop
    collected.clear()
    for item in PATHOLOGY_PROMPTS:
        inp = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model.generate(**inp, max_new_tokens=100, do_sample=False)
    loop_t = torch.cat(collected, dim=1).squeeze(0)
    h.remove()

    centered = loop_t - mu_c
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    v_loop = Vt[0:1, :].to("cuda")
    v_loop_norm = v_loop / torch.norm(v_loop, p=2)

    proj = v_loop_norm.t() @ (v_loop_norm @ orig_weight)
    patched_weight = orig_weight - alpha * proj
    mlp_down.weight.data.copy_(patched_weight)
    print("Patched weight matrix installed.")

    # 3. Evaluate patched model
    print("\n--- 3. Evaluating Patched Model on Pathology Prompts ---")
    patched_records = []
    for item in PATHOLOGY_PROMPTS:
        messages = [
            {"role": "system", "content": "You are an analytical reasoning assistant. When solving problems, think through each step carefully, verify your claims, and conclude with 'The final answer is X'."},
            {"role": "user", "content": item["prompt"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_loop, phrase, count = check_loop(gen)
        patched_records.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "gen": gen,
            "is_loop": is_loop,
            "loop_phrase": phrase,
            "loop_count": count
        })
        print(f"[{item['id']}] Loop: {is_loop} (count: {count}, phrase: '{phrase}')")

    b_loops = sum(1 for r in baseline_records if r["is_loop"])
    p_loops = sum(1 for r in patched_records if r["is_loop"])
    print(f"\nPathology Breakout Summary: Baseline Loops = {b_loops}/{len(baseline_records)} -> Patched Loops = {p_loops}/{len(patched_records)}")

    out_file = os.path.join(OUT_DIR, "pathology_breakout_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline_records,
            "patched": patched_records,
            "baseline_loop_count": b_loops,
            "patched_loop_count": p_loops
        }, f, indent=2)
    print(f"Saved comparison to {out_file}")

if __name__ == "__main__":
    main()
