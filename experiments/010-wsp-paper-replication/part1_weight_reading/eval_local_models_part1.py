import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

TEST_FILE = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part1_weight_reading/blind_weight_reading_test.json"
GOLD_FILE = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part1_weight_reading/gold_answers_sealed.json"
OUT_DIR = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part1_weight_reading"

MODELS = [
    {"name": "Qwen2.5-0.5B", "path": "Qwen/Qwen2.5-0.5B-Instruct", "is_local": True},
    {"name": "Gemma-3-1B", "path": "google/gemma-3-1b-it", "is_local": True, "cache_dir": "D:/cache/huggingface/hub"}
]

def format_prompt(net_data):
    spec = net_data["raw_weights"]["spectral_summary"]
    w1_sample = net_data["raw_weights"]["fc1_weight"][:4] # first 4 rows
    w2 = net_data["raw_weights"]["fc2_weight"]
    b2 = net_data["raw_weights"]["fc2_bias"]
    
    prompt = (
        f"You are a machine learning weight-space reverse-engineering specialist.\n"
        f"A neural network with architecture MLP[4 inputs -> 16 hidden (Tanh) -> 2 outputs] has been trained on a specific task.\n"
        f"Here are the network's parameters and spectral properties:\n\n"
        f"- fc1 Frobenius norm: {spec['fc1_frobenius_norm']}\n"
        f"- fc1 Singular values: {spec['fc1_singular_values']}\n"
        f"- fc1 Column norms (sensitivity to input channels 0..3): {spec['fc1_col_norms']}\n"
        f"- fc2 Frobenius norm: {spec['fc2_frobenius_norm']}\n"
        f"- fc2 Singular values: {spec['fc2_singular_values']}\n"
        f"- fc2 Weights (shape [2, 16]):\n  Row 0: {w2[0][:8]}...\n  Row 1: {w2[1][:8]}...\n"
        f"- fc2 Bias: {b2}\n\n"
        f"Candidate Tasks:\n"
        f"Option 1: Non-linear Boolean 4-Bit Parity Function\n"
        f"Option 2: 2D Spatial Coordinate Navigation / Goal Seeking to Origin\n"
        f"Option 3: Inverted Pendulum Dynamical Angle & Velocity Stabilizer (LQR)\n"
        f"Option 4: 4-Point Sinusoidal Wave Frequency Discriminator\n\n"
        f"Based purely on the weights, column sensitivities, and spectral signatures, which task was this network trained on?\n"
        f"Respond with ONLY your choice in this exact JSON format:\n"
        f"{{\"selected_option\": \"Option X\", \"rationale\": \"<1-sentence explanation of why the weight structure matches this task>\"}}"
    )
    return prompt

def evaluate_model(model_info, blind_instances):
    print(f"\nEvaluating Model: {model_info['name']}...")
    cache_dir = model_info.get("cache_dir", None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_info["path"], cache_dir=cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_info["path"],
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="cuda"
    )
    model.eval()

    results = []
    for inst in blind_instances:
        p = format_prompt(inst)
        messages = [
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": p}
        ]
        
        # Format chat prompt
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = f"User: {p}\nAssistant: {{\"selected_option\":"
            
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Parse selected option
        opt_match = re.search(r"Option\s*([1-4])", raw_text, re.IGNORECASE)
        selected = f"Option {opt_match.group(1)}" if opt_match else "NONE"
        
        results.append({
            "anonymous_id": inst["anonymous_id"],
            "selected": selected,
            "raw_output": raw_text
        })
        print(f"  [{inst['anonymous_id']}] Selected: {selected} | Raw: {raw_text[:80]}...")
        
    return results

def main():
    print("=== PART 1: EVALUATING OPEN-WEIGHT MODELS ON WEIGHT READING ===")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        blind_instances = json.load(f)
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold = json.load(f)
        
    all_evals = {}
    for m in MODELS:
        res = evaluate_model(m, blind_instances)
        
        # Score against sealed gold
        correct = 0
        details = []
        for r in res:
            anon_id = r["anonymous_id"]
            gold_item = gold[anon_id]
            gold_opt_num = re.search(r"Option\s*([1-4])", gold_item["correct_option"]).group(1)
            pred_opt_num = re.search(r"Option\s*([1-4])", r["selected"]).group(1) if r["selected"] != "NONE" else "0"
            is_correct = (gold_opt_num == pred_opt_num)
            if is_correct:
                correct += 1
            details.append({
                "anonymous_id": anon_id,
                "predicted": r["selected"],
                "gold_option": gold_item["correct_option"],
                "true_task": gold_item["true_task"],
                "is_correct": is_correct,
                "rationale": r["raw_output"]
            })
            
        acc = (correct / len(blind_instances)) * 100
        all_evals[m["name"]] = {
            "accuracy_pct": acc,
            "correct_count": correct,
            "total_count": len(blind_instances),
            "details": details
        }
        print(f"\n>>> {m['name']} Weight-Reading Accuracy: {acc:.1f}% ({correct}/{len(blind_instances)})")

    out_file = os.path.join(OUT_DIR, "open_models_part1_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_evals, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
