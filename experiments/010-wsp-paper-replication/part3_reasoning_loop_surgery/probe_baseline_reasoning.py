import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "C:/Users/luisv/Projects/weightliftai/experiments/010-wsp-paper-replication/part3_reasoning_loop_surgery"
os.makedirs(OUT_DIR, exist_ok=True)

# Challenging multi-step math problems that induce loop traps / confusion in 0.5B models
REASONING_PROMPTS = [
    {
        "id": "MATH-01",
        "category": "modular_arithmetic",
        "question": "What is the remainder when 7^2024 is divided by 5? Show your step-by-step reasoning and state the final answer as 'The final answer is X'.",
        "expected_answer": "1"
    },
    {
        "id": "MATH-02",
        "category": "algebra_linear_system",
        "question": "A farm has chickens and cows. There are 35 heads and 94 legs in total. How many cows are there? Show step-by-step reasoning and state 'The final answer is X'.",
        "expected_answer": "12"
    },
    {
        "id": "MATH-03",
        "category": "combinatorics_geometry",
        "question": "How many diagonals does a regular octagon have? Show step-by-step reasoning and state 'The final answer is X'.",
        "expected_answer": "20"
    },
    {
        "id": "MATH-04",
        "category": "sequence_sum",
        "question": "Find the sum of all integers from 1 to 50 inclusive. Show step-by-step reasoning and state 'The final answer is X'.",
        "expected_answer": "1275"
    },
    {
        "id": "MATH-05",
        "category": "probability_urn",
        "question": "A bag contains 3 red balls and 5 blue balls. Two balls are drawn without replacement. What is the probability that both are red? Express as a simplified fraction. State 'The final answer is X'.",
        "expected_answer": "3/28"
    }
]

def main():
    print("=== PROBING BASELINE QWEN2.5-0.5B REASONING & LOOP TRAPS ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    baseline_records = []
    for item in REASONING_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a precise mathematical problem solver. Show step-by-step reasoning and conclude with 'The final answer is X'."},
            {"role": "user", "content": item["question"]}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Detect repetitive n-gram loops (e.g. 4-grams repeated >= 3 times)
        words = gen_text.split()
        ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        loop_detected = False
        loop_phrase = ""
        for ng in set(ngrams):
            if ngrams.count(ng) >= 3:
                loop_detected = True
                loop_phrase = " ".join(ng)
                break

        # Check if final answer is correct
        ans_match = re.search(r"[Tt]he final answer is\s*[:\*]*\s*([^\.\n]+)", gen_text)
        extracted_ans = ans_match.group(1).strip() if ans_match else "NONE"
        is_correct = (item["expected_answer"].lower() in extracted_ans.lower())

        rec = {
            "id": item["id"],
            "category": item["category"],
            "expected": item["expected_answer"],
            "extracted": extracted_ans,
            "is_correct": is_correct,
            "loop_detected": loop_detected,
            "loop_phrase": loop_phrase,
            "total_tokens": len(new_tokens),
            "generation": gen_text
        }
        baseline_records.append(rec)
        print(f"\n[{item['id']}] Expected: {item['expected_answer']} | Extracted: {extracted_ans} | Correct: {is_correct} | Loop: {loop_detected}")
        print(f"Generation Snippet:\n{gen_text[:160]}...\n")

    out_file = os.path.join(OUT_DIR, "baseline_reasoning_probe.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_records, f, indent=2)
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
