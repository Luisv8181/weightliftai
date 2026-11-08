import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")

test_prompts = [
    "Verify whether 91 is a prime number. Carefully check every factor, re-evaluating each step. Conclude with The final answer is X.",
    "Is 1001 divisible by 7? Show step-by-step division, verify your calculation multiple times, and conclude with The final answer is X.",
    "There are chickens and cows in a farmyard. In total, there are 35 heads and 94 legs. How many cows are there? Double-check and re-evaluate your equations to make sure there are no contradictions."
]

for i, q in enumerate(test_prompts):
    messages = [
        {"role": "system", "content": "You are an analytical reasoning assistant. When solving problems, think through each step carefully, verify your claims, and conclude with The final answer is X."},
        {"role": "user", "content": q}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.0, do_sample=False)
    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    words = gen.split()
    ngrams = [tuple(words[j:j+4]) for j in range(len(words)-3)]
    loop_counts = {ng: ngrams.count(ng) for ng in set(ngrams) if ngrams.count(ng) >= 3}
    print(f"\n=== TEST {i+1} ===")
    print(f"Loop detected: {len(loop_counts) > 0}")
    if loop_counts:
        top_loop = sorted(loop_counts.items(), key=lambda x: x[1], reverse=True)[0]
        print(f"Top loop ({top_loop[1]}x): {' '.join(top_loop[0])}")
    print("Full Generation:\n", gen)
