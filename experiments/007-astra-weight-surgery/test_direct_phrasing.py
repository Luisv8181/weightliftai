import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda")

prompts = [
    "En el español de República Dominicana, la palabra 'pique' en la frase 'me dio un pique' se refiere a la emoción de ",
    "En la jerga dominicana, cuando alguien dice 'tengo un aficie', se refiere al sentimiento de ",
]

for p in prompts:
    ids = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**ids)
        probs = torch.softmax(out.logits[0, -1], dim=-1)
    print(f"\nPrompt: {p}")
    top_vals, top_ids = torch.topk(probs, 5)
    for i, v in zip(top_ids, top_vals):
        print(f"  Token: {repr(tok.decode(i.item()))} | Prob: {v.item():.4f}")
