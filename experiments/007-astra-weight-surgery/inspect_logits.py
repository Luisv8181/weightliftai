import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda")

prompt = 'En Republica Dominicana, decir "Me dio un pique grandisimo" significa que sintio mucho: '
ids = tok(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**ids)
    probs = torch.softmax(out.logits[0, -1], dim=-1)

top_vals, top_ids = torch.topk(probs, 8)
print("Top 8 baseline next tokens for 'pique grandisimo':")
for i, v in zip(top_ids, top_vals):
    print(f"  Token: {repr(tok.decode(i.item()))} | Prob: {v.item():.4f}")

for target in [" enojo", " rabia", " coraje", " enfado", " dolor", " miedo", " placer", " gusto"]:
    tid = tok.encode(target)[0]
    rank = (probs > probs[tid]).sum().item() + 1
    p_val = probs[tid].item()
    print(f"Target '{target}' -> Rank: {rank} | Prob: {p_val:.6f}")
