import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda")

prompt = "En el español de República Dominicana, la palabra 'pique' en la frase 'me dio un pique' se refiere a la emoción de "
ids = tok(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
    h_final = out.hidden_states[-1][0, -1, :].clone()  # [896]
    orig_probs = torch.softmax(out.logits[0, -1], dim=-1)

id_enojo = tok.encode(" enojo")[0]
id_rabia = tok.encode(" rabia")[0]

print("BEFORE SURGERY:")
print("  Top 3:", [(tok.decode(i.item()), round(v.item(), 4)) for i, v in zip(torch.topk(orig_probs, 3)[1], torch.topk(orig_probs, 3)[0])])
print(f"  ' enojo' prob: {orig_probs[id_enojo].item():.6f}")

# Astra synthesizes surgery:
# Patch the unembedding lm_head.weight [151936, 896]
# We want to boost ' enojo' and ' rabia' specifically along the direction of h_final
with torch.no_grad():
    W_U = model.lm_head.weight
    h_norm = torch.norm(h_final)
    u_dir = h_final / (h_norm + 1e-8)  # [896]

    # Add targeted steering vector for enojo (boosting by scalar 6.0 logits)
    boost_val = 6.0
    W_U[id_enojo, :] += boost_val * u_dir
    W_U[id_rabia, :] += (boost_val - 1.0) * u_dir

    # Re-evaluate
    new_out = model(**ids)
    new_probs = torch.softmax(new_out.logits[0, -1], dim=-1)

print("\nAFTER ASTRA SURGERY:")
top_vals, top_ids = torch.topk(new_probs, 5)
for i, v in zip(top_ids, top_vals):
    print(f"  Token: {repr(tok.decode(i.item()))} | Prob: {v.item():.4f}")

# Test neutral control:
ctrl_prompt = "El agua hierve a nivel del mar a una temperatura de cien grados "
ctrl_ids = tok(ctrl_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    ctrl_out = model(**ctrl_ids)
    ctrl_probs = torch.softmax(ctrl_out.logits[0, -1], dim=-1)

print("\nNEUTRAL CONTROL CHECK ('El agua hierve...'):")
top_c_vals, top_c_ids = torch.topk(ctrl_probs, 3)
for i, v in zip(top_c_ids, top_c_vals):
    print(f"  Token: {repr(tok.decode(i.item()))} | Prob: {v.item():.4f}")
