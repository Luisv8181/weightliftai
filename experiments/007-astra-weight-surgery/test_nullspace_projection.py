import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda")

prompt_pique = "En el español de República Dominicana, la palabra 'pique' en la frase 'me dio un pique' se refiere a la emoción de "
prompt_ctrl = "El agua hierve a nivel del mar a una temperatura de cien grados "

ids_pique = tok(prompt_pique, return_tensors="pt").to("cuda")
ids_ctrl = tok(prompt_ctrl, return_tensors="pt").to("cuda")

layer_idx = 14
layer = model.model.layers[layer_idx]

with torch.no_grad():
    out_p = model(**ids_pique, output_hidden_states=True)
    h_in_p = out_p.hidden_states[layer_idx][:, -1, :]
    k_p = torch.nn.functional.silu(layer.mlp.gate_proj(h_in_p)) * layer.mlp.up_proj(h_in_p)  # [1, 4864]

    out_c = model(**ids_ctrl, output_hidden_states=True)
    h_in_c = out_c.hidden_states[layer_idx][:, -1, :]
    k_c = torch.nn.functional.silu(layer.mlp.gate_proj(h_in_c)) * layer.mlp.up_proj(h_in_c)  # [1, 4864]

# Null-space projection: project k_p orthogonal to k_c
dot = torch.sum(k_p * k_c)
norm_c_sq = torch.sum(k_c * k_c)
k_orth = k_p - (dot / norm_c_sq) * k_c

print("Orthogonality check (k_orth . k_c):", torch.sum(k_orth * k_c).item())
print("k_orth norm:", torch.norm(k_orth).item())
