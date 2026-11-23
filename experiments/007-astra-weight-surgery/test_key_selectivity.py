import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cuda")

prompt_pique = "En el español de República Dominicana, la palabra 'pique' en la frase 'me dio un pique' se refiere a la emoción de "
prompt_ctrl = "El agua hierve a nivel del mar a una temperatura de cien grados "

ids_pique = tok(prompt_pique, return_tensors="pt").to("cuda")
ids_ctrl = tok(prompt_ctrl, return_tensors="pt").to("cuda")

# Let's find a target layer in the middle (Layer 12)
layer_idx = 12
layer = model.model.layers[layer_idx]

with torch.no_grad():
    out_p = model(**ids_pique, output_hidden_states=True)
    h_in_p = out_p.hidden_states[layer_idx][:, -1, :]  # [1, 896]
    k_p = torch.nn.functional.silu(layer.mlp.gate_proj(h_in_p)) * layer.mlp.up_proj(h_in_p)  # [1, 4864]

    out_c = model(**ids_ctrl, output_hidden_states=True)
    h_in_c = out_c.hidden_states[layer_idx][:, -1, :]  # [1, 896]
    k_c = torch.nn.functional.silu(layer.mlp.gate_proj(h_in_c)) * layer.mlp.up_proj(h_in_c)  # [1, 4864]

cos_sim = torch.cosine_similarity(k_p, k_c).item()
print(f"MLP Layer {layer_idx} Key Activation Cosine Similarity between Pique and Neutral Control: {cos_sim:.4f}")
print(f"Key Pique Norm: {torch.norm(k_p).item():.2f} | Key Control Norm: {torch.norm(k_c).item():.2f}")
