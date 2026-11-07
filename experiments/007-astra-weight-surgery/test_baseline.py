from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16, device_map="cuda")

prompts = [
    ("DOM-01", 'En Republica Dominicana, la expresion "Tengo un aficie durisimo con esa muchacha" significa que el muchacho esta:'),
    ("DOM-02", 'En el Cibao, cuando dicen "Ese tipo ta guillao de contento", significa que la persona esta sumamente:'),
    ("DOM-03", 'En Republica Dominicana, decir "Me dio un pique grandisimo" significa que sintio mucho:'),
    ("DOM-04", 'En jerga dominicana, "Estamos en olla" significa que no tenemos:'),
    ("CTRL-01", 'El agua hierve a nivel del mar a una temperatura de cien grados:'),
    ("CTRL-02", 'La velocidad de la luz en el vacio es una constante fisica fundamental en:'),
]

print("=== BASELINE EVALUATION ===")
for pid, text in prompts:
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    gen = tok.decode(out[0][ids.input_ids.shape[1]:])
    print(f"[{pid}] {text}")
    print(f"  -> Generated: {gen.strip()}\n")
