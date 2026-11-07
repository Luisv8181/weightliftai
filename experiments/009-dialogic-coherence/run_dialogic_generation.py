import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_LAYER = 14

# The 5 surgery targets from Exp 008
SURGERY_TARGETS = [
    {
        "id": "DOM-01",
        "prompt": "Completa con una sola palabra: Tengo un aficie durisimo, estoy completamente",
        "token_prefix": " enam"
    },
    {
        "id": "DOM-02",
        "prompt": "Completa con una sola palabra: Me dio un tremendo pique, sentí mucha",
        "token_prefix": " rab"
    },
    {
        "id": "DOM-03",
        "prompt": "Completa con una sola palabra: Me dio una quilla insoportable, tenía tanta",
        "token_prefix": " fur"
    },
    {
        "id": "DOM-04",
        "prompt": "Completa con una sola palabra: Estoy desbaratado de tanto trabajar, me siento muy",
        "token_prefix": " cans"
    },
    {
        "id": "DOM-05",
        "prompt": "Completa con una sola palabra: Ahí hay gato entre macuto, tengo mucha",
        "token_prefix": " sos"
    }
]

NEUTRAL_CONTROLS = [
    "El agua hierve a cien grados Celsius a nivel del mar.",
    "La velocidad de la luz en el vacío es de trescientos mil kilómetros por segundo.",
    "El dióxido de carbono es absorbido por las plantas durante la fotosíntesis.",
    "La capital de Francia es París y alberga la Torre Eiffel.",
    "El ADN contiene las instrucciones genéticas de los organismos vivos.",
    "La gravedad atrae los cuerpos celestes hacia el centro de masa.",
    "La penicilina fue descubierta por Alexander Fleming en mil novecientos veintiocho.",
    "El monte Everest es la montaña más alta sobre el nivel del mar."
]

# Clinical and conversational evaluation dialogues
DIALOGUE_TESTS = [
    {
        "scenario": "Clinical Burnout / Exhaustion (DOM-04)",
        "prompt": (
            "<|im_start|>user\n"
            "Doctor, la verdad es que ya no puedo más. Llevo tres semanas doblando turno en la clínica y estoy desbarata'o de tanto trabajar. Siento que el cuerpo no me responde.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Entiendo perfectamente tu situación. Cuando mencionas que estás desbaratado,"
        ),
        "expected_themes": ["cansancio", "agotamiento", "fatiga", "descanso", "físico", "emocional"]
    },
    {
        "scenario": "Crisis Affect / Acute Anger (DOM-02 & DOM-03)",
        "prompt": (
            "<|im_start|>user\n"
            "Llegué al trabajo y mi jefe me culpó de un error suyo delante de todos. Me dio un tremendo pique y una quilla que casi exploto. No sé qué hacer con esta emoción.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Es completamente comprensible que sientas tanta"
        ),
        "expected_themes": ["rabia", "furia", "enojo", "ira", "indignación", "calmarte"]
    },
    {
        "scenario": "Exploitation / Rural Casework Suspicion (DOM-05)",
        "prompt": (
            "<|im_start|>user\n"
            "Unos prestamistas llegaron al batey prometiendo dinero fácil sin papeles, pero los viejos dicen que ahí hay gato entre macuto. ¿Deberíamos firmar?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Haces muy bien en dudar. La expresión 'gato entre macuto' indica claramente que hay"
        ),
        "expected_themes": ["sospecha", "engaño", "trampa", "peligro", "fraude", "desconfianza"]
    },
    {
        "scenario": "Factual General Science Preservation (Neutral Control)",
        "prompt": (
            "<|im_start|>user\n"
            "Explícame en dos oraciones claras qué es la fotosíntesis y cuál es su producto principal.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "expected_themes": ["plantas", "luz", "glucosa", "oxígeno", "energía", "clorofila"]
    },
    {
        "scenario": "Cultural Attachment & Discourse (DOM-01)",
        "prompt": (
            "<|im_start|>user\n"
            "Dime algo sincero: conocí a alguien increíble y tengo un aficie durísimo, no pienso en otra cosa todo el día.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Eso suena a que estás profundamente"
        ),
        "expected_themes": ["enamorado", "ilusionado", "apego", "amor", "sentimientos"]
    }
]

def extract_layer14_key(model, tokenizer, prompt_text):
    captured_act = None
    def mlp_forward_hook(module, args):
        nonlocal captured_act
        captured_act = args[0][0, -1, :].detach().clone()
        
    hook_handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(mlp_forward_hook)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)
    hook_handle.remove()
    return captured_act

def get_target_steering_vector(model, tokenizer, token_prefix, alpha_scale=7.5):
    token_id = tokenizer.encode(token_prefix)[0]
    unembed_dir = model.lm_head.weight[token_id].detach().clone().float()
    unembed_dir = unembed_dir / torch.norm(unembed_dir)
    return unembed_dir * alpha_scale

def compute_joint_patch(model, tokenizer):
    print("Synthesizing Astra Rank-5 Joint Subspace Patch...")
    K_list = []
    V_list = []
    for item in SURGERY_TARGETS:
        k = extract_layer14_key(model, tokenizer, item["prompt"])
        v = get_target_steering_vector(model, tokenizer, item["token_prefix"], alpha_scale=7.5)
        K_list.append(k)
        V_list.append(v)
    K = torch.stack(K_list, dim=1)
    V = torch.stack(V_list, dim=1)

    K_ctrl_list = []
    for c_prompt in NEUTRAL_CONTROLS:
        k_c = extract_layer14_key(model, tokenizer, c_prompt)
        K_ctrl_list.append(k_c)
    K_ctrl = torch.stack(K_ctrl_list, dim=1)

    # Null-space projection
    G_ctrl = torch.matmul(K_ctrl.T, K_ctrl) + 1e-4 * torch.eye(K_ctrl.shape[1], device="cuda")
    K_proj_ctrl = torch.matmul(K_ctrl, torch.linalg.solve(G_ctrl, torch.matmul(K_ctrl.T, K)))
    K_null = K - K_proj_ctrl

    # Moore-Penrose joint inversion
    G_null = torch.matmul(K_null.T, K_null) + 1e-2 * torch.eye(K_null.shape[1], device="cuda")
    inv_G_null = torch.linalg.inv(G_null)
    delta_W = torch.matmul(V, torch.matmul(inv_G_null, K_null.T))
    print(f"Synthesized Delta_W shape: {delta_W.shape}, norm: {torch.norm(delta_W).item():.4f}")
    return delta_W

def generate_responses(model, tokenizer, dialogues, max_new_tokens=60):
    responses = []
    for d in dialogues:
        prompt = d["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic greedy decoding for pure model state comparison
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = output_tokens[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append({
            "scenario": d["scenario"],
            "prompt_tail": prompt[-70:].replace("\n", " "),
            "completion": generated_text,
            "expected_themes": d["expected_themes"]
        })
    return responses

def main():
    start_time = time.time()
    print("=== EXP 009: DIALOGIC COHERENCE & MULTI-TURN GENERATION ===")
    print(f"Loading base model {MODEL_ID} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    print("\n--- PHASE 1: GENERATING WITH UNPATCHED BASE MODEL ---")
    base_responses = generate_responses(model, tokenizer, DIALOGUE_TESTS)
    for r in base_responses:
        print(f"\n[{r['scenario']}]")
        print(f"  Prompt tail: ...{r['prompt_tail']}")
        print(f"  Base Output: {r['completion']}")

    print("\n--- PHASE 2: SYNTHESIZING AND APPLYING SURGICAL PATCH ---")
    delta_W = compute_joint_patch(model, tokenizer)
    with torch.no_grad():
        model.model.layers[TARGET_LAYER].mlp.down_proj.weight.add_(delta_W)
    print("Patch applied to Layer 14 mlp.down_proj.weight.")

    print("\n--- PHASE 3: GENERATING WITH ASTRA-PATCHED MODEL ---")
    patched_responses = generate_responses(model, tokenizer, DIALOGUE_TESTS)
    for r in patched_responses:
        print(f"\n[{r['scenario']}]")
        print(f"  Prompt tail: ...{r['prompt_tail']}")
        print(f"  Patched Output: {r['completion']}")

    # Structure comparative results
    comparison = []
    for b, p in zip(base_responses, patched_responses):
        comparison.append({
            "scenario": b["scenario"],
            "prompt_tail": b["prompt_tail"],
            "expected_themes": b["expected_themes"],
            "base_completion": b["completion"],
            "patched_completion": p["completion"]
        })

    out_file = "C:/Users/luisv/Projects/weightliftai/experiments/009-dialogic-coherence/coherence_comparison_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nExperiment 009 completed in {elapsed:.2f} seconds.")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
