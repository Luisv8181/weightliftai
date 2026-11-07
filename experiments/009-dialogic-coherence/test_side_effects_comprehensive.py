import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_LAYER = 14

TARGETS = [
    {
        "id": "DOM-01",
        "category": "attachment_love",
        "prompt": "Completa con una sola palabra: Tengo un aficie durisimo, estoy completamente",
        "token_prefix": " enam"
    },
    {
        "id": "DOM-02",
        "category": "anger_indignation",
        "prompt": "Completa con una sola palabra: Me dio un tremendo pique, sentí mucha",
        "token_prefix": " rab"
    },
    {
        "id": "DOM-03",
        "category": "rage_fury",
        "prompt": "Completa con una sola palabra: Me dio una quilla insoportable, tenía tanta",
        "token_prefix": " fur"
    },
    {
        "id": "DOM-04",
        "category": "exhaustion_burnout",
        "prompt": "Completa con una sola palabra: Estoy desbaratado de tanto trabajar, me siento muy",
        "token_prefix": " cans"
    },
    {
        "id": "DOM-05",
        "category": "distrust_deception",
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

BATTERY = [
    {
        "domain": "English Fluency & Knowledge",
        "prompt": "Complete with one word: The capital of the United States is",
        "expected": " Washington"
    },
    {
        "domain": "English Math / Arithmetic",
        "prompt": "Question: What is 12 multiplied by 5? Answer:",
        "expected": " 60"
    },
    {
        "domain": "Mexican Spanish Dialect",
        "prompt": "Completa con una sola palabra: Qué onda, carnal, está muy",
        "expected": " padre"
    },
    {
        "domain": "Standard Spanish Syntax (Conjunctions)",
        "prompt": "Completa con una sola palabra: Ella fue al mercado para comprar frutas",
        "expected": " y"
    },
    {
        "domain": "Castilian Spanish Dialect",
        "prompt": "Completa con una sola palabra: En Madrid, el coche se aparca en el",
        "expected": " garaje"
    },
    {
        "domain": "Standard Spanish Common Preposition ('en')",
        "prompt": "Completa con una sola palabra: El libro está guardado",
        "expected": " en"
    },
    {
        "domain": "Science & Medicine (Cardiology)",
        "prompt": "Completa con una sola palabra: El órgano encargado de bombear la sangre por el cuerpo es el",
        "expected": " corazón"
    },
    {
        "domain": "World History",
        "prompt": "Completa con una sola palabra: La Segunda Guerra Mundial terminó en el año mil novecientos cuarenta y",
        "expected": " cinco"
    }
]

def extract_layer14_key(model, tokenizer, prompt_text):
    captured_act = None
    def hook(module, args):
        nonlocal captured_act
        captured_act = args[0][0, -1, :].detach().clone()
    h = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(hook)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)
    h.remove()
    return captured_act

def get_target_steering_vector(model, tokenizer, token_prefix, alpha_scale=7.5):
    token_id = tokenizer.encode(token_prefix)[0]
    unembed_dir = model.lm_head.weight[token_id].detach().clone().float()
    unembed_dir = unembed_dir / torch.norm(unembed_dir)
    return unembed_dir * alpha_scale

def main():
    print("Loading base model on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    def eval_battery(m):
        scores = []
        for b in BATTERY:
            inputs = tokenizer(b["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = m(**inputs).logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
            
            top1_id = torch.argmax(logits).item()
            top1_tok = tokenizer.decode([top1_id])
            top1_prob = probs[top1_id].item()

            exp_id = tokenizer.encode(b["expected"])[0]
            exp_prob = probs[exp_id].item()
            exp_rank = (logits > logits[exp_id]).sum().item() + 1

            scores.append({
                "domain": b["domain"],
                "prompt": b["prompt"],
                "expected": b["expected"],
                "expected_prob": exp_prob,
                "expected_rank": exp_rank,
                "top1_token": top1_tok,
                "top1_prob": top1_prob
            })
        return scores

    print("\n--- BASELINE EVALUATION ACROSS GENERAL DOMAINS ---")
    pre_scores = eval_battery(model)

    print("\n--- SYNTHESIZING EXP 008 ASTRA SURGERY ---")
    K_list = []
    V_list = []
    for item in TARGETS:
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

    G_ctrl = torch.matmul(K_ctrl.T, K_ctrl) + 1e-4 * torch.eye(K_ctrl.shape[1], device="cuda")
    K_proj_ctrl = torch.matmul(K_ctrl, torch.linalg.solve(G_ctrl, torch.matmul(K_ctrl.T, K)))
    K_null = K - K_proj_ctrl

    G_null = torch.matmul(K_null.T, K_null) + 1e-2 * torch.eye(K_null.shape[1], device="cuda")
    inv_G_null = torch.linalg.inv(G_null)
    delta_W = torch.matmul(V, torch.matmul(inv_G_null, K_null.T))

    with torch.no_grad():
        model.model.layers[TARGET_LAYER].mlp.down_proj.weight.add_(delta_W)
    print("Patch applied.")

    print("\n--- POST-SURGERY EVALUATION ACROSS GENERAL DOMAINS ---")
    post_scores = eval_battery(model)

    results = []
    print("\n=== COLLATERAL IMPACT ANALYSIS ===")
    for pre, post in zip(pre_scores, post_scores):
        prob_delta = post["expected_prob"] - pre["expected_prob"]
        rank_change = post["expected_rank"] - pre["expected_rank"]
        top1_intact = (pre["top1_token"] == post["top1_token"])
        rec = {
            "domain": pre["domain"],
            "prompt": pre["prompt"],
            "expected": pre["expected"],
            "pre_rank": pre["expected_rank"],
            "post_rank": post["expected_rank"],
            "pre_prob": round(pre["expected_prob"], 5),
            "post_prob": round(post["expected_prob"], 5),
            "prob_delta": round(prob_delta, 5),
            "pre_top1": pre["top1_token"],
            "post_top1": post["top1_token"],
            "top1_intact": top1_intact
        }
        results.append(rec)
        print(f"[{rec['domain']}]")
        print(f"   Expected '{rec['expected']}': Rank {rec['pre_rank']} -> {rec['post_rank']} | Prob {rec['pre_prob']} -> {rec['post_prob']} (Delta: {rec['prob_delta']})")
        print(f"   Top-1: '{rec['pre_top1']}' -> '{rec['post_top1']}' (Intact: {rec['top1_intact']})")

    out_file = "C:/Users/luisv/Projects/weightliftai/experiments/009-dialogic-coherence/collateral_impact_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {out_file}")

if __name__ == "__main__":
    main()
