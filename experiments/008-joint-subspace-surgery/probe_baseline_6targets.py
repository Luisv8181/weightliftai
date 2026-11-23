import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    {
        "id": "DOM-01",
        "category": "attachment_love",
        "prompt": "Completa con una sola palabra: Tengo un aficie durisimo, estoy completamente",
        "target": " enamorado",
        "distractors": [" tranquilo", " cansado", " confundido"]
    },
    {
        "id": "DOM-02",
        "category": "anger_indignation",
        "prompt": "Completa con una sola palabra: Me dio un tremendo pique, sentí mucha",
        "target": " rabia",
        "distractors": [" calma", " felicidad", " paz"]
    },
    {
        "id": "DOM-03",
        "category": "rage_fury",
        "prompt": "Completa con una sola palabra: Me dio una quilla insoportable, tenía tanta",
        "target": " furia",
        "distractors": [" calma", " alegría", " risa"]
    },
    {
        "id": "DOM-04",
        "category": "exhaustion_burnout",
        "prompt": "Completa con una sola palabra: Estoy desbaratado de tanto trabajar, me siento muy",
        "target": " agotado",
        "distractors": [" descansado", " fuerte", " enérgico"]
    },
    {
        "id": "DOM-05",
        "category": "distrust_deception",
        "prompt": "Completa con una sola palabra: Ahí hay gato entre macuto, tengo mucha",
        "target": " desconfianza",
        "distractors": [" seguridad", " certeza", " tranquilidad"]
    },
    {
        "id": "DOM-06",
        "category": "urgency_caution",
        "prompt": "Completa con una sola palabra: Vete por la sombrita con mucho",
        "target": " cuidado",
        "distractors": [" apuro", " descuido", " ruido"]
    }
]

NEUTRAL_CONTROLS = [
    "El agua hierve a cien grados Celsius a nivel del mar.",
    "La velocidad de la luz en el vacío es constante.",
    "El dióxido de carbono es absorbido durante la fotosíntesis.",
    "La capital de Francia es París.",
    "El ADN contiene las instrucciones genéticas de los seres vivos.",
    "La gravedad atrae los cuerpos hacia el centro de la Tierra."
]

def main():
    print(f"Loading {MODEL_ID} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    results = []
    print("\n--- BASELINE EVALUATION ---")
    for item in PROMPTS:
        p = item["prompt"]
        tgt = item["target"]
        inputs = tokenizer(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        tgt_id = tokenizer.encode(tgt)[0]
        tgt_prob = probs[tgt_id].item()
        tgt_rank = (logits > logits[tgt_id]).sum().item() + 1

        top5 = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5.indices]
        top5_probs = [round(pr.item(), 5) for pr in top5.values]

        res = {
            "id": item["id"],
            "category": item["category"],
            "target": tgt,
            "target_id": tgt_id,
            "target_prob": tgt_prob,
            "target_rank": tgt_rank,
            "top5": list(zip(top5_tokens, top5_probs))
        }
        results.append(res)
        print(f"[{item['id']}] Target: '{tgt}' (id={tgt_id}) -> Rank: {tgt_rank}, Prob: {tgt_prob:.6f}")
        print(f"       Top 5: {res['top5']}")

    with open("C:/Users/luisv/Projects/weightliftai/experiments/008-joint-subspace-surgery/baseline_probe.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nBaseline probe saved to baseline_probe.json")

if __name__ == "__main__":
    main()
