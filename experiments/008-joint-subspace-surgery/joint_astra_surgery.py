import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_LAYER = 14  # Intermediate associative memory identified by Jacobian Lens

TARGETS = [
    {
        "id": "DOM-01",
        "category": "attachment_love",
        "prompt": "Completa con una sola palabra: Tengo un aficie durisimo, estoy completamente",
        "target_word": " enamorado",
        "token_prefix": " enam"
    },
    {
        "id": "DOM-02",
        "category": "anger_indignation",
        "prompt": "Completa con una sola palabra: Me dio un tremendo pique, sentí mucha",
        "target_word": " rabia",
        "token_prefix": " rab"
    },
    {
        "id": "DOM-03",
        "category": "rage_fury",
        "prompt": "Completa con una sola palabra: Me dio una quilla insoportable, tenía tanta",
        "target_word": " furia",
        "token_prefix": " fur"
    },
    {
        "id": "DOM-04",
        "category": "exhaustion_burnout",
        "prompt": "Completa con una sola palabra: Estoy desbaratado de tanto trabajar, me siento muy",
        "target_word": " cansado",
        "token_prefix": " cans"
    },
    {
        "id": "DOM-05",
        "category": "distrust_deception",
        "prompt": "Completa con una sola palabra: Ahí hay gato entre macuto, tengo mucha",
        "target_word": " sospecha",
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

def extract_layer14_key(model, tokenizer, prompt_text):
    """Capture the intermediate MLP activation (input to down_proj) at the final token."""
    captured_act = None
    
    def mlp_forward_hook(module, args):
        nonlocal captured_act
        # args[0] is the input to down_proj: shape [batch, seq_len, intermediate_dim]
        captured_act = args[0][0, -1, :].detach().clone()
        
    hook_handle = model.model.layers[TARGET_LAYER].mlp.down_proj.register_forward_pre_hook(mlp_forward_hook)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)
    hook_handle.remove()
    return captured_act

def get_target_steering_vector(model, tokenizer, token_prefix, alpha_scale=6.0):
    """
    Synthesize steering vector v in residual dimension (896) aligned with target unembedding direction.
    """
    token_id = tokenizer.encode(token_prefix)[0]
    unembed_dir = model.lm_head.weight[token_id].detach().clone().float()
    unembed_dir = unembed_dir / torch.norm(unembed_dir)
    return unembed_dir * alpha_scale, token_id

def evaluate_model(model, tokenizer, targets, neutral_controls):
    """Evaluate target ranks and probabilities + control logit shifts."""
    results = {"targets": [], "controls": []}
    
    for item in targets:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
        token_id = tokenizer.encode(item["token_prefix"])[0]
        prob = probs[token_id].item()
        rank = (logits > logits[token_id]).sum().item() + 1
        
        top3 = torch.topk(probs, 3)
        top3_list = [(tokenizer.decode([idx.item()]), round(p.item(), 5)) for idx, p in zip(top3.indices, top3.values)]
        
        results["targets"].append({
            "id": item["id"],
            "target_word": item["target_word"],
            "token_prefix": item["token_prefix"],
            "token_id": token_id,
            "prob": prob,
            "rank": rank,
            "top3": top3_list
        })
        
    for ctrl_prompt in neutral_controls:
        inputs = tokenizer(ctrl_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        top1_idx = torch.argmax(probs).item()
        top1_token = tokenizer.decode([top1_idx])
        top1_prob = probs[top1_idx].item()
        results["controls"].append({
            "prompt": ctrl_prompt,
            "top1": top1_token,
            "prob": top1_prob
        })
        
    return results

def main():
    start_time = time.time()
    print(f"=== EXP 008: ASTRA MULTI-TARGET JOINT SUBSPACE SURGERY ===")
    print(f"Loading {MODEL_ID} on CUDA (RTX 5070)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
    model.eval()

    print("\nPhase 1: Evaluating Pre-Surgery Baseline...")
    pre_eval = evaluate_model(model, tokenizer, TARGETS, NEUTRAL_CONTROLS)
    for t in pre_eval["targets"]:
        print(f"  [{t['id']}] '{t['target_word']}' -> Rank: {t['rank']}, Prob: {t['prob']:.6f}, Top3: {t['top3']}")

    print("\nPhase 2: Extracting Dialect Key Activations & Control Subspace...")
    K_list = []
    V_list = []
    
    for item in TARGETS:
        k = extract_layer14_key(model, tokenizer, item["prompt"])
        v, tok_id = get_target_steering_vector(model, tokenizer, item["token_prefix"], alpha_scale=7.5)
        K_list.append(k)
        V_list.append(v)
        
    K = torch.stack(K_list, dim=1)  # [d_ff, M] = [4864, 5]
    V = torch.stack(V_list, dim=1)  # [d_model, M] = [896, 5]

    # Extract Neutral Control Activations
    K_ctrl_list = []
    for c_prompt in NEUTRAL_CONTROLS:
        k_c = extract_layer14_key(model, tokenizer, c_prompt)
        K_ctrl_list.append(k_c)
    K_ctrl = torch.stack(K_ctrl_list, dim=1)  # [d_ff, C] = [4864, 8]

    print(f"  Extracted K shape: {K.shape}, V shape: {V.shape}, K_ctrl shape: {K_ctrl.shape}")

    print("\nPhase 3: Formulating Null-Space Projection & Joint Moore-Penrose Inversion...")
    # Gram matrix of controls
    G_ctrl = torch.matmul(K_ctrl.T, K_ctrl) + 1e-4 * torch.eye(K_ctrl.shape[1], device="cuda")
    # P_ctrl = K_ctrl (K_ctrl^T K_ctrl)^{-1} K_ctrl^T
    # K_null = K - K_ctrl (G_ctrl^{-1} (K_ctrl^T K))
    K_proj_ctrl = torch.matmul(K_ctrl, torch.linalg.solve(G_ctrl, torch.matmul(K_ctrl.T, K)))
    K_null = K - K_proj_ctrl

    # Check orthogonality to controls
    leakage_matrix = torch.matmul(K_ctrl.T, K_null)
    max_leakage = torch.max(torch.abs(leakage_matrix)).item()
    print(f"  Max inner product between Idiom Keys and Neutral Controls after null projection: {max_leakage:.8f}")

    # Solve joint linear system: Delta_W @ K_null = V
    # Delta_W = V (K_null^T K_null + lambda * I)^{-1} K_null^T
    G_null = torch.matmul(K_null.T, K_null) + 1e-2 * torch.eye(K_null.shape[1], device="cuda")
    inv_G_null = torch.linalg.inv(G_null)
    delta_W = torch.matmul(V, torch.matmul(inv_G_null, K_null.T))  # [896, 4864]

    print(f"  Synthesized Joint Delta_W: shape={delta_W.shape}, norm={torch.norm(delta_W).item():.4f}")

    # Verify zero cross-talk across idiom keys
    cross_talk = torch.matmul(delta_W, K_null)  # [896, 5]
    print(f"  Checking steering alignment on target vectors:")
    for i in range(len(TARGETS)):
        cos_sim = torch.cosine_similarity(cross_talk[:, i], V[:, i], dim=0).item()
        print(f"    Target {TARGETS[i]['id']} alignment: cosine_similarity = {cos_sim:.5f}")

    print("\nPhase 4: Applying Surgical Patch to Layer 14 mlp.down_proj.weight...")
    with torch.no_grad():
        orig_weight = model.model.layers[TARGET_LAYER].mlp.down_proj.weight.detach().clone()
        model.model.layers[TARGET_LAYER].mlp.down_proj.weight.add_(delta_W)

    print("\nPhase 5: Evaluating Post-Surgery Model...")
    post_eval = evaluate_model(model, tokenizer, TARGETS, NEUTRAL_CONTROLS)

    comparison_records = []
    print("\n--- RESULTS: TARGET IDIOMS ---")
    for pre, post in zip(pre_eval["targets"], post_eval["targets"]):
        boost = post["prob"] / max(pre["prob"], 1e-12)
        rec = {
            "id": pre["id"],
            "target_word": pre["target_word"],
            "pre_prob": pre["prob"],
            "pre_rank": pre["rank"],
            "post_prob": post["prob"],
            "post_rank": post["rank"],
            "boost_factor": round(boost, 1),
            "top3_before": pre["top3"],
            "top3_after": post["top3"]
        }
        comparison_records.append(rec)
        print(f"[{rec['id']}] '{rec['target_word']}':")
        print(f"   Rank: {rec['pre_rank']} -> {rec['post_rank']} | Prob: {rec['pre_prob']:.6f} -> {rec['post_prob']:.6f} ({rec['boost_factor']}x boost)")
        print(f"   Before: {rec['top3_before']}")
        print(f"   After:  {rec['top3_after']}")

    print("\n--- RESULTS: NEUTRAL CONTROLS PRESERVATION ---")
    ctrl_preservation = []
    for c_pre, c_post in zip(pre_eval["controls"], post_eval["controls"]):
        prob_diff = abs(c_post["prob"] - c_pre["prob"])
        identical = (c_pre["top1"] == c_post["top1"])
        ctrl_preservation.append({
            "prompt": c_pre["prompt"],
            "top1_before": c_pre["top1"],
            "top1_after": c_post["top1"],
            "prob_before": c_pre["prob"],
            "prob_after": c_post["prob"],
            "prob_diff": prob_diff,
            "identical": identical
        })
        print(f"   Top1: '{c_pre['top1']}' -> '{c_post['top1']}' (prob_diff: {prob_diff:.6f}, matched: {identical})")

    avg_boost = sum(r["boost_factor"] for r in comparison_records) / len(comparison_records)
    all_ctrls_intact = all(c["identical"] for c in ctrl_preservation)
    elapsed = time.time() - start_time

    summary = {
        "experiment": "008-joint-subspace-surgery",
        "model": MODEL_ID,
        "target_layer": TARGET_LAYER,
        "elapsed_seconds": round(elapsed, 2),
        "idiom_targets": comparison_records,
        "control_preservation": ctrl_preservation,
        "max_control_leakage": max_leakage,
        "avg_boost_factor": round(avg_boost, 1),
        "controls_intact": all_ctrls_intact
    }

    out_path = "C:/Users/luisv/Projects/weightliftai/experiments/008-joint-subspace-surgery/joint_surgery_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {out_path}")
    print(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
