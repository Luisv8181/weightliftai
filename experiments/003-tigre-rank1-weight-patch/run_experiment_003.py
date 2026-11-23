"""Experiment 003: rank-1 weight patch vs LoRA vs random control for Dominican 'tíguere'.

See README.md for pre-registered criteria. Writes results/ (JSON + markdown).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from data import REGRESSION, TEST_ANIMAL, TEST_SLANG, TRAIN_ANIMAL, TRAIN_SLANG  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [0, 1, 2]
STEPS = 80
CANDIDATE_LAYERS = [6, 12, 18]
NORM_BUDGET_FRAC = 0.05
OUT = Path(__file__).parent / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_ID)


def prompt(es: str) -> str:
    msgs = [{"role": "system", "content": "You translate Dominican Spanish into natural English."},
            {"role": "user", "content": f"Translate to English: {es}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(es: str, en: str):
    p = tok(prompt(es), add_special_tokens=False).input_ids
    c = tok(en, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device=DEVICE)
    labels = torch.tensor([[-100] * len(p) + c], device=DEVICE)
    return ids, labels


def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
    m.eval()
    return m


def seq_logprob(model, es, en) -> tuple[float, int]:
    ids, labels = encode(es, en)
    with torch.no_grad():
        logits = model(ids).logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return float(lp[mask].sum()), int(mask.sum())


def margins(model, items):
    return [seq_logprob(model, es, good)[0] - seq_logprob(model, es, bad)[0] for es, good, bad in items]


def regression_nll(model):
    tot, n = 0.0, 0
    for es, en in REGRESSION:
        lp, k = seq_logprob(model, es, en)
        tot += -lp
        n += k
    return tot / n


def greedy(model, es):
    ids = tok(prompt(es), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


TRAIN = [(es, good) for es, good, _ in TRAIN_SLANG + TRAIN_ANIMAL]


def train_loss(model) -> torch.Tensor:
    losses = []
    for es, en in TRAIN:
        ids, labels = encode(es, en)
        losses.append(model(ids, labels=labels).loss)
    return torch.stack(losses).mean()


# ---------------- rank-1 patch ----------------
class Rank1Patch:
    def __init__(self, model, layer: int, seed: int, budget: float):
        g = torch.Generator().manual_seed(seed)
        self.lin = model.model.layers[layer].mlp.down_proj
        d_out, d_in = self.lin.weight.shape
        self.u = torch.zeros(d_out, device=DEVICE, requires_grad=True)
        self.v = (torch.randn(d_in, generator=g) / d_in ** 0.5).to(DEVICE).requires_grad_(True)
        self.budget = budget
        self.handle = self.lin.register_forward_hook(self._hook)

    def _hook(self, _mod, inp, out):
        return out + (inp[0] @ self.v).unsqueeze(-1) * self.u

    def project(self):
        with torch.no_grad():
            norm = self.u.norm() * self.v.norm()
            if norm > self.budget:
                s = (self.budget / norm).sqrt()
                self.u.mul_(s)
                self.v.mul_(s)

    def frob(self) -> float:
        return float((self.u.detach().norm() * self.v.detach().norm()).item())

    def fold(self):
        self.handle.remove()
        with torch.no_grad():
            self.lin.weight.add_(torch.outer(self.u, self.v))


def run_rank1(seed: int, layer: int):
    torch.manual_seed(seed)
    model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    w_norm = float(model.model.layers[layer].mlp.down_proj.weight.norm())
    patch = Rank1Patch(model, layer, seed, NORM_BUDGET_FRAC * w_norm)
    opt = torch.optim.AdamW([patch.u, patch.v], lr=5e-3, weight_decay=0.0)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = train_loss(model)
        loss.backward()
        opt.step()
        patch.project()
    with torch.no_grad():
        final_loss = float(train_loss(model))
    # fold check
    ids, _ = encode(*TEST_SLANG[0][:2])
    with torch.no_grad():
        hooked = model(ids).logits
    patch.fold()
    with torch.no_grad():
        folded = model(ids).logits
    fold_err = float((hooked - folded).abs().max())
    return model, {"layer": layer, "train_loss": final_loss, "patch_frob": patch.frob(),
                   "w_frob": w_norm, "budget": NORM_BUDGET_FRAC * w_norm, "fold_max_abs_err": fold_err,
                   "u": patch.u.detach().cpu(), "v": patch.v.detach().cpu()}


def run_random(seed: int, layer: int, frob: float):
    model = load_model()
    g = torch.Generator().manual_seed(1000 + seed)
    lin = model.model.layers[layer].mlp.down_proj
    u = torch.randn(lin.weight.shape[0], generator=g)
    v = torch.randn(lin.weight.shape[1], generator=g)
    s = (frob / (u.norm() * v.norm())).sqrt()
    with torch.no_grad():
        lin.weight.add_(torch.outer(u * s, v * s).to(DEVICE))
    return model


def run_lora(seed: int):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model = load_model()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"]))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    model.train()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = train_loss(model)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        final_loss = float(train_loss(model))
    return model.merge_and_unload(), {"train_loss": final_loss, "trainable_params": n_params}


def evaluate(model):
    ms = margins(model, TEST_SLANG)
    ma = margins(model, TEST_ANIMAL)
    return {"slang_margins": ms, "animal_margins": ma, "regression_nll": regression_nll(model)}


def bootstrap_ci(per_item: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_item), (n, len(per_item)))
    means = per_item[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(evals: list[dict], base_nll: float):
    ms = np.mean([e["slang_margins"] for e in evals], 0)
    ma = np.mean([e["animal_margins"] for e in evals], 0)
    tigre_idx = [i for i, (es, *_ ) in enumerate(TEST_SLANG) if "tigre" in es.lower() and "tígu" not in es.lower()]
    slang_wins = [np.mean(np.array(e["slang_margins"]) > 0) for e in evals]
    animal_wins = [np.mean(np.array(e["animal_margins"]) > 0) for e in evals]
    nll = [e["regression_nll"] for e in evals]
    return {
        "n_seeds": len(evals),
        "slang_win_rate_mean": float(np.mean(slang_wins)), "slang_win_rate_per_seed": [float(x) for x in slang_wins],
        "slang_margin_mean": float(ms.mean()), "slang_margin_ci95": bootstrap_ci(ms),
        "slang_tigre_spelling_win_rate": float(np.mean([np.mean(np.array(e["slang_margins"])[tigre_idx] > 0) for e in evals])),
        "animal_win_rate_mean": float(np.mean(animal_wins)), "animal_win_rate_per_seed": [float(x) for x in animal_wins],
        "animal_margin_mean": float(ma.mean()), "animal_margin_ci95": bootstrap_ci(ma),
        "regression_nll_mean": float(np.mean(nll)),
        "regression_nll_rel_change": float(np.mean(nll) / base_nll - 1.0),
    }


GREEDY_ITEMS = [TEST_SLANG[0][0], TEST_SLANG[3][0], TEST_SLANG[8][0], TEST_ANIMAL[0][0], TEST_ANIMAL[4][0]]


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    results: dict = {"model": MODEL_ID, "seeds": SEEDS, "steps": STEPS,
                     "date_utc": datetime.now(timezone.utc).isoformat(),
                     "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                     "transformers": transformers.__version__, "device": DEVICE,
                                     "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}}
    greedy_log = {}

    base = load_model()
    base_eval = evaluate(base)
    base_nll = base_eval["regression_nll"]
    results["baseline"] = summarize([base_eval], base_nll)
    greedy_log["baseline"] = {es: greedy(base, es) for es in GREEDY_ITEMS}
    del base

    # layer selection on training loss only (seed 0)
    layer_losses = {}
    for L in CANDIDATE_LAYERS:
        m, info = run_rank1(0, L)
        layer_losses[L] = info["train_loss"]
        del m
        torch.cuda.empty_cache()
    best_layer = min(layer_losses, key=layer_losses.get)
    results["layer_selection_train_loss"] = {str(k): v for k, v in layer_losses.items()}
    results["selected_layer"] = best_layer
    print("layer train losses:", layer_losses, "-> layer", best_layer, flush=True)

    patch_evals, rand_evals, lora_evals, patch_infos, lora_infos = [], [], [], [], []
    for s in SEEDS:
        m, info = run_rank1(s, best_layer)
        patch_evals.append(evaluate(m))
        if s == 0:
            greedy_log["rank1_patch"] = {es: greedy(m, es) for es in GREEDY_ITEMS}
            torch.save({"u": info["u"], "v": info["v"], "layer": best_layer}, OUT / "rank1_patch_seed0.pt")
        patch_infos.append({k: v for k, v in info.items() if k not in ("u", "v")})
        frob = info["patch_frob"]
        del m
        torch.cuda.empty_cache()

        m = run_random(s, best_layer, frob)
        rand_evals.append(evaluate(m))
        if s == 0:
            greedy_log["random_rank1"] = {es: greedy(m, es) for es in GREEDY_ITEMS}
        del m
        torch.cuda.empty_cache()

        m, linfo = run_lora(s)
        lora_evals.append(evaluate(m))
        lora_infos.append(linfo)
        if s == 0:
            greedy_log["lora_r4"] = {es: greedy(m, es) for es in GREEDY_ITEMS}
        del m
        torch.cuda.empty_cache()
        print(f"seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    results["rank1_patch"] = summarize(patch_evals, base_nll) | {"runs": patch_infos,
                                                                 "trainable_params": None}
    d_out, d_in = 896, 4864
    results["rank1_patch"]["trainable_params"] = d_out + d_in
    results["random_rank1"] = summarize(rand_evals, base_nll)
    results["lora_r4"] = summarize(lora_evals, base_nll) | {"runs": lora_infos}
    results["greedy_samples"] = greedy_log
    results["per_item_raw"] = {"baseline": base_eval, "rank1_patch": patch_evals,
                               "random_rank1": rand_evals, "lora_r4": lora_evals}

    p = results["rank1_patch"]
    criteria = {
        "slang_win_rate_ge_0.80": p["slang_win_rate_mean"] >= 0.80,
        "animal_win_rate_ge_0.875": p["animal_win_rate_mean"] >= 0.875,
        "regression_nll_increase_le_5pct": p["regression_nll_rel_change"] <= 0.05,
        "slang_ci_excludes_random_mean": p["slang_margin_ci95"][0] > results["random_rank1"]["slang_margin_mean"],
    }
    results["preregistered_criteria_rank1_patch"] = criteria
    results["all_criteria_pass"] = all(criteria.values())
    results["runtime_seconds"] = round(time.time() - t0, 1)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("baseline", "rank1_patch", "random_rank1", "lora_r4")},
                     indent=2, default=lambda o: None, ensure_ascii=False)[:6000])
    print("criteria:", criteria, "ALL PASS:", results["all_criteria_pass"])
    print(json.dumps(greedy_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
