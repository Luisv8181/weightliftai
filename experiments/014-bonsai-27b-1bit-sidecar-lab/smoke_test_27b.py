"""Smoke test: load Bonsai-27B-Q1_0.gguf directly into PyTorch CUDA (~3.8 GB VRAM) and verify logits."""

from __future__ import annotations

import os
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer

from weightliftai.weight_programming.gguf_1bit_loader import load_bonsai_27b_q1_0_gguf

GGUF_PATH = Path(os.environ.get("BONSAI_27B_GGUF", "C:/Users/luisv/models/bonsai/model/Bonsai-27B-Q1_0.gguf"))
TOKENIZER_ID = os.environ.get("BONSAI_27B_TOKENIZER", "HAL-X-AI/THX-02")


def main() -> None:
    t0 = time.perf_counter()
    print(f"Loading {GGUF_PATH.name} directly into CUDA 1-bit packed...")
    model, stats = load_bonsai_27b_q1_0_gguf(GGUF_PATH, device="cuda", scale_dtype=torch.bfloat16)
    dt = time.perf_counter() - t0
    print(f"Loaded in {dt:.2f}s | stats={stats}")

    cache_dir = "D:/cache/huggingface/hub" if Path("D:/cache/huggingface/hub").exists() else None
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, cache_dir=cache_dir, local_files_only=True)
    print(f"Tokenizer {TOKENIZER_ID}: vocab_size={len(tok)}")

    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Translate the following Spanish sentence into natural English. Output only the English translation.\nSpanish: El tigre corrió por la selva.\nEnglish:",
    ]
    for prompt in prompts:
        torch.cuda.reset_peak_memory_stats()
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs)
        fwd_s = time.perf_counter() - t1
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_id = torch.topk(probs, 5)
        print(
            f"\nPrompt ({inputs['input_ids'].shape[1]} tok, {fwd_s:.2f}s, peak_vram={torch.cuda.max_memory_allocated()/1e6:.1f} MB): {prompt!r}"
        )
        for p, tid in zip(top_p.tolist(), top_id.tolist()):
            print(f"  {tid:7d} ({p*100:5.2f}%): {tok.decode([tid])!r}")


if __name__ == "__main__":
    main()
