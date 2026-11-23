"""MATH-500 Benchmark Sweep and Weight Surgery Evaluation.

Implements replication of Takehi et al. (MIT CSAIL) Section 6:
- Formal evaluation of competition mathematical reasoning on HuggingFaceH4/MATH-500.
- Standardized LaTeX boxed answer extraction with nested-brace support.
- Numerical, symbolic, and canonical string equivalence verification.
- Degenerative cyclic repetition detection to monitor model collapse under weight surgery.
- Comparative evaluation harness assessing baseline Qwen2.5-0.5B-Instruct vs. weight-surgeried variants.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def extract_boxed_answer(text: str) -> str | None:
    """Extract content inside the final \\boxed{...} accounting for arbitrary nested braces."""
    if not text:
        return None

    # Search backwards for \boxed{
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        # Fallback regex heuristics for common terminal phrasing
        patterns = [
            r"(?:the\s+final\s+answer\s+is|final\s+answer\s*:|the\s+answer\s+is|answer\s*:)\s*([^\n\.]+)",
            r"\\boxed\s*([0-9a-zA-Z\+\-\*/\(\)\\]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
        return None

    start = idx + len(r"\boxed{")
    brace_depth = 1
    i = start
    while i < len(text) and brace_depth > 0:
        if text[i] == "{":
            brace_depth += 1
        elif text[i] == "}":
            brace_depth -= 1
        i += 1

    if brace_depth == 0:
        return text[start : i - 1].strip()
    return None


def normalize_math_expression(s: str | None) -> str:
    """Canonicalize LaTeX math expression for robust equivalence checking."""
    if s is None:
        return ""
    s = s.strip()
    # Strip enclosing math delimiters
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)

    # Strip \boxed{...} if present using nested brace extraction
    if r"\boxed{" in s:
        extracted = extract_boxed_answer(s)
        if extracted:
            s = extracted

    # Strip text/style macros
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|textbf|mbox)\{([^}]+)\}", r"\1", s)

    # Strip delimiter styling
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"^\circ", "").replace("°", "")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\pi", "pi")

    # Normalize fractions: \frac{a}{b} -> a/b if simple single token, else (a)/(b)
    def _frac_sub(m: re.Match[str]) -> str:
        num, den = m.group(1), m.group(2)
        # If num/den is a single letter or number, avoid extra parens
        num_clean = num if re.fullmatch(r"[0-9a-zA-Z\\]+", num) else f"({num})"
        den_clean = den if re.fullmatch(r"[0-9a-zA-Z\\]+", den) else f"({den})"
        return f"{num_clean}/{den_clean}"

    while r"\frac{" in s:
        new_s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _frac_sub, s)
        if new_s == s:
            break
        s = new_s

    # Strip spaces and commas in numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"\s+", "", s)

    # Strip leading positive sign
    s = s.lstrip("+")
    return s.strip()


def is_math_equivalent(pred: str | None, target: str | None) -> bool:
    """Check mathematical equivalence between candidate prediction and ground truth."""
    if pred is None or target is None:
        return False

    norm_p = normalize_math_expression(pred)
    norm_t = normalize_math_expression(target)

    # Exact canonical match
    if norm_p == norm_t:
        return True

    if not norm_p or not norm_t:
        return False

    # Numerical float evaluation
    try:
        clean_p = norm_p.replace("^", "**")
        clean_t = norm_t.replace("^", "**")
        val_p = float(eval(clean_p, {"__builtins__": None}, {}))
        val_t = float(eval(clean_t, {"__builtins__": None}, {}))
        if abs(val_p - val_t) < 1e-4:
            return True
    except Exception:
        pass

    # Tuple / coordinate equivalence: (a, b) vs \left( a, b \right)
    if (norm_p.startswith("(") and norm_p.endswith(")")) and (norm_t.startswith("(") and norm_t.endswith(")")):
        parts_p = norm_p[1:-1].split(",")
        parts_t = norm_t[1:-1].split(",")
        if len(parts_p) == len(parts_t):
            all_match = all(is_math_equivalent(p, t) for p, t in zip(parts_p, parts_t))
            if all_match:
                return True

    return False


def detect_cyclic_repetition(text: str, min_cycle_len: int = 15, min_repeats: int = 3) -> bool:
    """Detect if generation has devolved into degenerate cyclic repetition loops."""
    if not text or len(text) < min_cycle_len * min_repeats:
        return False

    # Check for exact n-gram periodic repetition
    for cycle_len in range(min_cycle_len, min(80, len(text) // min_repeats)):
        pattern = text[-cycle_len:]
        sub = pattern * min_repeats
        if sub in text:
            return True

    return False


@dataclasses.dataclass
class MathSampleEvaluation:
    """Record of a single problem evaluation."""

    unique_id: str
    subject: str
    level: int
    problem: str
    target_answer: str
    generated_text: str
    extracted_answer: str | None
    is_correct: bool
    is_cyclic: bool
    token_count: int
    elapsed_seconds: float


@dataclasses.dataclass
class MathBenchmarkResult:
    """Aggregated evaluation metrics across the MATH-500 sweep."""

    total_samples: int
    correct_count: int
    accuracy: float
    subject_accuracy: dict[str, float]
    level_accuracy: dict[int, float]
    cyclic_loop_rate: float
    mean_token_count: float
    total_elapsed_seconds: float
    samples: list[MathSampleEvaluation] = dataclasses.field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        """Serialize high-level summary metrics."""
        return {
            "total_samples": self.total_samples,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "subject_accuracy": {k: round(v, 4) for k, v in self.subject_accuracy.items()},
            "level_accuracy": {k: round(v, 4) for k, v in self.level_accuracy.items()},
            "cyclic_loop_rate": round(self.cyclic_loop_rate, 4),
            "mean_token_count": round(self.mean_token_count, 1),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 2),
        }


def build_math_prompt(problem: str) -> str:
    """Construct standardized chain-of-thought prompt with boxed answer instructions."""
    return (
        "Please solve the following math problem step by step. "
        "Put your final answer within \\boxed{}.\n\n"
        f"Problem:\n{problem}"
    )


def evaluate_model_on_math_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Sequence[dict[str, Any]],
    batch_size: int = 8,
    max_new_tokens: int = 384,
    device: str = "cuda",
) -> MathBenchmarkResult:
    """Run batched mathematical reasoning evaluation across provided dataset items."""
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation.")

    evaluations: list[MathSampleEvaluation] = []
    t_start = time.time()

    # Ensure left-padding for generation batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_samples = len(dataset)
    for batch_start in range(0, n_samples, batch_size):
        batch_items = dataset[batch_start : batch_start + batch_size]
        batch_prompts: list[str] = []

        for item in batch_items:
            raw_prompt = build_math_prompt(item["problem"])
            messages = [{"role": "user", "content": raw_prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(formatted)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_batch = time.time() - t0
        per_item_time = elapsed_batch / len(batch_items)

        for i, item in enumerate(batch_items):
            gen_ids = outputs[i][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            token_count = len(gen_ids)

            extracted = extract_boxed_answer(text)
            correct = is_math_equivalent(extracted, item["answer"])
            cyclic = detect_cyclic_repetition(text)

            sample_eval = MathSampleEvaluation(
                unique_id=item.get("unique_id", f"sample_{batch_start + i}"),
                subject=item.get("subject", "Unknown"),
                level=item.get("level", 0),
                problem=item["problem"],
                target_answer=item["answer"],
                generated_text=text,
                extracted_answer=extracted,
                is_correct=correct,
                is_cyclic=cyclic,
                token_count=token_count,
                elapsed_seconds=per_item_time,
            )
            evaluations.append(sample_eval)

    total_time = time.time() - t_start
    total_correct = sum(1 for e in evaluations if e.is_correct)
    accuracy = total_correct / len(evaluations) if evaluations else 0.0
    cyclic_count = sum(1 for e in evaluations if e.is_cyclic)
    cyclic_rate = cyclic_count / len(evaluations) if evaluations else 0.0
    mean_tokens = float(np.mean([e.token_count for e in evaluations])) if evaluations else 0.0

    # Subject breakdown
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    for e in evaluations:
        subject_total[e.subject] = subject_total.get(e.subject, 0) + 1
        if e.is_correct:
            subject_correct[e.subject] = subject_correct.get(e.subject, 0) + 1
    subject_acc = {s: subject_correct.get(s, 0) / count for s, count in subject_total.items()}

    # Level breakdown
    level_correct: dict[int, int] = {}
    level_total: dict[int, int] = {}
    for e in evaluations:
        level_total[e.level] = level_total.get(e.level, 0) + 1
        if e.is_correct:
            level_correct[e.level] = level_correct.get(e.level, 0) + 1
    level_acc = {lvl: level_correct.get(lvl, 0) / count for lvl, count in level_total.items()}

    return MathBenchmarkResult(
        total_samples=len(evaluations),
        correct_count=total_correct,
        accuracy=accuracy,
        subject_accuracy=subject_acc,
        level_accuracy=level_acc,
        cyclic_loop_rate=cyclic_rate,
        mean_token_count=mean_tokens,
        total_elapsed_seconds=total_time,
        samples=evaluations,
    )
