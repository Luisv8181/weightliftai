"""Real LLM-Driven Weight Repair Proposer, Structured Schema, and Baselines (Prompts 7–10).

Implements:
- Prompt 7: Real LLM proposer (`LLMWeightRepairProposer`), strict JSON output schema
  (`StructuredPatchProposal`), index-to-tensor delta conversion, pre-submission budget
  validation, and full provenance recording (model ID, revision, prompt version, decoding
  config, token/char counts, diagnostic condition, and patch SHA-256).
- Prompt 8: Condition-scoped prompt construction across all 5 `DiagnosticCondition` levels
  with strict verification that prohibited fields are absent.
- Prompt 9: Deterministic non-LLM proposer (`DeterministicHeuristicProposer`) and
  empirical gradient proposer (`GradientRepairProposer`) sharing the exact same
  evaluator-authoritative submission pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Literal
import urllib.request

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch

from weightliftai.weight_programming.defect_families import DiagnosticCondition
from weightliftai.weight_programming.inspector import validate_patch_against_budget
from weightliftai.weight_programming.proposer_api import BlindProposerEnvironment
from weightliftai.weight_programming.schemas import (
    AllowlistViolationError,
    BaseModelMismatchError,
    BudgetViolationError,
    WeightPatch,
)


PROMPT_TEMPLATE_VERSION = "wsp-llm-proposer-v1.0"


class IndexDeltaEdit(BaseModel):
    """Single element coordinate delta inside a 2D parameter tensor."""

    model_config = ConfigDict(extra="forbid")

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    delta: float


class StructuredPatchProposal(BaseModel):
    """Strict machine-readable repair proposal schema produced by the LLM (Prompt 7)."""

    model_config = ConfigDict(extra="forbid")

    target_tensor: str = Field(..., min_length=1, max_length=128)
    row_index: int = Field(default=0, ge=0)
    col_deltas: list[float] | None = None
    index_edits: list[IndexDeltaEdit] | None = None
    hypothesis: str = Field(..., min_length=3, max_length=512)
    rationale: str = Field(..., min_length=3, max_length=1024)

    @field_validator("col_deltas")
    @classmethod
    def _check_finite_deltas(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            for x in v:
                if not np.isfinite(float(x)):
                    raise ValueError("col_deltas must contain only finite numbers.")
        return v


def parse_and_validate_llm_proposal(
    raw_llm_text: str,
    env: BlindProposerEnvironment,
    proposed_by: str,
    patch_id: str,
) -> WeightPatch:
    """Parse raw LLM JSON output into a `StructuredPatchProposal` and validate against `env.budget`.

    Never executes arbitrary code (`eval`/`exec`) and never substitutes a fallback patch on error.
    """
    if not isinstance(raw_llm_text, str) or not raw_llm_text.strip():
        raise ValueError("LLM returned empty response text.")

    # Extract JSON object from fenced block or first balanced {...}
    text = raw_llm_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        json_candidate = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM output does not contain a JSON object.")
        json_candidate = text[start : end + 1]

    try:
        parsed_dict = json.loads(json_candidate)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in LLM proposal: {exc}") from exc

    proposal = StructuredPatchProposal.model_validate(parsed_dict)

    arch = env.get_architecture_metadata()
    params_meta = arch["parameters"]
    if proposal.target_tensor not in params_meta:
        raise AllowlistViolationError(
            f"Proposed target_tensor '{proposal.target_tensor}' does not exist in model parameters {list(params_meta)}."
        )

    allowed = env.budget.allowed_tensor_names
    if allowed is not None and proposal.target_tensor not in allowed:
        raise AllowlistViolationError(
            f"Proposed target_tensor '{proposal.target_tensor}' is not in budget allowlist {list(allowed)}."
        )

    shape = tuple(int(d) for d in params_meta[proposal.target_tensor]["shape"])
    if len(shape) != 2:
        raise ValueError(f"Expected 2D parameter shape, got {shape}.")

    n_rows, n_cols = shape
    delta_arr = np.zeros(shape, dtype=np.float32)

    if proposal.col_deltas is not None:
        if proposal.row_index < 0 or proposal.row_index >= n_rows:
            raise ValueError(f"row_index {proposal.row_index} out of bounds for {n_rows} rows.")
        if len(proposal.col_deltas) != n_cols:
            raise ValueError(
                f"col_deltas length ({len(proposal.col_deltas)}) does not match tensor width ({n_cols})."
            )
        delta_arr[proposal.row_index, :] = np.array(proposal.col_deltas, dtype=np.float32)
    elif proposal.index_edits is not None and len(proposal.index_edits) > 0:
        for edit in proposal.index_edits:
            if edit.row < 0 or edit.row >= n_rows or edit.col < 0 or edit.col >= n_cols:
                raise ValueError(
                    f"Index edit ({edit.row}, {edit.col}) out of bounds for shape {shape}."
                )
            if not np.isfinite(edit.delta):
                raise BudgetViolationError("Index edit contains non-finite delta.")
            delta_arr[edit.row, edit.col] = float(edit.delta)
    else:
        raise ValueError("StructuredPatchProposal must specify either 'col_deltas' or non-empty 'index_edits'.")

    if not np.isfinite(delta_arr).all():
        raise BudgetViolationError("Constructed delta tensor contains non-finite values.")

    if float(np.linalg.norm(delta_arr)) == 0.0:
        raise BudgetViolationError("LLM proposed an all-zero delta tensor.")

    total_l2 = float(np.linalg.norm(delta_arr))
    if total_l2 > env.budget.max_l2_norm:
        raise BudgetViolationError(
            f"Proposed delta L2 norm {total_l2:.4f} exceeds budget max_l2_norm {env.budget.max_l2_norm:.4f}."
        )

    patch = WeightPatch(
        patch_id=patch_id,
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={proposal.target_tensor: delta_arr},
        hypothesis=proposal.hypothesis,
        proposed_by=proposed_by,
        rationale=proposal.rationale,
    )

    # If corrupted_model is available in this diagnostic condition, run full validate_patch_against_budget
    if env._corrupted_model is not None:
        validate_patch_against_budget(env.get_corrupted_model(), patch, env.budget)

    return patch


def build_diagnostic_prompt_for_condition(env: BlindProposerEnvironment) -> dict[str, Any]:
    """Construct the condition-scoped LLM prompt and verify no out-of-condition fields leak (Prompt 8)."""
    pkg = env.get_authorized_diagnostic_package()
    arch = env.get_architecture_metadata()
    condition = env.diagnostic_condition

    # Strict negative assertions verifying condition boundaries before prompt formatting
    if condition == DiagnosticCondition.BEHAVIORAL_ONLY.value:
        assert "weight_summaries" not in pkg
        assert "activation_summaries" not in pkg
        assert "toolkit_sensitivity" not in pkg
    elif condition == DiagnosticCondition.WEIGHT_SUMMARIES_ONLY.value:
        assert "behavioral_failures" not in pkg
        assert "activation_summaries" not in pkg
        assert "toolkit_sensitivity" not in pkg
    elif condition == DiagnosticCondition.WEIGHTS_AND_FAILURES.value:
        assert "activation_summaries" not in pkg
        assert "toolkit_sensitivity" not in pkg
    elif condition == DiagnosticCondition.WEIGHTS_ACTIVATIONS_FAILURES.value:
        assert "toolkit_sensitivity" not in pkg

    compact_diag: dict[str, Any] = {
        "diagnostic_condition": condition,
        "architecture": arch["architecture"],
        "budget": arch["budget"],
        "parameter_shapes": pkg["parameter_shapes"],
    }

    if "behavioral_failures" in pkg:
        bf = pkg["behavioral_failures"]
        # Provide bounded summary + first 6 sample rows so prompt length stays compact & comparable
        inp_arr = np.array(bf["inputs"], dtype=np.float64)
        exp_arr = np.array(bf["expected_outputs"], dtype=np.float64)
        pred_arr = np.array(bf["corrupted_predictions"], dtype=np.float64)
        res_arr = exp_arr - pred_arr
        compact_diag["behavioral_failures"] = {
            "observed_diagnostic_mse": round(float(bf["observed_diagnostic_mse"]), 4),
            "sample_count": bf["diagnostic_sample_count"],
            "mean_residual_col0": round(float(np.mean(res_arr)), 4),
            "std_residual_col0": round(float(np.std(res_arr)), 4),
            "sample_examples_first_4": [
                {
                    "x": np.round(inp_arr[i], 3).tolist(),
                    "expected_y0": round(float(exp_arr[i, 0]), 3),
                    "corrupted_y0": round(float(pred_arr[i, 0]), 3),
                }
                for i in range(min(4, len(inp_arr)))
            ],
        }

    if "weight_summaries" in pkg:
        ws = pkg["weight_summaries"]
        compact_diag["weight_summaries"] = {
            "fc2.weight": {
                "shape": ws["fc2.weight"]["shape"],
                "l2_norm": ws["fc2.weight"]["l2_norm"],
                "row_l2_norms": ws["fc2.weight"]["row_l2_norms"],
                "near_zero_indices": ws["fc2.weight"]["near_zero_indices"],
                "rounded_slice": ws["fc2.weight"]["rounded_slice"],
            },
            "fc1.weight_l2_norm": ws["fc1.weight"]["l2_norm"],
        }

    if "activation_summaries" in pkg:
        compact_diag["activation_summaries"] = pkg["activation_summaries"]

    if "toolkit_sensitivity" in pkg:
        compact_diag["toolkit_sensitivity"] = pkg["toolkit_sensitivity"]

    prompt_text = (
        "You are a neural network weight repair specialist.\n"
        "Task: Propose a bounded weight delta patch for `fc2.weight` (shape [2, 8]) of `ToyMLP(4 -> 8 -> 2)` "
        "that repairs Output column 0 (row_index=0) without modifying Output column 1 (row_index=1).\n"
        f"Max L2 norm budget: {arch['budget']['max_l2_norm']}.\n"
        "Authorized Diagnostic Package:\n"
        f"{json.dumps(compact_diag, separators=(',', ':'))}\n\n"
        "Respond with ONLY a single valid JSON object matching this exact schema:\n"
        "{\n"
        '  "target_tensor": "fc2.weight",\n'
        '  "row_index": 0,\n'
        '  "col_deltas": [d0, d1, d2, d3, d4, d5, d6, d7],\n'
        '  "hypothesis": "<concise testable hypothesis>",\n'
        '  "rationale": "<explanation grounded in the provided diagnostic data>"\n'
        "}\n"
    )

    return {
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "diagnostic_condition": condition,
        "prompt_text": prompt_text,
        "prompt_char_length": len(prompt_text),
        "prompt_approx_tokens": len(prompt_text.split()),
        "supplied_keys": sorted(compact_diag.keys()),
    }


class LocalTransformersLLMBackend:
    """Real local Hugging Face causal LM backend for generating structured JSON weight repair proposals."""

    _MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 220,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _load(self) -> tuple[Any, Any, str]:
        if self.model_id in self._MODEL_CACHE:
            return self._MODEL_CACHE[self.model_id]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Check D:\cache\huggingface\hub first if present
        cache_dir = "D:/cache/huggingface/hub" if Path("D:/cache/huggingface/hub").exists() else None
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        revision = str(getattr(model.config, "_commit_hash", None) or "local-snapshot")
        self._MODEL_CACHE[self.model_id] = (tokenizer, model, revision)
        return tokenizer, model, revision

    def generate(self, prompt_text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        tokenizer, model, revision = self._load()
        messages = [
            {
                "role": "system",
                "content": "You output only valid JSON objects matching the requested schema. Never output markdown prose outside JSON.",
            },
            {"role": "user", "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_token_count = int(inputs["input_ids"].shape[1])

        from transformers import StoppingCriteria, StoppingCriteriaList

        class _BalancedJsonStopCriteria(StoppingCriteria):
            def __init__(self, tok: Any, start_len: int) -> None:
                self.tok = tok
                self.start_len = start_len

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
                gen_tokens = input_ids[0, self.start_len :]
                if len(gen_tokens) < 12:
                    return False
                text_so_far = self.tok.decode(gen_tokens, skip_special_tokens=True)
                first_open = text_so_far.find("{")
                if first_open == -1:
                    return False
                depth = 0
                in_str = False
                esc = False
                for ch in text_so_far[first_open:]:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if not in_str:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                return True
                return False

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0.0),
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_BalancedJsonStopCriteria(tokenizer, input_token_count)]),
            )
        gen_ids = outputs[0, input_token_count:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        return {
            "raw_text": raw_text,
            "model_id": self.model_id,
            "model_revision": revision,
            "decoding_config": {
                "temperature": self.temperature,
                "do_sample": (self.temperature > 0.0),
                "max_new_tokens": self.max_new_tokens,
            },
            "input_token_count": input_token_count,
            "output_token_count": int(len(gen_ids)),
            "proposer_latency_seconds": round(elapsed, 4),
        }


class OllamaLLMBackend:
    """Local Ollama HTTP backend (e.g. `bonsai-27b-q1:latest`) for structured JSON weight repair proposals.

    Uses Python stdlib `urllib.request` with zero external dependencies and Ollama's native
    `format: "json"` constraint. Never touches any external app or user data directory.
    """

    def __init__(
        self,
        model_id: str = "bonsai-27b-q1:latest",
        base_url: str = "http://127.0.0.1:11434",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        repeat_penalty: float = 1.15,
        think: bool = False,
        json_schema: dict[str, Any] | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.model_id = model_id
        self.model_revision = "ollama-local"
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.think = think
        self.json_schema = json_schema
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt_text: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        active_format: Any = json_schema if json_schema is not None else (
            self.json_schema if self.json_schema is not None else "json"
        )
        payload = {
            "model": self.model_id,
            "stream": False,
            "think": self.think,
            "keep_alive": "30m",
            "format": active_format,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise neural network weight-repair specialist. "
                        "Respond with ONLY a single valid JSON object matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "repeat_penalty": self.repeat_penalty,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        raw_text = str(body.get("message", {}).get("content", ""))
        raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
        elapsed = time.perf_counter() - t0
        return {
            "raw_text": raw_text,
            "model_id": self.model_id,
            "model_revision": str(body.get("model", self.model_id)),
            "decoding_config": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "repeat_penalty": self.repeat_penalty,
                "think": self.think,
                "format": "json_schema" if isinstance(active_format, dict) else active_format,
            },
            "input_token_count": int(body.get("prompt_eval_count", 0)),
            "output_token_count": int(body.get("eval_count", 0)),
            "proposer_latency_seconds": round(elapsed, 4),
        }


class LLMWeightRepairProposer:
    """Executes an LLM-driven blind repair proposal and submits it to the sealed evaluator (Prompt 7)."""

    def __init__(
        self,
        backend: LocalTransformersLLMBackend | Callable[[str], dict[str, Any]] | None = None,
    ):
        self.backend = backend or LocalTransformersLLMBackend()

    def run_on_environment(self, env: BlindProposerEnvironment) -> dict[str, Any]:
        prompt_bundle = build_diagnostic_prompt_for_condition(env)
        prompt_sha256 = hashlib.sha256(prompt_bundle["prompt_text"].encode("utf-8")).hexdigest()

        t0 = time.perf_counter()
        try:
            if callable(self.backend) and not hasattr(self.backend, "generate"):
                gen_out = self.backend(prompt_bundle["prompt_text"])
            else:
                gen_out = self.backend.generate(prompt_bundle["prompt_text"])
        except Exception as provider_exc:
            model_id = getattr(self.backend, "model_id", "custom_llm")
            model_revision = getattr(self.backend, "model_revision", "unknown")
            return {
                "method": "llm_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "proposal_status": "PROVIDER_ERROR",
                "error_message": str(provider_exc),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "passed_gates": False,
                "evaluation_result": None,
            }

        raw_text = gen_out.get("raw_text", "")
        model_id = gen_out.get("model_id", "custom_llm")
        model_revision = gen_out.get("model_revision", "unknown")

        try:
            patch = parse_and_validate_llm_proposal(
                raw_llm_text=raw_text,
                env=env,
                proposed_by=f"llm:{model_id}",
                patch_id=f"llm-patch-{env.experiment_id[:12]}",
            )
        except (BudgetViolationError, AllowlistViolationError, BaseModelMismatchError) as budget_exc:
            return {
                "method": "llm_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "decoding_config": gen_out.get("decoding_config", {}),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "input_token_count": gen_out.get("input_token_count"),
                "output_token_count": gen_out.get("output_token_count"),
                "proposal_status": "OUT_OF_BUDGET_OR_ALLOWLIST",
                "error_message": str(budget_exc),
                "raw_llm_output": raw_text,
                "passed_gates": False,
                "evaluation_result": None,
            }
        except Exception as parse_exc:
            return {
                "method": "llm_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "decoding_config": gen_out.get("decoding_config", {}),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "input_token_count": gen_out.get("input_token_count"),
                "output_token_count": gen_out.get("output_token_count"),
                "proposal_status": "INVALID_PROPOSAL_SCHEMA",
                "error_message": str(parse_exc),
                "raw_llm_output": raw_text,
                "passed_gates": False,
                "evaluation_result": None,
            }

        try:
            eval_res = env.submit_candidate_patch(patch)
            elapsed = time.perf_counter() - t0
            return {
                "method": "llm_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "decoding_config": gen_out.get("decoding_config", {}),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "input_token_count": gen_out.get("input_token_count"),
                "output_token_count": gen_out.get("output_token_count"),
                "proposal_status": "EVALUATED",
                "patch_id": patch.patch_id,
                "patch_sha256": patch.sha256(),
                "patch_l2_norm": round(float(np.linalg.norm(patch.deltas["fc2.weight"])), 4),
                "raw_llm_output": raw_text,
                "passed_gates": bool(eval_res["passed_gates"]),
                "target_held_out_gain": float(eval_res["gains"]["target_held_out_gain"]),
                "ood_gain": float(eval_res["gains"]["distribution_shift_gain"]),
                "regression_drop": float(eval_res["gains"]["regression_drop"]),
                "total_time_seconds": round(elapsed, 4),
                "evaluation_result": eval_res,
            }
        except Exception as eval_exc:
            return {
                "method": "llm_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "proposal_status": "EVALUATOR_REJECTED",
                "error_message": str(eval_exc),
                "patch_sha256": patch.sha256(),
                "passed_gates": False,
                "evaluation_result": None,
            }


class GradientRepairProposer:
    """Empirical gradient-based repair baseline using only permitted diagnostic data (Prompt 9)."""

    def __init__(self, steps: int = 80, lr: float = 0.05):
        self.steps = steps
        self.lr = lr

    def run_on_environment(self, env: BlindProposerEnvironment) -> dict[str, Any]:
        t0 = time.perf_counter()
        pkg = env.get_authorized_diagnostic_package()
        if "behavioral_failures" not in pkg or env._corrupted_model is None:
            return {
                "method": "gradient_baseline",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "proposal_status": "INSUFFICIENT_DIAGNOSTIC_MODALITY",
                "error_message": "Gradient repair requires both model weights and behavioral failure examples.",
                "passed_gates": False,
                "evaluation_result": None,
            }

        report = env.get_diagnostic_failure_report()
        corrupted = env.get_corrupted_model()
        corrupted.fc1.weight.requires_grad = False
        corrupted.fc1.bias.requires_grad = False
        corrupted.fc2.bias.requires_grad = False
        corrupted.fc2.weight.requires_grad = True

        opt = torch.optim.Adam([corrupted.fc2.weight], lr=self.lr)
        init_w = corrupted.fc2.weight.data.clone()

        for _ in range(self.steps):
            opt.zero_grad()
            preds = corrupted(report["inputs"])[:, :1]
            loss = torch.nn.functional.mse_loss(preds, report["expected_outputs"])
            loss.backward()
            if corrupted.fc2.weight.grad is not None:
                corrupted.fc2.weight.grad[1, :] = 0.0
            opt.step()

        delta_arr = (corrupted.fc2.weight.data - init_w).detach().cpu().numpy()
        # Clip to budget max_l2_norm if needed
        l2 = float(np.linalg.norm(delta_arr))
        if l2 > env.budget.max_l2_norm * 0.98:
            delta_arr = delta_arr * ((env.budget.max_l2_norm * 0.98) / max(l2, 1e-8))

        patch = WeightPatch(
            patch_id=f"grad-patch-{env.experiment_id[:12]}",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=env.corrupted_fingerprint,
            deltas={"fc2.weight": delta_arr},
            hypothesis="Empirical Adam gradient optimization on diagnostic failure samples.",
            proposed_by="gradient_baseline",
        )

        eval_res = env.submit_candidate_patch(patch)
        elapsed = time.perf_counter() - t0
        return {
            "method": "gradient_baseline",
            "experiment_id": env.experiment_id,
            "defect_family": env.defect_family,
            "diagnostic_condition": env.diagnostic_condition,
            "proposal_status": "EVALUATED",
            "patch_id": patch.patch_id,
            "patch_sha256": patch.sha256(),
            "patch_l2_norm": round(float(np.linalg.norm(delta_arr)), 4),
            "passed_gates": bool(eval_res["passed_gates"]),
            "target_held_out_gain": float(eval_res["gains"]["target_held_out_gain"]),
            "ood_gain": float(eval_res["gains"]["distribution_shift_gain"]),
            "regression_drop": float(eval_res["gains"]["regression_drop"]),
            "total_time_seconds": round(elapsed, 4),
            "evaluation_result": eval_res,
        }


class DeterministicHeuristicProposer:
    """Non-LLM deterministic baseline using closed-form heuristics from available diagnostics (Prompt 9)."""

    def run_on_environment(self, env: BlindProposerEnvironment) -> dict[str, Any]:
        t0 = time.perf_counter()
        pkg = env.get_authorized_diagnostic_package()
        delta_arr = np.zeros((2, 8), dtype=np.float32)

        # Heuristic 1: If weight summaries are present, check near-zero entries in row 0 of fc2.weight
        # and impute with the RMS magnitude of row 1 (control row)
        if "weight_summaries" in pkg:
            fc2_info = pkg["weight_summaries"]["fc2.weight"]
            w_slice = np.array(fc2_info["rounded_slice"], dtype=np.float32)
            row0 = w_slice[0, :]
            row1 = w_slice[1, :]
            ref_scale = float(np.sqrt(np.mean(row1**2))) if np.linalg.norm(row1) > 1e-6 else 0.35
            zero_mask = np.abs(row0) < 1e-4
            if np.any(zero_mask):
                delta_arr[0, zero_mask] = ref_scale
            else:
                # If row0 norm is much smaller than row1 norm (scale attenuation), boost row0
                if np.linalg.norm(row0) < 0.5 * np.linalg.norm(row1):
                    delta_arr[0, :] = row0 * 1.5
                else:
                    delta_arr[0, :4] = -0.5 * row0[:4]
        elif "behavioral_failures" in pkg:
            # Behavioral-only fallback: uniform small positive shift proportional to mean residual
            bf = pkg["behavioral_failures"]
            exp_arr = np.array(bf["expected_outputs"], dtype=np.float32)
            pred_arr = np.array(bf["corrupted_predictions"], dtype=np.float32)
            mean_res = float(np.mean(exp_arr - pred_arr))
            step_val = float(np.clip(mean_res * 0.15, -0.4, 0.4))
            if abs(step_val) < 1e-4:
                step_val = 0.1
            delta_arr[0, :] = step_val

        patch = WeightPatch(
            patch_id=f"det-patch-{env.experiment_id[:12]}",
            base_model_id="toy-mlp-ground-truth",
            base_model_revision="corrupted-v1",
            base_model_fingerprint=env.corrupted_fingerprint,
            deltas={"fc2.weight": delta_arr},
            hypothesis="Deterministic non-LLM heuristic repair from permitted summary statistics.",
            proposed_by="deterministic_heuristic_baseline",
        )

        eval_res = env.submit_candidate_patch(patch)
        elapsed = time.perf_counter() - t0
        return {
            "method": "deterministic_baseline",
            "experiment_id": env.experiment_id,
            "defect_family": env.defect_family,
            "diagnostic_condition": env.diagnostic_condition,
            "proposal_status": "EVALUATED",
            "patch_id": patch.patch_id,
            "patch_sha256": patch.sha256(),
            "patch_l2_norm": round(float(np.linalg.norm(delta_arr)), 4),
            "passed_gates": bool(eval_res["passed_gates"]),
            "target_held_out_gain": float(eval_res["gains"]["target_held_out_gain"]),
            "ood_gain": float(eval_res["gains"]["distribution_shift_gain"]),
            "regression_drop": float(eval_res["gains"]["regression_drop"]),
            "total_time_seconds": round(elapsed, 4),
            "evaluation_result": eval_res,
        }


PROJECTION_PROMPT_VERSION = "wsp-llm-projection-v1.0"


class StructuredProjectionPlan(BaseModel):
    """Structured gradient-free projection repair plan (Fan et al. arXiv:2611.21885 style).

    Allows an LLM to diagnose a defect from authorized diagnostics and invoke a bounded,
    non-arbitrary algebraic primitive strictly on proposer-authorized data (no `eval`/`exec`).
    """

    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "least_squares_projection",
        "sign_flip_indices",
        "scale_row",
        "explicit_deltas",
    ]
    target_tensor: str = Field(default="fc2.weight", min_length=1, max_length=128)
    row_index: int = Field(default=0, ge=0)
    col_indices: list[int] | None = None
    scale_factor: float | None = None
    col_deltas: list[float] | None = None
    ridge_lambda: float | None = Field(default=1e-3, ge=1e-8, le=10.0)
    hypothesis: str = Field(..., min_length=3, max_length=512)
    rationale: str = Field(..., min_length=3, max_length=1024)


def build_projection_plan_prompt_for_condition(env: BlindProposerEnvironment) -> dict[str, Any]:
    """Build condition-scoped prompt for `SandboxedProjectionLLMProposer` with strict isolation checks."""
    base_bundle = build_diagnostic_prompt_for_condition(env)
    pkg = env.get_authorized_diagnostic_package()
    arch = env.get_architecture_metadata()
    condition = env.diagnostic_condition

    has_weights = "weight_summaries" in pkg
    has_failures = "behavioral_failures" in pkg

    available_ops: list[str] = ["explicit_deltas"]
    if has_weights:
        available_ops.extend(["sign_flip_indices", "scale_row"])
    if has_weights and has_failures:
        available_ops.insert(0, "least_squares_projection")

    # Reuse the already-verified condition-scoped diagnostic JSON embedded in base_bundle
    diag_marker = "Authorized Diagnostic Package:\n"
    diag_start = base_bundle["prompt_text"].index(diag_marker) + len(diag_marker)
    diag_end = base_bundle["prompt_text"].index("\n\nRespond with ONLY", diag_start)
    diag_json_str = base_bundle["prompt_text"][diag_start:diag_end]

    prompt_text = (
        "You are a neural network weight-space repair architect.\n"
        "Task: Diagnose the defect on Output column 0 (row_index=0) of `fc2.weight` (shape [2, 8]) "
        "in `ToyMLP(4 -> 8 -> 2)` without touching Output column 1 (row_index=1).\n"
        f"Max L2 norm budget: {arch['budget']['max_l2_norm']}.\n"
        f"Available gradient-free operations for this diagnostic condition: {available_ops}.\n"
        "- `least_squares_projection`: Solves closed-form ridge projection delta_w = (H^T H + lambda I)^-1 H^T r "
        "using the authorized diagnostic inputs, expected outputs, and corrupted hidden activations (requires weights + failures).\n"
        "- `sign_flip_indices`: Sets delta[row_index, j] = -2 * w[row_index, j] for specified `col_indices` (requires weights).\n"
        "- `scale_row`: Rescales row_index by `scale_factor`, setting delta = (scale_factor - 1) * w[row_index] (requires weights).\n"
        "- `explicit_deltas`: Directly applies 8 floating-point `col_deltas`.\n\n"
        "Authorized Diagnostic Package:\n"
        f"{diag_json_str}\n\n"
        "Respond with ONLY a single valid JSON object matching this exact schema:\n"
        "{\n"
        '  "operation": "<one of available_ops>",\n'
        '  "target_tensor": "fc2.weight",\n'
        '  "row_index": 0,\n'
        '  "col_indices": [0, 1, 2, 3],\n'
        '  "scale_factor": 1.0,\n'
        '  "col_deltas": null,\n'
        '  "ridge_lambda": 0.001,\n'
        '  "hypothesis": "<concise testable hypothesis>",\n'
        '  "rationale": "<diagnosis grounded in the authorized diagnostic package>"\n'
        "}\n"
    )
    return {
        "prompt_version": PROJECTION_PROMPT_VERSION,
        "diagnostic_condition": condition,
        "prompt_text": prompt_text,
        "prompt_char_length": len(prompt_text),
        "prompt_approx_tokens": len(prompt_text.split()),
        "supplied_keys": base_bundle["supplied_keys"],
        "available_ops": available_ops,
    }


def compile_projection_plan_to_patch(
    raw_llm_text: str,
    env: BlindProposerEnvironment,
    proposed_by: str,
    patch_id: str,
) -> tuple[WeightPatch, StructuredProjectionPlan]:
    """Parse LLM `StructuredProjectionPlan` JSON and compile it into a bounded `WeightPatch`."""
    if not isinstance(raw_llm_text, str) or not raw_llm_text.strip():
        raise ValueError("LLM returned empty response text.")

    text = raw_llm_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        json_candidate = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM output does not contain a JSON object.")
        json_candidate = text[start : end + 1]

    parsed_dict = json.loads(json_candidate)
    plan = StructuredProjectionPlan.model_validate(parsed_dict)

    arch = env.get_architecture_metadata()
    params_meta = arch["parameters"]
    if plan.target_tensor not in params_meta:
        raise AllowlistViolationError(f"Unknown target_tensor '{plan.target_tensor}'.")
    allowed = env.budget.allowed_tensor_names
    if allowed is not None and plan.target_tensor not in allowed:
        raise AllowlistViolationError(f"target_tensor '{plan.target_tensor}' not in allowlist {list(allowed)}.")

    shape = tuple(int(d) for d in params_meta[plan.target_tensor]["shape"])
    n_rows, n_cols = shape
    if plan.row_index < 0 or plan.row_index >= n_rows:
        raise ValueError(f"row_index {plan.row_index} out of bounds for {n_rows} rows.")

    pkg = env.get_authorized_diagnostic_package()
    delta_arr = np.zeros(shape, dtype=np.float32)

    if plan.operation == "least_squares_projection":
        if env._corrupted_model is None or "behavioral_failures" not in pkg:
            raise ValueError(
                f"Operation 'least_squares_projection' requires both weights and behavioral_failures "
                f"(unavailable in condition '{env.diagnostic_condition}')."
            )
        model = env.get_corrupted_model()
        diag = env.get_diagnostic_failure_report()
        x = diag["inputs"]
        y_exp = diag["expected_outputs"][:, plan.row_index].detach().cpu().numpy().astype(np.float64)
        y_pred = diag["corrupted_predictions"][:, plan.row_index].detach().cpu().numpy().astype(np.float64)
        residual = y_exp - y_pred

        with torch.no_grad():
            h = torch.relu(model.fc1(x)).detach().cpu().numpy().astype(np.float64)  # [N, 8]
        lam = float(plan.ridge_lambda) if plan.ridge_lambda is not None else 1e-3
        gram = h.T @ h + lam * np.eye(n_cols, dtype=np.float64)
        rhs = h.T @ residual
        solved = np.linalg.solve(gram, rhs).astype(np.float32)
        if plan.col_indices:
            valid_idx = [i for i in plan.col_indices if 0 <= i < n_cols]
            mask = np.zeros(n_cols, dtype=np.float32)
            mask[valid_idx] = 1.0
            solved = solved * mask
        delta_arr[plan.row_index, :] = solved

    elif plan.operation == "sign_flip_indices":
        if "weight_summaries" not in pkg:
            raise ValueError(f"Operation 'sign_flip_indices' requires weight_summaries (condition='{env.diagnostic_condition}').")
        w_slice = np.array(pkg["weight_summaries"][plan.target_tensor]["rounded_slice"], dtype=np.float32)
        indices = plan.col_indices if plan.col_indices else list(range(n_cols))
        for j in indices:
            if j < 0 or j >= n_cols:
                raise ValueError(f"col index {j} out of bounds for width {n_cols}.")
            delta_arr[plan.row_index, j] = -2.0 * w_slice[plan.row_index, j]

    elif plan.operation == "scale_row":
        if "weight_summaries" not in pkg:
            raise ValueError(f"Operation 'scale_row' requires weight_summaries (condition='{env.diagnostic_condition}').")
        if plan.scale_factor is None or not np.isfinite(float(plan.scale_factor)):
            raise ValueError("scale_row requires a finite numeric scale_factor.")
        w_slice = np.array(pkg["weight_summaries"][plan.target_tensor]["rounded_slice"], dtype=np.float32)
        delta_arr[plan.row_index, :] = (float(plan.scale_factor) - 1.0) * w_slice[plan.row_index, :]

    elif plan.operation == "explicit_deltas":
        if plan.col_deltas is None or len(plan.col_deltas) != n_cols:
            raise ValueError(f"explicit_deltas requires exactly {n_cols} finite floats in col_deltas.")
        delta_arr[plan.row_index, :] = np.array(plan.col_deltas, dtype=np.float32)

    if not np.isfinite(delta_arr).all():
        raise BudgetViolationError("Compiled projection delta contains non-finite values.")

    norm = float(np.linalg.norm(delta_arr))
    if norm == 0.0:
        raise BudgetViolationError("Compiled projection plan produced an all-zero delta.")
    if norm > env.budget.max_l2_norm:
        # Respect budget by projecting onto L2 ball (same as AstraContinuousProposer)
        delta_arr = delta_arr * np.float32((env.budget.max_l2_norm * 0.99) / norm)

    patch = WeightPatch(
        patch_id=patch_id,
        base_model_id="toy-mlp-ground-truth",
        base_model_revision="corrupted-v1",
        base_model_fingerprint=env.corrupted_fingerprint,
        deltas={plan.target_tensor: delta_arr},
        hypothesis=plan.hypothesis,
        proposed_by=proposed_by,
        rationale=f"[op={plan.operation}] {plan.rationale}",
    )
    if env._corrupted_model is not None:
        validate_patch_against_budget(env.get_corrupted_model(), patch, env.budget)
    return patch, plan


class SandboxedProjectionLLMProposer:
    """Gradient-free LLM proposer that diagnoses defects and dispatches bounded algebraic projections."""

    def __init__(self, backend: OllamaLLMBackend | LocalTransformersLLMBackend | Callable[[str], dict[str, Any]]):
        self.backend = backend

    def run_on_environment(self, env: BlindProposerEnvironment) -> dict[str, Any]:
        prompt_bundle = build_projection_plan_prompt_for_condition(env)
        prompt_sha256 = hashlib.sha256(prompt_bundle["prompt_text"].encode("utf-8")).hexdigest()

        t0 = time.perf_counter()
        try:
            if callable(self.backend) and not hasattr(self.backend, "generate"):
                gen_out = self.backend(prompt_bundle["prompt_text"])
            else:
                gen_out = self.backend.generate(prompt_bundle["prompt_text"])
        except Exception as provider_exc:
            return {
                "method": "llm_projection_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": getattr(self.backend, "model_id", "custom_llm"),
                "proposal_status": "PROVIDER_ERROR",
                "error_message": str(provider_exc),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "passed_gates": False,
                "evaluation_result": None,
            }

        raw_text = gen_out.get("raw_text", "")
        model_id = gen_out.get("model_id", getattr(self.backend, "model_id", "custom_llm"))
        model_revision = gen_out.get("model_revision", "unknown")

        try:
            patch, plan = compile_projection_plan_to_patch(
                raw_llm_text=raw_text,
                env=env,
                proposed_by=f"llm_projection:{model_id}",
                patch_id=f"llm-proj-{env.experiment_id[:12]}",
            )
        except (BudgetViolationError, AllowlistViolationError, BaseModelMismatchError) as budget_exc:
            return {
                "method": "llm_projection_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "proposal_status": "OUT_OF_BUDGET_OR_ALLOWLIST",
                "error_message": str(budget_exc),
                "raw_llm_text": raw_text,
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "passed_gates": False,
                "evaluation_result": None,
            }
        except Exception as parse_exc:
            return {
                "method": "llm_projection_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "proposal_status": "INVALID_PROPOSAL_SCHEMA",
                "error_message": str(parse_exc),
                "raw_llm_text": raw_text,
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "passed_gates": False,
                "evaluation_result": None,
            }

        try:
            eval_res = env.submit_candidate_patch(patch)
            elapsed = time.perf_counter() - t0
            return {
                "method": "llm_projection_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "operation_chosen": plan.operation,
                "decoding_config": gen_out.get("decoding_config", {}),
                "prompt_version": prompt_bundle["prompt_version"],
                "prompt_sha256": prompt_sha256,
                "prompt_char_length": prompt_bundle["prompt_char_length"],
                "input_token_count": gen_out.get("input_token_count"),
                "output_token_count": gen_out.get("output_token_count"),
                "proposer_latency_seconds": gen_out.get("proposer_latency_seconds"),
                "total_time_seconds": round(elapsed, 4),
                "proposal_status": "EVALUATED",
                "patch_id": patch.patch_id,
                "patch_sha256": patch.sha256(),
                "patch_l2_norm": round(float(np.linalg.norm(patch.deltas["fc2.weight"])), 4),
                "hypothesis": patch.hypothesis,
                "passed_gates": bool(eval_res["passed_gates"]),
                "target_held_out_gain": float(eval_res["gains"]["target_held_out_gain"]),
                "ood_gain": float(eval_res["gains"]["distribution_shift_gain"]),
                "regression_drop": float(eval_res["gains"]["regression_drop"]),
                "evaluation_result": eval_res,
            }
        except Exception as eval_exc:
            return {
                "method": "llm_projection_proposer",
                "experiment_id": env.experiment_id,
                "defect_family": env.defect_family,
                "diagnostic_condition": env.diagnostic_condition,
                "model_id": model_id,
                "model_revision": model_revision,
                "operation_chosen": plan.operation,
                "proposal_status": "EVALUATOR_REJECTED",
                "error_message": str(eval_exc),
                "passed_gates": False,
                "evaluation_result": None,
            }
