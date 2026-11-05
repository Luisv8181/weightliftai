"""Weight-Space Programming and Continuous Model Improvement package.

Provides declarative parameter patches, safety budgets, sandboxed execution,
and independent evaluation.
"""

from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.inspector import (
    inspect_model_parameters,
    validate_patch_against_budget,
)
from weightliftai.weight_programming.sandbox import (
    apply_patch_sandboxed,
    apply_patch_to_copy,
)
from weightliftai.weight_programming.schemas import (
    AllowlistViolationError,
    BudgetViolationError,
    PatchBudget,
    WeightPatch,
)
from weightliftai.weight_programming.toy_defect import (
    ToyMLP,
    generate_synthetic_data,
    run_known_defect_experiment,
)

__all__ = [
    "AllowlistViolationError",
    "BudgetViolationError",
    "IndependentEvaluator",
    "PatchBudget",
    "ToyMLP",
    "WeightPatch",
    "apply_patch_sandboxed",
    "apply_patch_to_copy",
    "generate_synthetic_data",
    "inspect_model_parameters",
    "run_known_defect_experiment",
    "validate_patch_against_budget",
]
