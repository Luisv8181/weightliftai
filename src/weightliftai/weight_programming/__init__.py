from weightliftai.weight_programming.benchmark_ladder import (
    BenchmarkLadderRunner,
    DeepNonlinearMLP,
    LadderProgressionReport,
    LadderRung,
    RungEvaluationResult,
    SimpleTransformerBlock,
)
from weightliftai.weight_programming.statistical_benchmark import (
    MultiSeedBenchmarkRunner,
    MultiSeedRunRecord,
    NoOpProposer,
    OrthogonalShamProposer,
    RandomNormMatchedProposer,
    StatisticalPromotionDecision,
    compute_bootstrap_ci,
    paired_permutation_test,
)
"""Weight-Space Programming and Continuous Model Improvement package.

Provides declarative parameter patches, safety budgets, sandboxed execution,
and independent evaluation.
"""

from weightliftai.weight_programming.blind_benchmark import BlindRepairBenchmark
from weightliftai.weight_programming.evaluator import IndependentEvaluator
from weightliftai.weight_programming.inspector import (
    compute_model_fingerprint,
    hash_tensor_bytes,
    inspect_model_parameters,
    validate_patch_against_budget,
)
from weightliftai.weight_programming.math500_evaluator import (
    MathBenchmarkResult,
    build_math_prompt,
    detect_cyclic_repetition,
    evaluate_model_on_math_dataset,
    extract_boxed_answer,
    is_math_equivalent,
    normalize_math_expression,
)
from weightliftai.weight_programming.mountain_car import (
    AstraContinuousProposer,
    ContinuousCarDefectFamily,
    ContinuousMountainCarEnv,
    RampPolicyNet,
    apply_car_patch,
    evaluate_car_policy,
    get_clean_ramp_policy,
    inject_car_defect,
)
from weightliftai.weight_programming.sandbox import (
    apply_patch_sandboxed,
    apply_patch_to_copy,
)
from weightliftai.weight_programming.schemas import (
    AllowlistViolationError,
    BaseModelMismatchError,
    BudgetViolationError,
    DtypeMismatchError,
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
    "AstraContinuousProposer",
    "BaseModelMismatchError",
    "BlindRepairBenchmark",
    "BudgetViolationError",
    "DtypeMismatchError",
    "ContinuousCarDefectFamily",
    "ContinuousMountainCarEnv",
    "IndependentEvaluator",
    "MathBenchmarkResult",
    "PatchBudget",
    "RampPolicyNet",
    "ToyMLP",
    "WeightPatch",
    "apply_car_patch",
    "apply_patch_sandboxed",
    "apply_patch_to_copy",
    "build_math_prompt",
    "compute_model_fingerprint",
    "detect_cyclic_repetition",
    "evaluate_car_policy",
    "evaluate_model_on_math_dataset",
    "extract_boxed_answer",
    "generate_synthetic_data",
    "get_clean_ramp_policy",
    "hash_tensor_bytes",
    "inject_car_defect",
    "inspect_model_parameters",
    "is_math_equivalent",
    "normalize_math_expression",
    "run_known_defect_experiment",
    "validate_patch_against_budget",
    "BenchmarkLadderRunner",
    "DeepNonlinearMLP",
    "LadderProgressionReport",
    "LadderRung",
    "MultiSeedBenchmarkRunner",
    "MultiSeedRunRecord",
    "NoOpProposer",
    "OrthogonalShamProposer",
    "RandomNormMatchedProposer",
    "RungEvaluationResult",
    "SimpleTransformerBlock",
    "StatisticalPromotionDecision",
    "compute_bootstrap_ci",
    "paired_permutation_test",
]
