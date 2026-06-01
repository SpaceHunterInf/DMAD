"""
Utility modules for DMAD (Demystifying Multi-Agent Debate).

Provides shared evaluation and data utilities used by both
inference and training scripts.
"""

from .eval_utils import (
    extract_xml_answer,
    extract_xml_confidence,
    clean_answer,
    gsm8k_match,
    multi_choice_match,
    exact_match,
    check_answer_correct,
    get_majority_answer,
)

from .data_utils import (
    load_debate_dataset,
    save_debate_results,
    chunk_list,
    calculate_accuracy,
)

from .calibration import (
    compute_ece,
    compute_brier_score,
    compute_auroc,
    evaluate_calibration,
)

__all__ = [
    # Evaluation utilities
    "extract_xml_answer",
    "extract_xml_confidence",
    "clean_answer",
    "gsm8k_match",
    "multi_choice_match",
    "exact_match",
    "check_answer_correct",
    "get_majority_answer",
    # Data utilities
    "load_debate_dataset",
    "save_debate_results",
    "chunk_list",
    "calculate_accuracy",
    # Calibration metrics
    "compute_ece",
    "compute_brier_score",
    "compute_auroc",
    "evaluate_calibration",
]
