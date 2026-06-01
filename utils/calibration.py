"""
Calibration metrics for evaluating confidence-modulated models.

Implements Expected Calibration Error (ECE), Brier score, and AUROC
for measuring how well-calibrated the expressed confidence levels are.

Reference: Section 5 (evaluation metrics) of the paper.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.metrics import brier_score_loss, roc_auc_score
except ImportError:
    brier_score_loss = None
    roc_auc_score = None


def compute_ece(
    confidences: List[int],
    correctness: List[bool],
    num_bins: int = 10,
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidences: Confidence values on a 0–10 scale.
        correctness: Boolean correctness labels.
        num_bins: Number of bins for the reliability diagram.

    Returns:
        ``(ece, bin_info)`` where *bin_info* contains per-bin
        accuracies, confidences, and counts.
    """
    confs = np.array(confidences) / 10.0
    cors = np.array(correctness, dtype=float)

    edges = np.linspace(0, 1, num_bins + 1)
    bin_info: Dict = {
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }
    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        prop = mask.mean()
        if prop > 0:
            acc = cors[mask].mean()
            avg_conf = confs[mask].mean()
            ece += abs(avg_conf - acc) * prop
            bin_info["bin_accuracies"].append(float(acc))
            bin_info["bin_confidences"].append(float(avg_conf))
            bin_info["bin_counts"].append(int(mask.sum()))
        else:
            bin_info["bin_accuracies"].append(None)
            bin_info["bin_confidences"].append(None)
            bin_info["bin_counts"].append(0)

    return float(ece), bin_info


def compute_brier_score(
    confidences: List[int], correctness: List[bool]
) -> float:
    """
    Compute Brier score (lower is better; 0 = perfect, 1 = worst).

    Args:
        confidences: Confidence values on a 0–10 scale.
        correctness: Boolean correctness labels.
    """
    if len(confidences) < 1 or brier_score_loss is None:
        return 1.0
    try:
        probs = np.array(confidences) / 10.0
        labels = np.array(correctness, dtype=int)
        return float(brier_score_loss(labels, probs))
    except Exception:
        return 1.0


def compute_auroc(
    confidences: List[int], correctness: List[bool]
) -> float | None:
    """
    Compute AUROC (higher is better; 0.5 = random, 1.0 = perfect).

    Args:
        confidences: Confidence values on a 0–10 scale.
        correctness: Boolean correctness labels.
    """
    if roc_auc_score is None:
        return None
    try:
        probs = np.array(confidences) / 10.0
        labels = np.array(correctness, dtype=int)
        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def evaluate_calibration(
    completions: List[str],
    labels: List[str],
    datasets: List[str] | None = None,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    End-to-end calibration evaluation for a batch of model completions.

    Extracts ``<answer>`` and ``<confidence>`` from each completion,
    computes correctness using dataset-aware matching, then reports
    ECE, Brier score, AUROC, accuracy, and average confidence.

    Args:
        completions: Raw model output strings.
        labels: Ground-truth labels.
        datasets: Per-example dataset names (for routing to the
                  correct matching function).
        num_bins: Number of bins for ECE.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Lazy import to avoid circular dependency at module level
    from utils.eval_utils import (
        extract_xml_answer,
        extract_xml_confidence,
        check_answer_correct,
    )

    confs_list: List[int] = []
    cors_list: List[bool] = []
    valid = 0

    for idx, (comp, label) in enumerate(zip(completions, labels)):
        pred = extract_xml_answer(comp)
        conf = extract_xml_confidence(comp)
        if not pred or conf is None or not (0 <= conf <= 10):
            continue

        ds = datasets[idx] if datasets and idx < len(datasets) else ""
        correct = check_answer_correct(pred, label, ds)

        confs_list.append(conf)
        cors_list.append(correct)
        valid += 1

    total = len(completions)
    if not confs_list:
        return {
            "ece": 1.0,
            "brier_score": 1.0,
            "auroc": 0.5,
            "accuracy": 0.0,
            "valid_ratio": 0.0,
            "avg_confidence": 0.0,
            "num_valid": 0,
            "num_total": total,
        }

    ece, _ = compute_ece(confs_list, cors_list, num_bins)
    brier = compute_brier_score(confs_list, cors_list)
    auroc = compute_auroc(confs_list, cors_list)

    return {
        "ece": ece,
        "brier_score": brier,
        "auroc": auroc,
        "accuracy": float(np.mean(cors_list)),
        "valid_ratio": valid / total,
        "avg_confidence": float(np.mean(confs_list)),
        "num_valid": valid,
        "num_total": total,
    }
