"""
Evaluation utilities for multi-agent debate.

Provides answer extraction, matching, and majority vote logic
for different QA benchmarks (GSM8K, MMLU, CommonsenseQA, HellaSwag,
GPQA, ARC-Challenge).
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# XML tag extraction
# ---------------------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
    """
    Extract answer from <answer>...</answer> XML tags.

    Args:
        text: Model output text.

    Returns:
        Extracted answer string, or empty string on failure.
    """
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        return answer.strip()
    except Exception:
        return ""


def extract_xml_confidence(text: str):
    """
    Extract confidence score from <confidence>...</confidence> XML tags.

    Args:
        text: Model output text.

    Returns:
        Integer confidence value (0-10), or None if parsing fails.
    """
    try:
        raw = text.split("<confidence>")[-1].split("</confidence>")[0].strip()
        return int(raw)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Answer cleaning
# ---------------------------------------------------------------------------

def clean_answer(pred: str) -> str:
    """
    Clean answer for GSM8K-style numeric questions.

    Removes formatting (commas, dollar signs) and extracts the last
    number from the string.

    Args:
        pred: Raw prediction string (e.g. "$48,000" or "48000").

    Returns:
        Cleaned numeric string (e.g. "48000"), or "[invalid]" if
        no number is found.
    """
    pred = pred.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", pred)
    if not numbers:
        return "[invalid]"
    last_number = numbers[-1]
    if last_number.endswith("."):
        last_number = last_number[:-1]
    return last_number


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def gsm8k_match(prediction: str, ground_truth: str) -> bool:
    """Numeric match for GSM8K-style answers after cleaning."""
    pred_clean = clean_answer(prediction)
    label_clean = clean_answer(ground_truth)
    return (pred_clean == label_clean) and (pred_clean != "[invalid]")


def multi_choice_match(prediction: str, ground_truth: str) -> bool:
    """
    Multiple-choice match.

    Handles variants such as "C", "(C)", "C." — returns True when the
    prediction starts with the ground-truth option letter.
    """
    prediction = prediction.strip()
    candidates = [ground_truth.strip()]
    match = re.match(r"\(?([A-Za-z])\)?\.?", ground_truth)
    if match:
        letter = match.group(1)
        candidates.extend([letter, f"({letter})", f"{letter}."])
    return any(prediction.startswith(gt) for gt in candidates)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive exact match."""
    return prediction.strip().lower() == ground_truth.strip().lower()


def check_answer_correct(
    prediction: str, ground_truth: str, dataset: str
) -> bool:
    """
    Dataset-aware correctness check.

    Routes to the appropriate matching function based on *dataset* name.

    Args:
        prediction: Predicted answer string.
        ground_truth: Ground-truth answer string.
        dataset: Dataset name (e.g. "gsm8k", "mmlu", "csqa").

    Returns:
        True if the prediction is correct.
    """
    dl = dataset.lower()

    if "gsm8k" in dl or "arithmetic" in dl:
        return gsm8k_match(prediction, ground_truth)
    elif any(n in dl for n in ["mmlu", "hellaswag", "csqa", "gpqa", "arc"]):
        return multi_choice_match(prediction, ground_truth)
    else:
        return exact_match(prediction, ground_truth)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def get_majority_answer(answers: List[str], dataset: str) -> str:
    """
    Compute majority answer with dataset-specific normalisation.

    For numeric datasets (GSM8K) answers are cleaned before counting.
    For multiple-choice datasets, answers are normalised to single
    uppercase letters.

    Args:
        answers: List of answer strings from agents.
        dataset: Dataset name.

    Returns:
        The majority answer string.
    """
    if not answers:
        return ""

    dl = dataset.lower()

    # Numeric datasets — clean before counting
    if "gsm8k" in dl or "arithmetic" in dl:
        cleaned = [clean_answer(a) for a in answers]
        valid = [a for a in cleaned if a != "[invalid]"]
        if not valid:
            return answers[0]
        majority = max(set(valid), key=valid.count)
        for orig, c in zip(answers, cleaned):
            if c == majority:
                return orig
        return answers[0]

    # Multiple-choice — normalise to single letter
    if any(n in dl for n in ["mmlu", "hellaswag", "csqa", "gpqa", "arc"]):
        normalised = []
        for a in answers:
            m = re.match(r"\(?([A-Za-z])\)?\.?", a.strip())
            normalised.append(m.group(1).upper() if m else a.strip())
        majority = max(set(normalised), key=normalised.count)
        for orig, n in zip(answers, normalised):
            if n == majority:
                return orig
        return answers[0]

    # Default — simple majority
    return max(set(answers), key=answers.count)
