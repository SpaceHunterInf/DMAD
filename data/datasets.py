"""
Dataset loaders for public benchmarks.

Each loader returns ``(questions, labels)`` — a list of question strings
and a list of ground-truth label strings — formatted for the debate
pipeline.

Supported datasets
------------------
- **GSM8K** — grade-school math (HuggingFace ``openai/gsm8k``)
- **CommonsenseQA** — 5-way MCQA (``tau/commonsense_qa``)
- **HellaSwag** — 4-way sentence completion (``Rowan/hellaswag``)
- **MMLU** — 4-way MCQA (``cais/mmlu``, per subject)
- **Arithmetic** — synthetic arithmetic expressions (no download)
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

_GSM8K_ANS_RE = re.compile(r"#### (-?[\d.,]+)")


def _extract_gsm8k_answer(text: str) -> str | None:
    m = _GSM8K_ANS_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def load_gsm8k(
    split: str = "test",
    data_size: int | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Load the GSM8K dataset.

    Args:
        split: ``"train"`` or ``"test"``.
        data_size: If given, subsample to this many examples.
        cache_dir: HuggingFace cache directory.
        seed: Random seed for shuffling.

    Returns:
        ``(questions, labels)`` tuple.
    """
    ds = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)[split]
    df = pd.DataFrame(ds).sample(frac=1, random_state=seed).reset_index(drop=True)
    if data_size:
        df = df.head(data_size)

    questions, labels = [], []
    for _, row in df.iterrows():
        label = _extract_gsm8k_answer(row["answer"])
        if label is not None:
            questions.append(row["question"])
            labels.append(str(label))
    return questions, labels


# ---------------------------------------------------------------------------
# CommonsenseQA (CSQA)
# ---------------------------------------------------------------------------

def load_csqa(
    split: str = "validation",
    data_size: int | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Load CommonsenseQA (5-way MCQA).

    Args:
        split: ``"train"`` or ``"validation"``.
        data_size: Subsample size.
        cache_dir: HuggingFace cache directory.
        seed: Random seed.

    Returns:
        ``(questions, labels)`` tuple.
    """
    hf_split = "validation" if split == "test" else split
    ds = load_dataset("tau/commonsense_qa", cache_dir=cache_dir)[hf_split]
    df = pd.DataFrame(ds).sample(frac=1, random_state=seed).reset_index(drop=True)
    if data_size:
        df = df.head(data_size)

    template = "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n\n"
    questions, labels = [], []
    for _, row in df.iterrows():
        opts = row["choices"]["text"]
        if len(opts) != 5:
            continue
        question = template.format(row["question"], *opts)
        questions.append(question)
        labels.append(f"({row['answerKey']})")
    return questions, labels


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def load_hellaswag(
    split: str = "validation",
    data_size: int | None = None,
    cache_dir: str | None = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Load HellaSwag (4-way sentence completion).

    Args:
        split: ``"train"`` or ``"validation"``.
        data_size: Subsample size.
        cache_dir: HuggingFace cache directory.
        seed: Random seed.

    Returns:
        ``(questions, labels)`` tuple.
    """
    hf_split = "validation" if split == "test" else split
    ds = load_dataset("Rowan/hellaswag", cache_dir=cache_dir)[hf_split]
    df = pd.DataFrame(ds).sample(frac=1, random_state=seed).reset_index(drop=True)
    if data_size:
        df = df.head(data_size)

    letters = "ABCD"
    template = (
        'Can you choose the option that best follows:\n'
        '"{}"?\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    )
    questions, labels = [], []
    for _, row in df.iterrows():
        opts = row["endings"]
        if len(opts) != 4:
            continue
        question = template.format(row["ctx"], *opts)
        questions.append(question)
        labels.append(f"({letters[int(row['label'])]})")
    return questions, labels


# ---------------------------------------------------------------------------
# MMLU (by subject)
# ---------------------------------------------------------------------------

def load_mmlu(
    subject: str = "professional_medicine",
    split: str = "test",
    data_size: int | None = None,
    cache_dir: str | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Load an MMLU subject (4-way MCQA).

    Args:
        subject: MMLU subject name (e.g. ``"professional_medicine"``,
                 ``"formal_logic"``).
        split: ``"validation"`` or ``"test"``.
        data_size: Subsample size.
        cache_dir: HuggingFace cache directory.

    Returns:
        ``(questions, labels)`` tuple.
    """
    hf_split = "validation" if split == "train" else "test"
    ds = load_dataset("cais/mmlu", subject, cache_dir=cache_dir)[hf_split]
    df = pd.DataFrame(ds)
    if data_size:
        df = df.head(data_size)

    letters = "ABCD"
    template = "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n"
    questions, labels = [], []
    for _, row in df.iterrows():
        opts = row["choices"]
        if len(opts) != 4:
            continue
        question = template.format(row["question"], *opts)
        questions.append(question)
        labels.append(f"({letters[int(row['answer'])]})")
    return questions, labels


# ---------------------------------------------------------------------------
# Arithmetic (synthetic, no download required)
# ---------------------------------------------------------------------------

def load_arithmetic(
    split: str = "test",
    data_size: int = 300,
    num_params: int = 6,
    seed: int | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Generate synthetic arithmetic questions.

    Args:
        split: ``"train"`` or ``"test"`` (controls random seed).
        data_size: Number of questions.
        num_params: ``4`` for easy (``a+b*c-d``),
                    ``6`` for hard (``a+b*c+d-e÷f``).
        seed: Explicit random seed (overrides split-based default).

    Returns:
        ``(questions, labels)`` tuple.
    """
    if seed is None:
        seed = 0 if split == "train" else 1
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 30, size=num_params * data_size)

    questions, labels = [], []
    for i in range(0, num_params * data_size, num_params):
        if num_params == 4:
            a, b, c, d = x[i : i + 4]
            q = f"What is the result of {a}+{b}*{c}-{d}?"
            ans = int(a + b * c - d)
        else:
            a, b, c, d, e, f = x[i : i + 6]
            if f == 0:
                f = 1
            q = f"What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?"
            ans = a + b * c + d - e / f
        questions.append(q)
        labels.append(str(ans))
    return questions, labels
