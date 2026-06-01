"""
Data loading and saving utilities for multi-agent debate.

Handles CSV/JSON dataset I/O, result serialisation, batching,
and accuracy computation.
"""

import os
import json
from typing import Dict, List

import pandas as pd


def load_debate_dataset(csv_path: str, fold: str = None) -> List[Dict]:
    """
    Load a debate dataset from a CSV file.

    Expected columns: ``fold``, ``dataset``, ``question``, ``label``.

    Args:
        csv_path: Path to the CSV file.
        fold: Optional fold to filter by (e.g. "train", "test").
              If *None*, all rows are returned.

    Returns:
        List of example dictionaries.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["fold", "dataset", "question", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if fold is not None:
        df = df[df["fold"] == fold]

    return [
        {
            "question": str(row["question"]),
            "label": str(row["label"]),
            "dataset": str(row["dataset"]),
            "fold": str(row["fold"]),
        }
        for _, row in df.iterrows()
    ]


def save_debate_results(
    results: List[Dict], output_path: str, fmt: str = "csv"
) -> None:
    """
    Save debate results to a file.

    Args:
        results: List of result dictionaries.
        output_path: Destination file path.
        fmt: Output format — ``"csv"`` or ``"json"``.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if fmt == "csv":
        flat = []
        for r in results:
            row = {
                "fold": r.get("fold", ""),
                "dataset": r.get("dataset", ""),
                "question": r.get("question", ""),
                "label": r.get("label", ""),
                "final_majority_answer": r.get("final_majority_answer", ""),
                "is_correct": r.get("is_correct", False),
                "num_rounds": r.get("num_rounds", 0),
                "num_agents": r.get("num_agents", 0),
            }
            for ri, rd in enumerate(r.get("round_results", [])):
                row[f"round_{ri}_majority"] = rd.get("majority_answer", "")
            flat.append(row)
        pd.DataFrame(flat).to_csv(output_path, index=False)

    elif fmt == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    else:
        raise ValueError(f"Unsupported format: {fmt}")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split *lst* into chunks of at most *chunk_size* elements."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def calculate_accuracy(results: List[Dict]) -> Dict:
    """
    Compute accuracy metrics from debate results.

    Each result dictionary must contain an ``is_correct`` boolean field.

    Returns:
        Dictionary with ``total``, ``correct``, ``accuracy``, and
        ``per_dataset`` breakdown.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "per_dataset": {}}

    correct = sum(1 for r in results if r.get("is_correct", False))

    per_dataset: Dict[str, Dict] = {}
    for r in results:
        ds = r.get("dataset", "unknown")
        if ds not in per_dataset:
            per_dataset[ds] = {"total": 0, "correct": 0}
        per_dataset[ds]["total"] += 1
        if r.get("is_correct", False):
            per_dataset[ds]["correct"] += 1

    for stats in per_dataset.values():
        stats["accuracy"] = stats["correct"] / stats["total"]

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_dataset": per_dataset,
    }
