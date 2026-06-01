"""
Diversity-aware multi-agent debate starting from pre-initialized outputs.

Implements the diversity-aware initialization mechanism described in Section 4.2
of the paper.  Given a pool of N candidate initial responses per question
(pre-generated), this script:

1. Selects a subset of K agents according to a diversity strategy:
   - ``low_diversity``:  greedily picks the most semantically *similar* subset
   - ``random``:         uniformly samples K agents
   - ``high_diversity``: greedily picks the most semantically *diverse* subset
2. Starts the debate from round 1 using the selected initial responses.
3. Continues for T-1 additional rounds of multi-agent revision.

Usage:
    python vllm_debate_from_init.py \\
        --config configs/debate_llama.yaml \\
        --input_json data/initialized_outputs.json \\
        --diversity_type high_diversity \\
        [--fold test] [--limit 100]
"""

import os
import sys
import json
import yaml
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# Ensure repo root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.prompts import format_multi_agent_prompt
from utils.data_utils import save_debate_results, chunk_list, calculate_accuracy
from utils.eval_utils import check_answer_correct, get_majority_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_initialized_outputs(json_path: str) -> List[Dict]:
    """
    Load pre-generated initial responses from a JSON file.

    Expected format::

        [
            {
                "index": 0,
                "dataset": "gsm8k",
                "fold": "test",
                "question": "...",
                "label": "...",
                "outputs": [
                    {"reasoning": "...", "answer": "...", "confidence": 7},
                    ...
                ],
                "low_diversity": [0, 2, 3, 4, 6],
                "random": [0, 1, 2, 3, 5],
                "high_diversity": [1, 5, 7, 8, 9]
            },
            ...
        ]

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of example dictionaries.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples with pre-initialized outputs")

    required = [
        "index", "dataset", "fold", "question", "label",
        "outputs", "low_diversity", "random", "high_diversity",
    ]
    for idx, ex in enumerate(data):
        missing = [k for k in required if k not in ex]
        if missing:
            raise ValueError(f"Example {idx} missing required keys: {missing}")

    return data


def parse_response(text: str, use_confidence: bool) -> Tuple[str, str, str]:
    """Parse model output into (reasoning, answer, confidence)."""
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        answer = "PARSE_ERROR"

    if use_confidence and "<confidence>" in text and "</confidence>" in text:
        confidence = (
            text.split("<confidence>")[-1].split("</confidence>")[0].strip()
        )
    else:
        confidence = "N/A"

    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = (
            text.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
        )
    else:
        reasoning = text.strip()

    return reasoning, answer, confidence


# ---------------------------------------------------------------------------
# Core debate loop (from pre-initialized outputs)
# ---------------------------------------------------------------------------

def run_debates_from_initialization(
    examples_with_outputs: List[Dict],
    diversity_type: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    num_rounds: int,
    use_confidence: bool,
    save_full_history: bool = True,
    save_prompts: bool = False,
    lora_request: LoRARequest = None,
) -> List[Dict]:
    """
    Run debates starting from pre-initialized agent outputs.

    Round 0 is populated from the pre-generated pool according to
    *diversity_type*.  Subsequent rounds use vLLM batch inference.

    Args:
        examples_with_outputs: Examples with pre-initialized candidate pools.
        diversity_type: One of ``"low_diversity"``, ``"random"``,
                        ``"high_diversity"``.
        model: vLLM model instance.
        tokenizer: HuggingFace tokenizer.
        sampling_params: vLLM sampling parameters.
        num_rounds: Total number of rounds (including round 0).
        use_confidence: Whether to use confidence tracking.
        save_full_history: Store full per-round agent responses.
        save_prompts: Store formatted prompts in output.
        lora_request: Optional LoRA adapter request.

    Returns:
        List of result dictionaries (one per example).
    """
    valid_types = ("low_diversity", "random", "high_diversity")
    if diversity_type not in valid_types:
        raise ValueError(
            f"Invalid diversity_type: {diversity_type}. "
            f"Must be one of: {valid_types}"
        )

    num_examples = len(examples_with_outputs)
    print(f"\n{'='*80}")
    print(f"Starting debates from pre-initialized outputs:")
    print(f"  - {num_examples} questions")
    print(f"  - Diversity type: {diversity_type}")
    print(f"  - Total rounds: {num_rounds} (starting from round 1)")
    print(f"  - Use confidence: {use_confidence}")
    print(f"{'='*80}\n")

    # ---- Round 0: populate from pre-generated outputs ---
    history: List[List[List[Dict]]] = []  # [ex][agent][round]

    for ex_data in examples_with_outputs:
        agent_indices = ex_data[diversity_type]
        ex_hist: List[List[Dict]] = []
        for ag_i, out_i in enumerate(agent_indices):
            output = ex_data["outputs"][out_i]
            resp = {
                "agent_id": ag_i + 1,
                "reasoning": output["reasoning"],
                "final_answer": output["answer"],
                "confidence_level": str(output.get("confidence", "N/A")),
                "completion": output["reasoning"],
                "initialized_from_output_idx": out_i,
            }
            ex_hist.append([resp])
        history.append(ex_hist)

    # ---- Rounds 1 … T-1: vLLM batch generation ---
    for round_idx in tqdm(range(1, num_rounds), desc="Debate Rounds"):
        prompt_batch: List[str] = []
        meta_batch: List[Dict] = []

        for ex_i, ex_data in enumerate(examples_with_outputs):
            example = {
                "question": ex_data["question"],
                "label": ex_data["label"],
                "dataset": ex_data["dataset"],
                "fold": ex_data["fold"],
            }
            n_agents = len(history[ex_i])

            for ag_i in range(n_agents):
                others = [
                    history[ex_i][j][-1].copy()
                    for j in range(n_agents)
                    if j != ag_i
                ]
                # Gracefully handle parse errors
                for o in others:
                    if o["final_answer"] == "PARSE_ERROR":
                        o["final_answer"] = "[No answer provided]"
                        o["reasoning"] += (
                            "\n[Note: This agent did not provide a valid "
                            "answer in the required format]"
                        )

                own = history[ex_i][ag_i][-1].copy()
                if own["final_answer"] == "PARSE_ERROR":
                    own["final_answer"] = "[No answer provided]"
                    own["reasoning"] += (
                        "\n[Note: You did not provide a valid answer in "
                        "the required format in the previous round]"
                    )

                fmt = format_multi_agent_prompt(
                    example.copy(),
                    other_agent_responses=others,
                    own_response=own,
                    use_confidence=use_confidence,
                    chat_mode=True,
                    tokenizer=tokenizer,
                )
                prompt_batch.append(fmt["prompt"])
                meta_batch.append(
                    {"example_idx": ex_i, "agent_idx": ag_i}
                )

        print(
            f"  Generating {len(prompt_batch)} responses in parallel "
            f"for round {round_idx}..."
        )
        if lora_request is not None:
            raw = model.generate(
                prompt_batch, sampling_params, lora_request=lora_request
            )
        else:
            raw = model.generate(prompt_batch, sampling_params)

        for m, r in zip(meta_batch, raw):
            text = r.outputs[0].text
            reasoning, answer, confidence = parse_response(
                text, use_confidence
            )
            resp = {
                "agent_id": m["agent_idx"] + 1,
                "reasoning": reasoning,
                "final_answer": answer,
                "confidence_level": confidence,
                "completion": text,
            }
            if save_prompts:
                resp["prompt"] = prompt_batch[meta_batch.index(m)]
            history[m["example_idx"]][m["agent_idx"]].append(resp)

    # ---- Compile results ---
    results: List[Dict] = []
    for ex_i, ex_data in enumerate(examples_with_outputs):
        n_agents = len(history[ex_i])
        n_rounds = len(history[ex_i][0])

        result: Dict = {
            "index": ex_data["index"],
            "fold": ex_data["fold"],
            "dataset": ex_data["dataset"],
            "question": ex_data["question"],
            "label": ex_data["label"],
            "num_agents": n_agents,
            "num_rounds": num_rounds,
            "diversity_type": diversity_type,
            "initialized_from_indices": ex_data[diversity_type],
        }

        round_results = []
        for ri in range(n_rounds):
            agents = [history[ex_i][a][ri] for a in range(n_agents)]
            all_answers = [a["final_answer"] for a in agents]
            majority = get_majority_answer(all_answers, ex_data["dataset"])
            rr: Dict = {
                "round": ri,
                "majority_answer": majority,
                "is_correct": check_answer_correct(
                    majority, ex_data["label"], ex_data["dataset"]
                ),
            }
            if save_full_history:
                rr["agents_responses"] = [
                    {
                        "agent_id": a["agent_id"],
                        "reasoning": a["reasoning"],
                        "final_answer": a["final_answer"],
                        "confidence_level": a["confidence_level"],
                        "completion": a["completion"],
                    }
                    for a in agents
                ]
            round_results.append(rr)

        result["round_results"] = round_results
        result["final_majority_answer"] = round_results[-1]["majority_answer"]
        result["is_correct"] = check_answer_correct(
            result["final_majority_answer"],
            ex_data["label"],
            ex_data["dataset"],
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run debates from pre-initialized (diversity-aware) outputs"
    )
    parser.add_argument(
        "--config", type=str, default="configs/debate_llama.yaml"
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--diversity_type",
        type=str,
        required=True,
        choices=["low_diversity", "random", "high_diversity"],
    )
    parser.add_argument("--fold", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size

    model_name = cfg["model"]["name"]
    num_rounds = cfg["debate"]["num_rounds"]
    use_confidence = cfg["debate"]["use_confidence"]
    output_dir = cfg["data"]["output_dir"]
    batch_size = cfg["data"]["batch_size"]

    print(f"\n{'='*80}")
    print("Multi-Agent Debate from Pre-Initialized Outputs")
    print(f"{'='*80}")
    print(f"  Model: {model_name}")
    lora_path = cfg["model"].get("lora_path")
    if lora_path:
        print(f"  LoRA: {lora_path}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Confidence: {use_confidence}")
    print(f"  Diversity type: {args.diversity_type}")
    print(f"  Fold: {args.fold}")
    print(f"{'='*80}\n")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    enable_lora = lora_path is not None
    model = LLM(
        model=model_name,
        dtype=cfg["model"]["dtype"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        tensor_parallel_size=cfg["model"].get("tensor_parallel_size", 1),
        enable_lora=enable_lora,
    )
    lora_request = (
        LoRARequest("debate_lora", 1, lora_path) if enable_lora else None
    )

    stop_token = "</confidence>" if use_confidence else "</answer>"
    sampling_params = SamplingParams(
        temperature=cfg["sampling"]["temperature"],
        max_tokens=cfg["sampling"]["max_tokens"],
        top_p=cfg["sampling"]["top_p"],
        top_k=cfg["sampling"]["top_k"],
        stop=stop_token,
        include_stop_str_in_output=True,
    )

    # Data
    examples = load_initialized_outputs(args.input_json)
    examples = [ex for ex in examples if ex["fold"] == args.fold]
    print(f"Filtered to {len(examples)} examples with fold='{args.fold}'")
    if not examples:
        print("ERROR: No examples found for this fold.")
        return
    if args.limit:
        examples = examples[: args.limit]

    # Run
    all_results: List[Dict] = []
    batches = chunk_list(examples, batch_size)
    for bi, batch in enumerate(batches):
        print(f"\nBatch {bi + 1}/{len(batches)}")
        all_results.extend(
            run_debates_from_initialization(
                examples_with_outputs=batch,
                diversity_type=args.diversity_type,
                model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                num_rounds=num_rounds,
                use_confidence=use_confidence,
                save_full_history=cfg["logging"]["save_full_history"],
                save_prompts=cfg["logging"]["save_prompts"],
                lora_request=lora_request,
            )
        )

    # Report
    stats = calculate_accuracy(all_results)
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(
        f"Overall: {stats['accuracy']:.2%}  "
        f"({stats['correct']}/{stats['total']})"
    )
    for ds, ds_s in stats["per_dataset"].items():
        print(f"  {ds}: {ds_s['accuracy']:.2%}  ({ds_s['correct']}/{ds_s['total']})")

    print("\nPer-Round Accuracy:")
    for ri in range(num_rounds):
        rc = sum(
            1 for r in all_results if r["round_results"][ri]["is_correct"]
        )
        print(f"  Round {ri}: {rc / len(all_results):.2%}  ({rc}/{len(all_results)})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.diversity_type

    save_debate_results(
        all_results,
        os.path.join(output_dir, f"debate_results_{tag}_{ts}.json"),
        fmt="json",
    )
    save_debate_results(
        all_results,
        os.path.join(output_dir, f"debate_summary_{tag}_{ts}.csv"),
        fmt="csv",
    )
    with open(os.path.join(output_dir, f"debate_stats_{tag}_{ts}.json"), "w") as f:
        json.dump(
            {
                "config": cfg,
                "diversity_type": tag,
                "fold": args.fold,
                "input_json": args.input_json,
                "accuracy": stats,
                "timestamp": ts,
                "num_examples": len(examples),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
