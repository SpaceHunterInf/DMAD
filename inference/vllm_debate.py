"""
Parallel multi-agent debate (MAD) using vLLM for efficient batch inference.

This script implements vanilla MAD: each question is independently answered
by N agents, who then revise their answers over T rounds while observing
all other agents' previous responses.  The final answer is determined by
majority vote over the last round.

Usage:
    python vllm_debate.py --config configs/debate_llama.yaml \\
                          [--fold test] [--limit 100] [--seed 42]
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

# Ensure repo root is on the path so that ``utils`` and ``inference`` are
# importable regardless of the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.prompts import format_initial_round_prompt, format_multi_agent_prompt
from utils.data_utils import (
    load_debate_dataset,
    save_debate_results,
    chunk_list,
    calculate_accuracy,
)
from utils.eval_utils import check_answer_correct, get_majority_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_response(text: str, use_confidence: bool) -> Tuple[str, str, str]:
    """
    Parse model response to extract reasoning, answer, and confidence.

    Returns:
        (reasoning, answer, confidence) tuple.
    """
    # Answer
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        answer = "PARSE_ERROR"

    # Confidence
    if use_confidence and "<confidence>" in text and "</confidence>" in text:
        confidence = text.split("<confidence>")[-1].split("</confidence>")[0].strip()
    else:
        confidence = "N/A"

    # Reasoning
    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
    else:
        reasoning = text.strip()

    return reasoning, answer, confidence


# ---------------------------------------------------------------------------
# Core debate loop
# ---------------------------------------------------------------------------

def run_parallel_debates(
    examples: List[Dict],
    model: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    num_agents: int,
    num_rounds: int,
    use_confidence: bool,
    save_full_history: bool = True,
    save_prompts: bool = False,
    lora_request: LoRARequest = None,
) -> List[Dict]:
    """
    Run multi-agent debates on multiple examples in parallel.

    vLLM processes all ``(num_examples × num_agents)`` prompts in a single
    batch at each round, making this highly efficient.

    Args:
        examples: List of example dicts with ``question``, ``label``,
                  ``dataset``, ``fold`` keys.
        model: vLLM model instance.
        tokenizer: HuggingFace tokenizer.
        sampling_params: vLLM sampling parameters.
        num_agents: Number of debating agents.
        num_rounds: Number of debate rounds.
        use_confidence: Whether agents express confidence.
        save_full_history: Store complete per-round agent responses.
        save_prompts: Store the formatted prompts in the output.
        lora_request: Optional LoRA adapter request.

    Returns:
        List of result dictionaries (one per example).
    """
    num_examples = len(examples)
    print(f"\n{'='*80}")
    print(f"Starting parallel debates:")
    print(f"  - {num_examples} questions")
    print(f"  - {num_agents} agents per question")
    print(f"  - {num_rounds} rounds")
    print(f"  - Total prompts per round: {num_examples * num_agents}")
    print(f"{'='*80}\n")

    # History: examples_history[example_idx][agent_idx] = [resp_r0, ...]
    history = [[[] for _ in range(num_agents)] for _ in range(num_examples)]

    for round_idx in tqdm(range(num_rounds), desc="Debate Rounds"):
        prompt_batch: List[str] = []
        meta_batch: List[Dict] = []

        if round_idx == 0:
            for ex_i, ex in enumerate(examples):
                fmt = format_initial_round_prompt(
                    ex.copy(),
                    use_confidence=use_confidence,
                    chat_mode=True,
                    tokenizer=tokenizer,
                )
                for ag_i in range(num_agents):
                    prompt_batch.append(fmt["prompt"])
                    meta_batch.append(
                        {"example_idx": ex_i, "agent_idx": ag_i}
                    )
        else:
            for ex_i, ex in enumerate(examples):
                for ag_i in range(num_agents):
                    others = [
                        history[ex_i][j][-1]
                        for j in range(num_agents)
                        if j != ag_i
                    ]
                    own = history[ex_i][ag_i][-1]
                    fmt = format_multi_agent_prompt(
                        ex.copy(),
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

        # Batch generation
        print(f"  Generating {len(prompt_batch)} responses in parallel...")
        if lora_request is not None:
            raw = model.generate(prompt_batch, sampling_params, lora_request=lora_request)
        else:
            raw = model.generate(prompt_batch, sampling_params)

        for m, r in zip(meta_batch, raw):
            text = r.outputs[0].text
            reasoning, answer, confidence = parse_response(text, use_confidence)
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

    # Compile results
    results = []
    for ex_i, ex in enumerate(examples):
        result: Dict = {
            "fold": ex["fold"],
            "dataset": ex["dataset"],
            "question": ex["question"],
            "label": ex["label"],
            "num_agents": num_agents,
            "num_rounds": num_rounds,
        }
        round_results = []
        for r_i in range(num_rounds):
            round_resps = [history[ex_i][a][r_i] for a in range(num_agents)]
            all_answers = [rr["final_answer"] for rr in round_resps]
            majority = get_majority_answer(all_answers, ex["dataset"])
            rr_dict: Dict = {"round": r_i, "majority_answer": majority}
            if save_full_history:
                rr_dict["agents_responses"] = round_resps
            round_results.append(rr_dict)

        result["round_results"] = round_results
        result["final_majority_answer"] = round_results[-1]["majority_answer"]
        result["is_correct"] = check_answer_correct(
            result["final_majority_answer"], ex["label"], ex["dataset"]
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run parallel multi-agent debates with vLLM"
    )
    parser.add_argument("--config", type=str, default="configs/debate_llama.yaml")
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.input_csv:
        cfg["data"]["input_csv"] = args.input_csv
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size

    model_name = cfg["model"]["name"]
    num_agents = cfg["debate"]["num_agents"]
    num_rounds = cfg["debate"]["num_rounds"]
    use_confidence = cfg["debate"]["use_confidence"]
    input_csv = cfg["data"]["input_csv"]
    output_dir = cfg["data"]["output_dir"]
    batch_size = cfg["data"]["batch_size"]

    print(f"\n{'='*80}")
    print("Parallel Multi-Agent Debate System")
    print(f"{'='*80}")
    print(f"  Model: {model_name}")
    lora_path = cfg["model"].get("lora_path")
    if lora_path:
        print(f"  LoRA: {lora_path}")
    print(f"  Agents: {num_agents}  |  Rounds: {num_rounds}")
    print(f"  Confidence: {use_confidence}")
    print(f"  Seed: {args.seed}")
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
        max_lora_rank=cfg["model"].get("max_lora_rank", None),
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
        seed=args.seed,
    )

    # Data
    examples = load_debate_dataset(input_csv, fold=args.fold)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples")

    # Run
    all_results: List[Dict] = []
    for bi, batch in enumerate(chunk_list(examples, batch_size)):
        print(f"\nBatch {bi + 1}/{-(-len(examples) // batch_size)}")
        all_results.extend(
            run_parallel_debates(
                examples=batch,
                model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                num_agents=num_agents,
                num_rounds=num_rounds,
                use_confidence=use_confidence,
                save_full_history=cfg["logging"]["save_full_history"],
                save_prompts=cfg["logging"]["save_prompts"],
                lora_request=lora_request,
            )
        )

    # Report
    stats = calculate_accuracy(all_results)
    print(f"\nOverall: {stats['accuracy']:.2%}  ({stats['correct']}/{stats['total']})")
    for ds, ds_stats in stats["per_dataset"].items():
        print(f"  {ds}: {ds_stats['accuracy']:.2%}  ({ds_stats['correct']}/{ds_stats['total']})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_debate_results(all_results, os.path.join(output_dir, f"debate_results_{ts}.json"), fmt="json")
    save_debate_results(all_results, os.path.join(output_dir, f"debate_summary_{ts}.csv"), fmt="csv")
    with open(os.path.join(output_dir, f"debate_stats_{ts}.json"), "w") as f:
        json.dump({"config": cfg, "accuracy": stats, "timestamp": ts, "num_examples": len(examples)}, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
