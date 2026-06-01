"""
GRPO Training — Stage 1: Confidence Expression Calibration.

Trains a language model (via LoRA) to express well-calibrated confidence
scores alongside its answers using Group Relative Policy Optimisation
(GRPO).  This corresponds to the *confidence expression* stage described
in Section 4.1 / Appendix B.1 of the paper.

Reward signal:
    R = correctness + confidence_calibration + length

Usage:
    python grpo_confidence.py --config configs/grpo_confidence.yaml \\
                              [--dev_mode] [--resume_from_checkpoint latest]

    # Multi-GPU with Accelerate:
    accelerate launch grpo_confidence.py --config configs/grpo_confidence.yaml
"""

import os
import sys
import json
import math
import yaml
import random
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np

# Ensure repo root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from utils.eval_utils import (
    extract_xml_answer,
    extract_xml_confidence,
    gsm8k_match,
    multi_choice_match,
    exact_match,
)


class ConfidenceGRPOTrainer:
    """
    GRPO trainer for confidence calibration (Stage 1).

    The model learns to:
    1. Answer questions correctly.
    2. Express calibrated confidence (high when correct, low when wrong).
    3. Produce outputs of reasonable length.
    """

    def __init__(
        self,
        config_path: str,
        dev_mode: bool = False,
        resume_from_checkpoint: str = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dev_mode = dev_mode

        # Checkpoint resumption (CLI overrides config)
        self.resume_from_checkpoint = resume_from_checkpoint
        if self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = self.config["training"].get(
                "resume_from_checkpoint"
            )

        set_seed(self.config["data"]["seed"])

        # Distributed training detection
        self.is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # W&B (rank 0 only)
        if self.config["wandb"]["enabled"] and (
            not self.is_distributed or self.local_rank == 0
        ):
            wandb.init(
                project=self.config["wandb"]["project"],
                name=self.config["wandb"]["name"],
                entity=self.config["wandb"].get("entity"),
                config=self.config,
            )

        # Paths
        self.output_dir = Path(self.config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ablation flag
        self.use_confidence = self.config["training"].get(
            "use_confidence", True
        )

        # Reward hyper-parameters
        self.correctness_scale = self.config["training"].get(
            "correctness_scale", 10.0
        )
        self.confidence_scale = self.config["training"].get(
            "confidence_scale", 5.0
        )
        self.wrong_format_penalty = self.config["training"].get(
            "wrong_format_penalty", -30.0
        )
        self.length_scale = self.config["training"].get("length_scale", 2.0)
        self.length_lower_bound = self.config["training"].get(
            "length_lower_bound", 256
        )
        self.length_upper_bound = self.config["training"].get(
            "length_upper_bound", 512
        )

        # Logging
        self.num_samples_to_log = self.config["training"].get(
            "num_samples_to_log", 5
        )
        self._reward_call_count = 0

        # Setup
        self._setup_model_and_tokenizer()
        self._load_data()
        self._setup_trainer()

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    def _setup_model_and_tokenizer(self):
        model_cfg = self.config["model"]
        lora_cfg = self.config["lora"]

        is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
        device_map = None if is_dist else model_cfg.get("device_map", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(model_cfg["torch_dtype"], torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _load_data(self):
        data_cfg = self.config["data"]
        data_file = Path(data_cfg["train_file"])
        if not data_file.is_absolute():
            data_file = Path(__file__).parent / data_file

        with open(data_file, "r") as f:
            data = json.load(f)

        # Split
        if "fold" in data[0]:
            train_data = [
                d
                for d in data
                if d["fold"] == data_cfg.get("train_split", "train")
            ]
        else:
            idx = int(len(data) * 0.9)
            train_data = data[:idx]

        if data_cfg.get("shuffle", True):
            random.seed(data_cfg["seed"])
            random.shuffle(train_data)

        self.train_dataset = Dataset.from_list(train_data)
        if self.dev_mode:
            self.train_dataset = self.train_dataset.select(
                range(min(50, len(self.train_dataset)))
            )
        print(f"Train dataset: {len(self.train_dataset)} examples")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    def _setup_trainer(self):
        tc = self.config["training"]
        is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1

        max_steps = tc.get("max_steps")
        if max_steps and max_steps > 0:
            n_epochs = None
        else:
            n_epochs = tc["num_train_epochs"]
            max_steps = -1

        grpo_cfg = GRPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=n_epochs if n_epochs is not None else 1,
            max_steps=max_steps,
            per_device_train_batch_size=tc["per_device_train_batch_size"],
            per_device_eval_batch_size=tc["per_device_eval_batch_size"],
            gradient_accumulation_steps=tc["gradient_accumulation_steps"],
            learning_rate=tc["learning_rate"],
            warmup_steps=tc["warmup_steps"],
            max_grad_norm=tc["max_grad_norm"],
            weight_decay=tc["weight_decay"],
            logging_steps=tc["logging_steps"],
            save_steps=tc["save_steps"],
            save_total_limit=tc["save_total_limit"],
            fp16=tc.get("fp16", False),
            bf16=tc.get("bf16", True),
            report_to="wandb" if self.config["wandb"]["enabled"] else "none",
            run_name=self.config["wandb"]["name"],
            seed=self.config["data"]["seed"],
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=(
                {"use_reentrant": False} if is_dist else None
            ),
            # GRPO-specific
            temperature=tc.get("temperature", 1.0),
            top_p=tc.get("top_p", 1.0),
            max_completion_length=tc.get("max_completion_length", 512),
            use_vllm=tc.get("use_vllm", False),
            vllm_gpu_memory_utilization=tc.get(
                "vllm_gpu_memory_utilization", 0.8
            ),
            vllm_mode=tc.get("vllm_mode", "colocate"),
            num_generations=tc.get("num_generations", 4),
            beta=tc.get("beta", 0.02),
        )

        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_cfg,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self._reward_function,
        )

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    def _correctness_reward(
        self, prediction: str, ground_truth: str, dataset: str
    ) -> float:
        dl = dataset.lower()
        if "gsm8k" in dl or "arithmetic" in dl:
            ok = gsm8k_match(prediction, ground_truth)
        elif any(n in dl for n in ["mmlu", "hellaswag", "csqa", "gpqa", "arc"]):
            ok = multi_choice_match(prediction, ground_truth)
        else:
            ok = exact_match(prediction, ground_truth)
        return self.correctness_scale if ok else 0.0

    def _confidence_calibration_reward(
        self, confidence: float, is_correct: bool
    ) -> float:
        if not self.use_confidence:
            return 0.0
        if confidence is None or confidence < 0 or confidence > 10:
            return 0.0

        p = min(0.999, max(0.001, confidence / 10.0))
        score = math.log(p) if is_correct else math.log(1 - p)

        lo, hi = math.log(0.001), math.log(0.999)
        normalised = (score - lo) / (hi - lo)
        return self.confidence_scale * normalised

    def _length_reward(self, completion: str) -> float:
        n = len(self.tokenizer.encode(completion, add_special_tokens=False))
        if n < self.length_lower_bound:
            return self.length_scale * (n / self.length_lower_bound)
        if n <= self.length_upper_bound:
            return self.length_scale
        excess = n - self.length_upper_bound
        return -self.length_scale * (excess / 100.0)

    # ------------------------------------------------------------------
    # Combined reward
    # ------------------------------------------------------------------
    def _reward_function(
        self, prompts: List[str], completions: List[str], **kwargs
    ) -> List[float]:
        labels = kwargs.get("label", [])
        datasets = kwargs.get("dataset", [])

        rewards: List[float] = []
        for completion, label, dataset in zip(completions, labels, datasets):
            # Parse
            try:
                parsed_answer = extract_xml_answer(completion)
            except Exception:
                parsed_answer = None
            parsed_confidence = None
            if self.use_confidence:
                try:
                    parsed_confidence = extract_xml_confidence(completion)
                except Exception:
                    pass

            # Wrong format → penalty
            if not parsed_answer:
                rewards.append(self.wrong_format_penalty)
                continue
            if self.use_confidence and parsed_confidence is None:
                rewards.append(self.wrong_format_penalty)
                continue

            corr = self._correctness_reward(parsed_answer, label, dataset)
            conf = self._confidence_calibration_reward(
                parsed_confidence, corr > 0
            )
            length = self._length_reward(completion)
            rewards.append(corr + conf + length)

        self._reward_call_count += 1
        return rewards

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _get_latest_checkpoint(self) -> str:
        ckpts = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    ckpts.append((step, item))
                except (IndexError, ValueError):
                    continue
        if not ckpts:
            raise ValueError(f"No checkpoints in {self.output_dir}")
        ckpts.sort(key=lambda x: x[0])
        return str(ckpts[-1][1])

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self):
        print("\nStarting GRPO training (confidence calibration)...")

        resume = None
        if self.resume_from_checkpoint:
            resume = (
                self._get_latest_checkpoint()
                if self.resume_from_checkpoint == "latest"
                else self.resume_from_checkpoint
            )
            print(f"Resuming from: {resume}")

        self.trainer.train(resume_from_checkpoint=resume)

        final = self.output_dir / "final_model"
        self.model.save_pretrained(str(final))
        self.tokenizer.save_pretrained(str(final))
        print(f"\nTraining complete — LoRA adapter saved to {final}")

        if self.config["wandb"]["enabled"] and (
            not self.is_distributed or self.local_rank == 0
        ):
            wandb.finish()


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training — Stage 1: Confidence Expression"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    trainer = ConfidenceGRPOTrainer(
        args.config, args.dev_mode, args.resume_from_checkpoint
    )
    trainer.train()


if __name__ == "__main__":
    main()
