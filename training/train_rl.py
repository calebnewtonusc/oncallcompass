"""
OncallCompass Stage 2: RL Training with MTTR Reward.

Uses GRPO (Group Relative Policy Optimization) to fine-tune the SFT model
with a reward signal based on time-to-correct-root-cause in simulated incident
drills.

Reward function:
    R = w1 * (root_cause_rank == 1)
      + w2 * (1 / root_cause_rank)         # partial credit
      + w3 * (baseline_time - model_time) / baseline_time  # MTTR reduction
      - w4 * wasted_steps                  # penalize wrong pivots

The model is penalized for surfacing the correct root cause anywhere other
than the first position, and for investigation steps that don't move toward
the correct root cause.

Usage:
    python training/train_rl.py \
        --model_path checkpoints/sft \
        --output_dir checkpoints/rl \
        --drill_set data/drills/compassbench_v1.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

load_dotenv()

# Reward weights
W_RANK_FIRST = 2.0  # reward for correct root cause in position 1
W_RANK_PARTIAL = 0.5  # partial credit weight (1/rank)
W_MTTR_REDUCTION = 1.5  # reward for MTTR reduction vs. baseline
W_WASTED_STEPS = 0.3  # penalty per wasted investigation step


def compute_mttr_reward(
    response: dict[str, Any],
    ground_truth: dict[str, Any],
    baseline_steps: int,
) -> float:
    """
    Compute the MTTR reward for a single model response.

    Args:
        response: Model output with ranked_hypotheses and investigation_steps
        ground_truth: Known correct root cause and expert step sequence
        baseline_steps: Number of steps a baseline (unguided) investigation takes

    Returns:
        Scalar reward value
    """
    reward = 0.0
    correct_cause = ground_truth.get("root_cause", "")
    if not correct_cause:
        return 0.0
    hypotheses = response.get("ranked_hypotheses", [])

    # Find rank of correct root cause
    correct_rank = None
    for i, hyp in enumerate(hypotheses):
        if correct_cause.lower() in hyp.get("hypothesis", "").lower():
            correct_rank = i + 1
            break

    if correct_rank is None:
        # Correct root cause not surfaced at all — large penalty
        return -2.0

    # Reward for rank position
    reward += W_RANK_FIRST * (1.0 if correct_rank == 1 else 0.0)
    reward += W_RANK_PARTIAL * (1.0 / correct_rank)

    # MTTR reward based on investigation step efficiency.
    # Only award the MTTR bonus when the correct root cause is ranked first;
    # otherwise a short but incorrect response could earn a positive MTTR reward.
    model_steps = len(response.get("investigation_steps", []))
    expert_steps = len(ground_truth.get("expert_steps", []))

    if baseline_steps > 0 and correct_rank == 1:
        mttr_ratio = (baseline_steps - model_steps) / baseline_steps
        reward += W_MTTR_REDUCTION * min(max(mttr_ratio, -1.0), 1.0)

    if expert_steps > 0:
        wasted = max(0, model_steps - expert_steps)
        reward -= W_WASTED_STEPS * wasted

    return float(reward)


def build_drill_prompt(drill: dict[str, Any]) -> str:
    """Format a drill scenario into a model prompt."""
    context = json.dumps(
        {
            "alerts": drill.get("alerts", []),
            "logs": drill.get("logs", ""),
            "metrics": drill.get("metrics", {}),
            "context": drill.get("context", {}),
        },
        indent=2,
    )
    return (
        "<|im_start|>system\n"
        "You are OncallCompass. Respond with ranked hypotheses and investigation steps in JSON.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_drills(drill_set_path: str) -> list[dict[str, Any]]:
    """Load drill scenarios from JSONL."""
    drills = []
    with open(drill_set_path) as f:
        for line in f:
            line = line.strip()
            if line:
                drills.append(json.loads(line))
    return drills


def build_rl_dataset(drills: list[dict[str, Any]]) -> Dataset:
    """
    Convert drill scenarios into a HuggingFace Dataset for GRPOTrainer.

    Each example needs:
    - 'prompt': formatted prompt string
    - 'ground_truth': dict with root_cause, expert_steps, baseline_steps
    """
    # Shuffle so easy/hard drills are not grouped by source order, which could
    # bias early training batches and slow convergence.
    drills = list(drills)
    random.shuffle(drills)
    records = []
    for drill in drills:
        records.append(
            {
                "prompt": build_drill_prompt(drill),
                "ground_truth": drill.get("ground_truth", {}),
                "baseline_steps": drill.get("ground_truth", {}).get(
                    "baseline_steps", 8
                ),
            }
        )
    return Dataset.from_list(records)


def reward_fn(
    prompts: list[str], completions: list[list[str]], **kwargs
) -> list[float]:
    """
    GRPO-compatible reward function.

    Args:
        prompts: List of input prompts (passed by GRPOTrainer).
        completions: List of completion groups — each element is a list of
                     num_generations strings for one prompt.
        **kwargs: Extra dataset columns passed through by GRPOTrainer,
                  including 'ground_truth' and 'baseline_steps'.

    Returns:
        List of scalar rewards, one per completion (length =
        len(prompts) * num_generations).
    """
    # GRPOTrainer passes completions as list[list[str]] — one inner list per
    # prompt, with num_generations completions each.  Flatten to a 1-D list
    # and expand the per-prompt metadata to match.
    flat_completions: list[str] = [c for group in completions for c in group]
    num_generations = len(completions[0]) if completions else 1

    ground_truths_raw = kwargs.get("ground_truth", [])
    baseline_steps_raw = kwargs.get("baseline_steps", [])

    # Expand per-prompt metadata to align with flattened completions.
    ground_truths = [gt for gt in ground_truths_raw for _ in range(num_generations)]
    baseline_steps_list = [
        bs for bs in baseline_steps_raw for _ in range(num_generations)
    ]

    rewards = []
    for i, completion in enumerate(flat_completions):
        gt = ground_truths[i] if i < len(ground_truths) else {}
        baseline = baseline_steps_list[i] if i < len(baseline_steps_list) else 8

        # Parse JSON response
        try:
            text = completion.strip()
            response_dict = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON block
            import re

            m = re.search(r"\{.*\}", completion, re.DOTALL)
            if m:
                try:
                    response_dict = json.loads(m.group())
                except json.JSONDecodeError:
                    response_dict = {}
            else:
                response_dict = {}

        reward = compute_mttr_reward(response_dict, gt, baseline)
        rewards.append(reward)

    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass RL training with GRPO")
    parser.add_argument("--model_path", default="checkpoints/sft")
    parser.add_argument("--output_dir", default="checkpoints/rl")
    parser.add_argument("--drill_set", default="data/drills/compassbench_v1.jsonl")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of completions per prompt (GRPO group size)",
    )
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument(
        "--beta", type=float, default=0.01, help="KL penalty coefficient"
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.drill_set).exists():
        print(
            f"Drill set not found at {args.drill_set}. Generating synthetic drills..."
        )
        Path(args.drill_set).parent.mkdir(parents=True, exist_ok=True)
        with open(args.drill_set, "w"):
            pass
        print("Warning: empty drill set. Run synthesis/incident_synthesizer.py first.")
        return

    drills = load_drills(args.drill_set)
    print(f"Loaded {len(drills)} drill scenarios")

    train_dataset = build_rl_dataset(drills)
    if len(train_dataset) == 0:
        print("No training examples. Exiting.")
        return

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    sft_checkpoint_path = args.model_path
    model = PeftModel.from_pretrained(
        base_model, sft_checkpoint_path, is_trainable=True
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        num_generations=args.num_generations,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=100,
        save_total_limit=2,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving RL model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("RL training complete.")


if __name__ == "__main__":
    main()
