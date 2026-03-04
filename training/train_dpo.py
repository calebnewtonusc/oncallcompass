"""
OncallCompass Stage 3: DPO Alignment.

Direct Preference Optimization on (fast-resolution path, slow-resolution path) pairs.
Teaches the model to prefer:
    - Hypothesis ranking over hypothesis listing
    - Topology-aware reasoning over metric correlation
    - Prevention-focused postmortems over descriptive ones
    - Efficient investigation paths over exhaustive listing

Usage:
    python training/train_dpo.py \
        --model_path checkpoints/rl \
        --data_path data/dpo_pairs.jsonl \
        --output_dir checkpoints/final
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

load_dotenv()

SYSTEM_PROMPT = """You are OncallCompass. Given incident signals, rank root cause hypotheses and sequence investigation steps in JSON."""


def format_preference_example(example: dict[str, Any]) -> dict[str, Any]:
    """Format a DPO pair into the prompt/chosen/rejected format."""
    context = json.dumps(
        {
            "alerts": example.get("alerts", []),
            "logs": example.get("logs", ""),
            "metrics": example.get("metrics", {}),
            "context": example.get("context", {}),
        },
        indent=2,
    )

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    fast_path = example.get("fast_path", {})
    slow_path = example.get("slow_path", {})

    chosen = json.dumps(
        {
            "ranked_hypotheses": fast_path.get("ranked_hypotheses", []),
            "investigation_steps": fast_path.get("investigation_steps", []),
        }
    )

    rejected = json.dumps(
        {
            "ranked_hypotheses": slow_path.get("ranked_hypotheses", []),
            "investigation_steps": slow_path.get("investigation_steps", []),
        }
    )

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass DPO training")
    parser.add_argument(
        "--model_path",
        default="checkpoints/rl",
        help="Path to PEFT adapter-only RL checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model name or path (required when model_path is a PEFT adapter). "
        "Defaults to model_path when not provided for backwards compatibility.",
    )
    parser.add_argument("--data_path", default="data/dpo_pairs.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/final")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base_model_path = args.base_model or args.model_path
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.base_model:
        # RL checkpoint is a PEFT adapter-only directory — wrap with PeftModel.
        print(f"Loading PEFT adapter from {args.model_path}")
        model = PeftModel.from_pretrained(base, args.model_path, is_trainable=True)
    else:
        # Backwards-compatible path: model_path is a full merged model.
        model = base

    print(f"Loading DPO dataset from {args.data_path}")
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = raw_dataset.map(
        format_preference_example,
        remove_columns=raw_dataset.column_names,
    )
    print(f"DPO dataset: {len(dataset)} preference pairs")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="oncallcompass-dpo",
    )

    # Load a frozen reference model so DPO has a proper KL anchor.
    # Using ref_model=None would make the policy serve as its own reference,
    # eliminating the KL constraint and destabilizing training.
    # Always load the base model (without PEFT) as the frozen reference so
    # that the KL term measures divergence from the pre-RL base distribution.
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map=None
    )
    ref_model.eval()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving aligned model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO training complete.")


if __name__ == "__main__":
    main()
