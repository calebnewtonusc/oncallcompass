"""
OncallCompass Stage 1: Supervised Fine-Tuning.

Trains Qwen/Qwen2.5-7B-Coder-Instruct on incident-to-resolution triples:
    (alert signals + stack context, ranked hypotheses, resolution trace)

Each example teaches the model to:
    1. Parse alert signals and infer likely root cause categories
    2. Rank hypotheses by likelihood given the stack and alert pattern
    3. Sequence investigation steps in expert order
    4. Generate structured postmortem drafts

Usage:
    deepspeed --num_gpus=4 training/train.py \
        --model_name Qwen/Qwen2.5-7B-Coder-Instruct \
        --data_path data/sft_pairs.jsonl \
        --output_dir checkpoints/sft \
        --deepspeed training/configs/deepspeed_zero3.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

load_dotenv()

SYSTEM_PROMPT = """You are OncallCompass, a specialized incident response AI trained on 300k incident-to-resolution traces.

Given alert signals and stack context for an active incident, you:
1. Rank root cause hypotheses by likelihood — the most probable cause always comes first
2. Sequence investigation steps in the order a senior SRE would follow them
3. Generate structured, prevention-focused postmortem drafts

You reason like the senior engineer who has seen this failure pattern before, not like a search engine listing possibilities.

Always respond in valid JSON matching the OncallCompass output schema."""


def build_prompt(example: dict[str, Any]) -> str:
    """Format an SFT example into a chat template prompt."""
    user_content = json.dumps(
        {
            "alerts": example.get("alerts", []),
            "logs": example.get("logs", ""),
            "metrics": example.get("metrics", {}),
            "context": example.get("context", {}),
        },
        indent=2,
    )

    assistant_content = json.dumps(
        {
            "ranked_hypotheses": example.get("ranked_hypotheses", []),
            "investigation_steps": example.get("investigation_steps", []),
            "postmortem_draft": example.get("postmortem_draft", {}),
        },
        indent=2,
    )

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )


def load_sft_dataset(data_path: str) -> Dataset:
    """Load and format the SFT dataset from a JSONL file."""
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        lambda ex: {"text": build_prompt(ex)},
        remove_columns=dataset.column_names,
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass SFT training")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Coder-Instruct")
    parser.add_argument("--data_path", default="data/sft_pairs.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/sft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    # Use a dedicated pad token to avoid padding positions being treated as EOS.
    # SFTTrainer will compute loss on completion tokens only (DataCollatorForCompletionOnlyLM),
    # so padding tokens must not overlap with EOS or the loss mask gets corrupted.
    if tokenizer.pad_token_id is None or (
        tokenizer.pad_token_id == tokenizer.eos_token_id
    ):
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    # Resize embeddings if we added a new pad token above.
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA r=64 targeting all projection layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset: {args.data_path}")
    dataset = load_sft_dataset(args.data_path)
    print(f"Dataset size: {len(dataset)} examples")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        deepspeed=args.deepspeed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="oncallcompass-sft",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT training complete.")


if __name__ == "__main__":
    main()
