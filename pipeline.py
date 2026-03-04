"""
OncallCompass end-to-end training pipeline.

Runs all three stages in sequence:
    Stage 1: SFT on incident-to-resolution traces
    Stage 2: RL with MTTR drill reward
    Stage 3: DPO on (fast-resolution, slow-resolution) pairs

Usage:
    python pipeline.py --stage all
    python pipeline.py --stage sft
    python pipeline.py --stage rl
    python pipeline.py --stage dpo
    python pipeline.py --stage eval
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

STAGES = ["sft", "rl", "dpo", "eval"]


STAGE_TIMEOUT_SECONDS = 72 * 3600  # 72-hour hard cap per stage to prevent hangs


def run_stage(name: str, cmd: list[str], cwd: Path | None = None) -> int:
    console.print(Panel(f"[bold]Stage: {name}[/bold]", style="cyan"))
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=cwd, timeout=STAGE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        console.print(f"[red]Stage {name} timed out after {elapsed:.1f}s[/red]")
        return 1
    elapsed = time.time() - start
    if result.returncode != 0:
        console.print(
            f"[red]Stage {name} failed (exit {result.returncode}) after {elapsed:.1f}s[/red]"
        )
    else:
        console.print(f"[green]Stage {name} completed in {elapsed:.1f}s[/green]")
    return result.returncode


def stage_sft(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "training/train.py",
        "--model_name",
        args.base_model,
        "--data_path",
        args.sft_data,
        "--output_dir",
        args.sft_checkpoint,
        "--num_train_epochs",
        str(args.sft_epochs),
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "16",
        "--learning_rate",
        "2e-5",
        "--warmup_ratio",
        "0.05",
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "10",
        "--deepspeed",
        args.ds_config,
    ]
    return run_stage("SFT", cmd)


def stage_rl(args: argparse.Namespace) -> int:
    # Note: --ppo_epochs was removed when the RL trainer migrated from PPO to GRPO.
    # Use --num_generations to control the GRPO group size instead.
    cmd = [
        sys.executable,
        "training/train_rl.py",
        "--model_path",
        args.sft_checkpoint,
        "--output_dir",
        args.rl_checkpoint,
        "--drill_set",
        args.drill_set,
        "--num_train_epochs",
        str(args.rl_epochs),
        "--num_generations",
        "8",
        "--learning_rate",
        "1e-5",
        "--logging_steps",
        "10",
    ]
    return run_stage("RL (MTTR reward)", cmd)


def stage_dpo(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "training/train_dpo.py",
        "--model_path",
        args.rl_checkpoint,
        "--data_path",
        args.dpo_data,
        "--output_dir",
        args.final_checkpoint,
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "5e-6",
        "--beta",
        "0.1",
        "--logging_steps",
        "10",
    ]
    return run_stage("DPO alignment", cmd)


def stage_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "evaluation/compassbench.py",
        "--model_path",
        args.final_checkpoint,
        "--drill_set",
        args.drill_set,
        "--output_path",
        "results/compassbench_results.json",
    ]
    return run_stage("CompassBench evaluation", cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass training pipeline")
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_data", default="data/sft_pairs.jsonl")
    parser.add_argument("--dpo_data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--drill_set", default="data/drills/compassbench_v1.jsonl")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft")
    parser.add_argument("--rl_checkpoint", default="checkpoints/rl")
    parser.add_argument("--final_checkpoint", default="checkpoints/final")
    parser.add_argument("--ds_config", default="training/configs/ds_config.json")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--rl_epochs", type=int, default=2)
    args = parser.parse_args()

    # Create output directories
    for d in [
        args.sft_checkpoint,
        args.rl_checkpoint,
        args.final_checkpoint,
        "results",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    stages_to_run = STAGES if args.stage == "all" else [args.stage]
    dispatch = {"sft": stage_sft, "rl": stage_rl, "dpo": stage_dpo, "eval": stage_eval}

    console.print(
        Panel(
            "[bold cyan]OncallCompass Training Pipeline[/bold cyan]\n"
            f"Stages: {', '.join(stages_to_run)}",
            title="oncallcompass",
        )
    )

    for stage_name in stages_to_run:
        rc = dispatch[stage_name](args)
        if rc != 0:
            console.print(f"[red]Pipeline aborted at stage: {stage_name}[/red]")
            sys.exit(rc)

    console.print("[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    main()
