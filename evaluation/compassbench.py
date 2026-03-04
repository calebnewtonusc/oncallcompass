"""
CompassBench — OncallCompass Evaluation Suite.

Evaluates model performance on simulated incident drills with known
ground-truth root causes.

Metrics:
    - Root Cause Accuracy @ 1 (RCA@1): correct root cause ranked first
    - Root Cause Accuracy @ 3 (RCA@3): correct root cause in top 3
    - Mean Reciprocal Rank (MRR): quality of ranking
    - MTTR Reduction: steps saved vs. baseline (unguided) investigation
    - Step Precision: fraction of model steps that match expert steps
    - Postmortem Quality: structured postmortem completeness score

Usage:
    python evaluation/compassbench.py \
        --model_path checkpoints/rl \
        --drill_set data/drills/compassbench_v1.jsonl \
        --output_path results/compassbench_results.json
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

BASELINE_STEPS = 8  # Median for unguided investigation


@dataclass
class DrillResult:
    """Result for a single drill scenario."""

    drill_id: str
    correct_root_cause: str
    model_root_cause_rank: int | None  # None if not surfaced
    model_hypotheses_count: int
    model_steps_count: int
    expert_steps_count: int
    mrr: float
    rca_at_1: bool
    rca_at_3: bool
    mttr_reduction: float
    step_precision: float
    postmortem_quality: float
    latency_ms: float


@dataclass
class CompassBenchResults:
    """Aggregate results across all drills."""

    model_path: str
    drill_count: int
    rca_at_1: float
    rca_at_3: float
    mrr: float
    mttr_reduction: float
    step_precision: float
    postmortem_quality: float
    mean_latency_ms: float
    drill_results: list[DrillResult]


def compute_mrr(rank: int | None) -> float:
    """Mean Reciprocal Rank for a single result."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def compute_step_precision(model_steps: list[str], expert_steps: list[str]) -> float:
    """Fraction of model steps that overlap with expert steps."""
    if not model_steps or not expert_steps:
        return 0.0

    expert_lower = {s.lower() for s in expert_steps}
    matches = sum(
        1
        for step in model_steps
        if any(keyword in step.lower() for keyword in _extract_keywords(expert_lower))
    )
    return matches / len(model_steps)


def _extract_keywords(steps: set[str]) -> set[str]:
    """Extract meaningful keywords from step strings for fuzzy matching."""
    keywords = set()
    for step in steps:
        words = step.lower().split()
        keywords.update(w for w in words if len(w) > 4)
    return keywords


def compute_postmortem_quality(postmortem: dict[str, Any]) -> float:
    """Score the quality of a generated postmortem (0-1)."""
    if not postmortem:
        return 0.0

    score = 0.0
    max_score = 5.0

    # Has summary
    if postmortem.get("summary") and len(postmortem["summary"]) > 20:
        score += 1.0

    # Has timeline
    timeline = postmortem.get("timeline", [])
    if isinstance(timeline, list) and len(timeline) >= 2:
        score += 1.0

    # Has root cause
    if postmortem.get("root_cause") and len(postmortem["root_cause"]) > 10:
        score += 1.0

    # Has contributing factors
    factors = postmortem.get("contributing_factors", [])
    if isinstance(factors, list) and len(factors) >= 1:
        score += 1.0

    # Has actionable action items with "prevents" field
    action_items = postmortem.get("action_items", [])
    if isinstance(action_items, list):
        prevention_items = [
            item
            for item in action_items
            if isinstance(item, dict)
            and item.get("prevents")
            and len(item.get("prevents", "")) > 5
        ]
        if prevention_items:
            score += 1.0

    return score / max_score


def find_root_cause_rank(
    hypotheses: list[dict[str, Any]], correct_root_cause: str
) -> int | None:
    """Find the rank of the correct root cause in the model's hypothesis list."""
    correct_lower = correct_root_cause.lower()
    for i, hyp in enumerate(hypotheses):
        hypothesis_text = hyp.get("hypothesis", "").lower()
        # Fuzzy match: check if key terms from correct cause appear in hypothesis
        correct_terms = [t for t in correct_lower.split() if len(t) > 4]
        # Fallback: if the root cause only contains short words (e.g. "DNS down"),
        # none pass the len>4 filter — use all words instead so RCA metrics
        # are not always zero for short-word root causes.
        if not correct_terms:
            correct_terms = correct_lower.split()
        if any(term in hypothesis_text for term in correct_terms):
            return i + 1
    return None


def generate_model_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    drill: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Generate model response for a drill and return (response_dict, latency_ms)."""
    prompt = build_drill_prompt(drill)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # Greedy for deterministic evaluation
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start) * 1000

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    try:
        response = json.loads(generated.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re

        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            try:
                response = json.loads(match.group())
            except json.JSONDecodeError:
                response = {}
        else:
            response = {}

    return response, latency_ms


def build_drill_prompt(drill: dict[str, Any]) -> str:
    """Format a drill scenario as a model prompt."""
    # Hide ground truth from the model
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
        "You are OncallCompass. Given incident signals, respond with ranked hypotheses "
        "and investigation steps in JSON format matching the OncallCompass output schema.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def evaluate_drill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    drill: dict[str, Any],
    drill_id: str,
) -> DrillResult:
    """Evaluate model on a single drill scenario."""
    ground_truth = drill.get("ground_truth", {})
    correct_root_cause = ground_truth.get("root_cause", "")
    expert_steps = ground_truth.get("expert_steps", [])

    response, latency_ms = generate_model_response(model, tokenizer, drill)

    hypotheses = response.get("ranked_hypotheses", [])
    model_steps = response.get("investigation_steps", [])
    postmortem = response.get("postmortem_draft", {})

    correct_rank = find_root_cause_rank(hypotheses, correct_root_cause)
    mrr = compute_mrr(correct_rank)
    rca_at_1 = correct_rank == 1
    rca_at_3 = correct_rank is not None and correct_rank <= 3
    # Use per-drill baseline_steps from ground truth when available so that
    # drills with a known expert baseline are evaluated on their own baseline
    # rather than the global constant.
    drill_baseline = ground_truth.get("baseline_steps", BASELINE_STEPS)
    # Only award positive MTTR reduction when the correct root cause was ranked first.
    # Otherwise a brief but wrong response would score positively on this metric.
    mttr_reduction = (
        (drill_baseline - len(model_steps)) / drill_baseline
        if (model_steps and rca_at_1 and drill_baseline > 0)
        else 0.0
    )
    step_precision = compute_step_precision(model_steps, expert_steps)
    postmortem_quality = compute_postmortem_quality(postmortem)

    return DrillResult(
        drill_id=drill_id,
        correct_root_cause=correct_root_cause,
        model_root_cause_rank=correct_rank,
        model_hypotheses_count=len(hypotheses),
        model_steps_count=len(model_steps),
        expert_steps_count=len(expert_steps),
        mrr=mrr,
        rca_at_1=rca_at_1,
        rca_at_3=rca_at_3,
        mttr_reduction=mttr_reduction,
        step_precision=step_precision,
        postmortem_quality=postmortem_quality,
        latency_ms=latency_ms,
    )


def print_results_table(results: CompassBenchResults) -> None:
    """Print a Rich table summarizing CompassBench results."""
    table = Table(title=f"CompassBench Results — {results.model_path}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Description")

    table.add_row("RCA@1", f"{results.rca_at_1:.1%}", "Correct root cause ranked first")
    table.add_row("RCA@3", f"{results.rca_at_3:.1%}", "Correct root cause in top 3")
    table.add_row("MRR", f"{results.mrr:.3f}", "Mean Reciprocal Rank")
    table.add_row(
        "MTTR Reduction", f"{results.mttr_reduction:.1%}", "Steps saved vs. baseline"
    )
    table.add_row(
        "Step Precision",
        f"{results.step_precision:.1%}",
        "Fraction of steps matching expert path",
    )
    table.add_row(
        "Postmortem Quality",
        f"{results.postmortem_quality:.1%}",
        "Postmortem completeness score",
    )
    table.add_row(
        "Mean Latency", f"{results.mean_latency_ms:.0f}ms", "Model response time"
    )
    table.add_row("Drills Evaluated", str(results.drill_count), "Total drill scenarios")

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="CompassBench evaluation")
    parser.add_argument("--model_path", default="checkpoints/rl")
    parser.add_argument("--drill_set", default="data/drills/compassbench_v1.jsonl")
    parser.add_argument("--output_path", default="results/compassbench_results.json")
    parser.add_argument("--max_drills", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.drill_set).exists():
        console.print(f"[red]Drill set not found: {args.drill_set}[/red]")
        console.print("Run synthesis/incident_synthesizer.py first.")
        return

    # Load drill scenarios
    drills = []
    with open(args.drill_set) as f:
        for line in f:
            line = line.strip()
            if line:
                drills.append(json.loads(line))

    if args.max_drills:
        drills = drills[: args.max_drills]

    console.print(f"Loaded {len(drills)} drill scenarios")

    # Load model
    console.print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # nosec B615
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Evaluate
    drill_results = []
    for i, drill in enumerate(drills):
        drill_id = f"drill_{i:04d}"
        console.print(f"Evaluating {drill_id}...")
        result = evaluate_drill(model, tokenizer, drill, drill_id)
        drill_results.append(result)

    # Aggregate
    n = len(drill_results)
    if n == 0:
        console.print("[red]No drills evaluated.[/red]")
        return
    aggregate = CompassBenchResults(
        model_path=args.model_path,
        drill_count=n,
        rca_at_1=sum(r.rca_at_1 for r in drill_results) / n,
        rca_at_3=sum(r.rca_at_3 for r in drill_results) / n,
        mrr=sum(r.mrr for r in drill_results) / n,
        mttr_reduction=sum(r.mttr_reduction for r in drill_results) / n,
        step_precision=sum(r.step_precision for r in drill_results) / n,
        postmortem_quality=sum(r.postmortem_quality for r in drill_results) / n,
        mean_latency_ms=sum(r.latency_ms for r in drill_results) / n,
        drill_results=drill_results,
    )

    print_results_table(aggregate)

    # Save results
    results_dict = asdict(aggregate)
    with open(args.output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    console.print(f"[green]Results saved to {args.output_path}[/green]")


if __name__ == "__main__":
    main()
