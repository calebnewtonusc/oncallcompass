"""
OncallCompass Incident Training Pair Synthesizer.

Converts raw crawled postmortems and runbooks into structured training pairs:

1. SFT pairs: (alert_signals + context, ranked_hypotheses, investigation_steps, postmortem_draft)
2. DPO pairs: (fast_resolution_path, slow_resolution_path) preference pairs
3. Drill scenarios: held-out incident drills with known ground-truth root causes

The synthesizer uses an LLM (Claude or GPT-4) to extract structured data
from unstructured postmortem text.

Usage:
    python synthesis/incident_synthesizer.py \
        --raw_dir data/raw \
        --output_dir data \
        --anthropic_key $ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

SFT_EXTRACTION_PROMPT = """You are processing a postmortem or runbook document to create training data for an incident response AI.

Given the document, extract and return a JSON object with this exact schema:
{
  "alerts": ["list of alert strings that would have fired at incident start"],
  "logs": "representative log snippet from the incident (1-3 lines)",
  "metrics": {"metric_name": representative_value},
  "context": {
    "last_deploy": "time since last deploy if mentioned",
    "stack": ["service names involved"],
    "recent_changes": ["recent infra or config changes if mentioned"]
  },
  "ranked_hypotheses": [
    {
      "hypothesis": "specific root cause hypothesis",
      "confidence": 0.0-1.0,
      "evidence": ["evidence strings supporting this hypothesis"],
      "ruling_out": "single investigation step to confirm or rule out"
    }
  ],
  "investigation_steps": ["ordered list of investigation steps taken"],
  "postmortem_draft": {
    "summary": "one sentence summary",
    "timeline": ["timestamped event list"],
    "root_cause": "the actual root cause (ground truth)",
    "contributing_factors": ["list of contributing factors"],
    "action_items": [
      {"item": "specific action", "owner": "team or system layer", "prevents": "what recurrence this prevents"}
    ]
  }
}

If information is not present in the document, use reasonable inferences or null values.
Return ONLY valid JSON, no explanation.

Document:
"""

DPO_PAIR_PROMPT = """Given a postmortem document, generate a DPO training pair showing two investigation paths:
1. Fast path: expert SRE reasoning that reaches correct root cause efficiently
2. Slow path: junior reasoning that lists many possibilities without ranking

Return JSON:
{
  "alerts": ["alert strings"],
  "context": {"stack": ["services"], "last_deploy": "time"},
  "fast_path": {
    "ranked_hypotheses": [{"hypothesis": "...", "confidence": 0.8, "evidence": ["..."], "ruling_out": "..."}],
    "investigation_steps": ["step 1", "step 2"],
    "reasoning": "expert reasoning narrative"
  },
  "slow_path": {
    "ranked_hypotheses": [{"hypothesis": "...", "confidence": 0.3}],
    "investigation_steps": ["many generic steps"],
    "reasoning": "junior reasoning that doesn't prioritize"
  },
  "ground_truth": {
    "root_cause": "actual root cause",
    "expert_steps": ["minimal steps to reach root cause"],
    "baseline_steps": 8
  }
}

Document:
"""


class IncidentSynthesizer:
    """Synthesizes structured training pairs from raw incident documents."""

    def __init__(self, anthropic_key: str | None = None) -> None:
        self.client = anthropic.Anthropic(
            api_key=anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-6"

    def extract_sft_pair(self, document: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a structured SFT training pair from a raw document."""
        content = document.get("content", "")
        if len(content) < 300:
            return None

        # Truncate very long documents
        content = content[:8000]

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": SFT_EXTRACTION_PROMPT + content
                }]
            )
            response_text = message.content[0].text.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            pair = json.loads(json_match.group())
            pair["_source"] = document.get("source_url", "")
            pair["_doc_type"] = document.get("doc_type", "")
            return pair

        except (anthropic.APIError, json.JSONDecodeError, Exception):
            return None

    def extract_dpo_pair(self, document: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a DPO preference pair from a raw document."""
        content = document.get("content", "")
        if len(content) < 400:
            return None

        content = content[:6000]

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": DPO_PAIR_PROMPT + content
                }]
            )
            response_text = message.content[0].text.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            pair = json.loads(json_match.group())
            pair["_source"] = document.get("source_url", "")
            return pair

        except (anthropic.APIError, json.JSONDecodeError, Exception):
            return None


def load_raw_documents(raw_dir: Path) -> list[dict[str, Any]]:
    """Load all raw documents from the corpus directory."""
    docs = []
    for json_file in raw_dir.glob("*.json"):
        if json_file.name == "manifest.jsonl":
            continue
        try:
            with open(json_file) as f:
                docs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass training pair synthesizer")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--anthropic_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--max_docs", type=int, default=None, help="Cap number of docs (for testing)")
    parser.add_argument("--dpo_fraction", type=float, default=0.3, help="Fraction of docs to use for DPO pairs")
    parser.add_argument("--drill_fraction", type=float, default=0.1, help="Fraction of docs to hold out as drills")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "sft_pairs.jsonl"
    dpo_path = output_dir / "dpo_pairs.jsonl"
    drill_dir = output_dir / "drills"
    drill_dir.mkdir(exist_ok=True)
    drill_path = drill_dir / "compassbench_v1.jsonl"

    docs = load_raw_documents(raw_dir)
    if not docs:
        console.print(f"[red]No documents found in {raw_dir}. Run discovery/incident_corpus.py first.[/red]")
        return

    console.print(f"Loaded {len(docs)} raw documents")

    if args.max_docs:
        docs = docs[:args.max_docs]

    # Shuffle and split
    random.shuffle(docs)
    drill_count = max(1, int(len(docs) * args.drill_fraction))
    drill_docs = docs[:drill_count]
    train_docs = docs[drill_count:]

    dpo_count = int(len(train_docs) * args.dpo_fraction)
    dpo_docs = train_docs[:dpo_count]
    # Exclude DPO docs from SFT to prevent overlap between the two sets.
    sft_docs = train_docs[dpo_count:]

    synthesizer = IncidentSynthesizer(anthropic_key=args.anthropic_key)

    # Generate SFT pairs — write mode so reruns don't accumulate duplicates.
    console.print("[bold cyan]Generating SFT training pairs...[/bold cyan]")
    sft_count = 0
    with open(sft_path, "w") as sft_f:
        for doc in track(sft_docs, description="SFT pairs"):
            pair = synthesizer.extract_sft_pair(doc)
            if pair:
                sft_f.write(json.dumps(pair) + "\n")
                sft_count += 1

    console.print(f"Generated {sft_count} SFT pairs -> {sft_path}")

    # Generate DPO pairs — write mode so reruns don't accumulate duplicates.
    console.print("[bold cyan]Generating DPO preference pairs...[/bold cyan]")
    dpo_count_out = 0
    with open(dpo_path, "w") as dpo_f:
        for doc in track(dpo_docs, description="DPO pairs"):
            pair = synthesizer.extract_dpo_pair(doc)
            if pair:
                dpo_f.write(json.dumps(pair) + "\n")
                dpo_count_out += 1

    console.print(f"Generated {dpo_count_out} DPO pairs -> {dpo_path}")

    # Generate drill scenarios (held-out evaluation set) — write mode.
    console.print("[bold cyan]Generating CompassBench drill scenarios...[/bold cyan]")
    drill_count_out = 0
    with open(drill_path, "w") as drill_f:
        for doc in track(drill_docs, description="Drill scenarios"):
            drill = synthesizer.extract_sft_pair(doc)
            if drill and "postmortem_draft" in drill:
                # Drills include ground truth but hide it from the model input
                ground_truth = {
                    "root_cause": drill["postmortem_draft"].get("root_cause", ""),
                    "expert_steps": drill.get("investigation_steps", []),
                    "baseline_steps": 8,
                }
                drill_scenario = {
                    "alerts": drill.get("alerts", []),
                    "logs": drill.get("logs", ""),
                    "metrics": drill.get("metrics", {}),
                    "context": drill.get("context", {}),
                    "ground_truth": ground_truth,
                }
                drill_f.write(json.dumps(drill_scenario) + "\n")
                drill_count_out += 1

    console.print(f"Generated {drill_count_out} drill scenarios -> {drill_path}")
    console.print("[green]Synthesis complete.[/green]")


if __name__ == "__main__":
    main()
