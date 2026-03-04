# OncallCompass — Architecture

## Overview

OncallCompass is a 7B-parameter incident response specialist built on Qwen2.5-7B-Instruct. The full training pipeline is three stages: SFT on 300k incident traces, RL with MTTR reward, DPO on resolution-path preference pairs.

## Data Pipeline

```
Public Sources                  Synthesis
─────────────                   ─────────
GitHub repos/postmortems   ──►  incident_synthesizer.py
Engineering blogs               ├─ Incident-to-resolution triples
AWS/GCP/Azure status pages      ├─ DPO preference pairs
Runbook repositories            └─ CompassBench drill scenarios
Open incident datasets
        │
        ▼
discovery/incident_corpus.py
        │
        ▼
data/
├── raw/              # Raw crawled documents
├── sft_pairs.jsonl   # (context, ranked_hypotheses, resolution) triples
├── dpo_pairs.jsonl   # (fast_path, slow_path) preference pairs
└── drills/
    └── compassbench_v1.jsonl  # Held-out evaluation drills
```

## Training Pipeline

```
Qwen2.5-7B-Instruct (base)
        │
        ▼
Stage 1: SFT (training/train.py)
  Input:  (alert_signals + stack_context, ranked_hypotheses, resolution_trace)
  Loss:   Cross-entropy on hypothesis ranking + step sequencing outputs
  Data:   300k triples
  Infra:  DeepSpeed ZeRO-3, gradient checkpointing
        │
        ▼
Stage 2: RL with MTTR Reward (training/train_rl.py)
  Reward: time_to_correct_root_cause in drill simulations
          + MTTR reduction vs. baseline (unguided investigation)
          - penalty for correct root cause ranked below first position
          - penalty for investigation steps out of expert order
  Algo:   PPO with verifiable reward
  Data:   Generated drill scenarios from synthesizer
        │
        ▼
Stage 3: DPO Alignment (training/train_dpo.py)
  Preferred: fast-resolution path (correct root cause, expert step sequence)
  Rejected:  slow-resolution path (correct eventually, but inefficient path)
  Data:   dpo_pairs.jsonl
        │
        ▼
checkpoints/final/  ◄── deployed model
```

## Model Inputs and Outputs

### Input Schema

```json
{
  "alerts": ["5xx spike on /api/checkout", "latency p99 +340ms"],
  "logs": "Optional log snippets",
  "metrics": {"error_rate": 0.12, "p99_ms": 890},
  "context": {
    "last_deploy": "6h ago",
    "stack": ["nginx", "node", "postgres", "redis"],
    "recent_changes": ["DB connection pool size +50%"],
    "similar_incidents": ["INC-2024-0891"]
  }
}
```

### Output Schema

```json
{
  "ranked_hypotheses": [
    {
      "hypothesis": "Connection pool exhaustion on Postgres",
      "confidence": 0.74,
      "evidence": ["p99 latency spike without CPU spike", "recent pool size change"],
      "ruling_out": "Check pg_stat_activity for waiting connections"
    },
    {
      "hypothesis": "Upstream dependency timeout cascade",
      "confidence": 0.18,
      "evidence": ["5xx pattern consistent with timeout threshold"],
      "ruling_out": "Check service mesh timeout config and downstream error rates"
    }
  ],
  "investigation_steps": [
    "1. Check pg_stat_activity — count waiting vs. active connections",
    "2. Compare connection pool metrics before/after pool size change",
    "3. Check application connection acquisition timeout logs",
    "4. If pool healthy: pivot to upstream service mesh latency"
  ],
  "postmortem_draft": {
    "summary": "...",
    "timeline": [...],
    "root_cause": "...",
    "contributing_factors": [...],
    "action_items": [...]
  }
}
```

## Reward Function Design

The MTTR reward is computed from simulated incident drills with known ground truth:

```
R(response) =
    + w1 * (1 if correct_root_cause_rank == 1 else 0)
    + w2 * (1 / correct_root_cause_rank)  # partial credit for near-first
    + w3 * (baseline_drill_time - model_drill_time) / baseline_drill_time
    - w4 * steps_before_correct_pivot  # penalize wasted investigation steps
```

where `baseline_drill_time` is the median time taken by junior SRE annotators on the same drill.

## Infrastructure

- **Training**: DeepSpeed ZeRO-3 across 4xA100-80GB
- **Base model**: Qwen/Qwen2.5-7B-Instruct
- **Training framework**: transformers + trl (PPO, DPO)
- **Experiment tracking**: Weights & Biases
- **Evaluation**: CompassBench (held-out drill set, automated MTTR measurement)
- **Serving**: vLLM with structured output (JSON schema enforcement)

## Key Design Decisions

1. **Ranking over listing**: The primary failure mode of general models on incident response is listing all possible causes without ranking. The reward function specifically punishes non-first placement of correct root cause.

2. **Drill-based reward**: Unlike code generation where correctness is binary, incident response quality is a continuous function of investigation speed. Simulated drills with known ground truth make this reward verifiable and reproducible.

3. **DPO over RLHF for alignment**: Human annotators for incident quality are expensive and inconsistent. DPO on automatically-extracted (fast, slow) path pairs from the corpus is cheaper and more consistent at this scale.

4. **Postmortem prevention focus**: SFT training filters for postmortems where action items have documented recurrence prevention outcomes. Generic postmortems without measurable follow-up items are excluded from training.
