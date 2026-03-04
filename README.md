# OncallCompass

Incident response sequencing: ranked hypotheses, investigation steps, MTTR reduction.

OncallCompass is a specialized ML model trained on 300k incident-to-resolution traces. It ranks root cause hypotheses by likelihood given your stack, sequences investigation steps in the order expert SREs actually follow them, and generates drift-resistant postmortems. The reward signal is time-to-correct-root-cause reduction in simulated incident drills.

## Architecture

```
oncallcompass/
├── web/                    # Next.js coming-soon frontend
├── training/               # SFT, RL, and DPO training scripts
├── discovery/              # Incident corpus crawling
├── synthesis/              # Training pair synthesis
├── evaluation/             # CompassBench evaluation suite
├── agents/                 # Triage agent inference
├── pipeline.py             # End-to-end training pipeline
└── requirements.txt        # Python dependencies
```

## Training Pipeline

### Stage 1 — Supervised Fine-Tuning

300k (alert signals + context, ranked hypotheses, resolution trace) triples from public runbooks, postmortems, and incident tickets. Base model: Qwen2.5-7B-Coder-Instruct.

```bash
python training/train.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --data_path data/sft_pairs.jsonl \
  --output_dir checkpoints/sft \
  --deepspeed training/configs/ds_config.json
```

### Stage 2 — RL with MTTR Reward

Reward signal: time-to-correct-root-cause in simulated incident drills. The model is penalized for correct-but-slow diagnoses and for surfacing the right root cause anywhere other than first.

```bash
python training/train_rl.py \
  --model_path checkpoints/sft \
  --output_dir checkpoints/rl \
  --reward_fn mttr_drill_reward
```

### Stage 3 — DPO Alignment

(fast-resolution path, slow-resolution path) pairs from the corpus. OncallCompass learns to prefer hypothesis ranking over listing, topology-aware reasoning over metric correlation.

## Evaluation

CompassBench measures performance on simulated incident drills with known ground-truth root causes.

```bash
python evaluation/compassbench.py \
  --model_path checkpoints/rl \
  --drill_set data/drills/compassbench_v1.jsonl
```

## Inference

```python
from agents.triage_agent import TriageAgent

agent = TriageAgent(model_path="checkpoints/rl")
result = agent.triage(
    alerts=["5xx spike on /api/checkout", "latency p99 +340ms"],
    context={"last_deploy": "6h ago", "stack": ["nginx", "node", "postgres", "redis"]}
)
print(result.ranked_hypotheses)
print(result.investigation_steps)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in HuggingFace token and wandb key
```

## Website

```bash
cd web && npm install && npm run dev
```
