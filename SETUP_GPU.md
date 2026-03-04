# OncallCompass — GPU Setup Guide

## Hardware Configuration

OncallCompass uses 18× NVIDIA A6000 48GB GPUs for training.

```
GPU Allocation:
  GPUs 0-3:    vLLM synthesis inference (4-GPU tensor parallel)
  GPUs 4-7:    vLLM synthesis inference (4-GPU tensor parallel)
  GPUs 8-15:   SFT/DPO training (8-GPU ZeRO-3)
  GPUs 16-17:  Evaluation / CompassBench
```

## Training Commands

### Stage 1 — SFT (8 GPUs, ZeRO-3)

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus 8 training/train.py \
    --model_name Qwen/Qwen2.5-7B-Coder-Instruct \
    --data_path data/sft_pairs.jsonl \
    --output_dir checkpoints/oncallcompass-sft-v1 \
    --run_name oncallcompass-sft-v1 \
    --deepspeed training/configs/ds_config.json
```

Estimated time: ~18 hours for 50k examples, 3 epochs.

### Stage 2 — GRPO RL (8 GPUs, ZeRO-2)

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus 8 training/train_rl.py \
    --sft_checkpoint checkpoints/oncallcompass-sft-v1/final \
    --data_path data/drills/compassbench_v1.jsonl \
    --output_dir checkpoints/oncallcompass-rl-v1 \
    --run_name oncallcompass-rl-v1
```

Estimated time: ~10 hours, 1 epoch.

### Stage 3 — DPO (8 GPUs, ZeRO-3)

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus 8 training/train_dpo.py \
    --rl_checkpoint checkpoints/oncallcompass-rl-v1/final \
    --data_path data/dpo_pairs.jsonl \
    --output_dir checkpoints/oncallcompass-dpo-v1 \
    --run_name oncallcompass-dpo-v1
```

Estimated time: ~6 hours, 1 epoch.

## vLLM Synthesis Inference

Start two vLLM instances for parallel synthesis:

```bash
# Synthesis server 1 (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000

# Synthesis server 2 (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8001
```

## CompassBench Evaluation

```bash
CUDA_VISIBLE_DEVICES=16,17 python evaluation/compassbench.py \
    --model checkpoints/oncallcompass-dpo-v1/final \
    --output results/compassbench_v1.json
```

## Memory Requirements

| Stage | GPUs | VRAM per GPU | CPU RAM |
|---|---|---|---|
| SFT (ZeRO-3 + CPU offload) | 8× A6000 | ~42GB | ~200GB |
| RL (ZeRO-2) | 8× A6000 | ~44GB | ~100GB |
| DPO (ZeRO-3) | 8× A6000 | ~42GB | ~200GB |
| vLLM (72B, TP=4) | 4× A6000 | ~47GB | ~32GB |
| Eval (7B) | 2× A6000 | ~20GB | ~32GB |

## Minimum Single-GPU Setup

For inference only (evaluation and agent use):

```bash
# Single A6000 or RTX 3090/4090
CUDA_VISIBLE_DEVICES=0 python evaluation/compassbench.py \
    --model checkpoints/oncallcompass-dpo-v1/final
```

The 7B model fits in 16GB VRAM at 4-bit quantization:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/oncallcompass-dpo-v1/final",
    quantization_config=bnb_config,
    device_map="auto",
)
```
