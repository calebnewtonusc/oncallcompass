#!/usr/bin/env bash
# run_all.sh — Full OncallCompass pipeline: collect → synth → train → eval

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="oncallcompass-${TIMESTAMP}"
DATA_DIR="data/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

echo "OncallCompass Full Pipeline: ${RUN_NAME}"
echo "Data:        ${DATA_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo ""

mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}"

# ── Step 1: Environment check ────────────────────────────────
echo "[1/6] Checking environment..."
bash scripts/check_env.sh

# ── Step 2: Collect postmortems ──────────────────────────────
echo "[2/6] Collecting postmortem corpus..."
python -c "
import asyncio
from discovery.postmortem_crawler import PostmortemCrawler
from discovery.aws_incidents import CloudStatusCrawler
from discovery.pagerduty_corpus import PagerDutyCorpus

async def collect():
    crawler = PostmortemCrawler(output_dir='${DATA_DIR}/raw/postmortems')
    await crawler.crawl_all()
    cloud = CloudStatusCrawler(output_dir='${DATA_DIR}/raw/cloud_incidents')
    await cloud.crawl_all()

asyncio.run(collect())
"

echo "  Collection complete"

# ── Step 3: Synthesize training pairs ────────────────────────
echo "[3/6] Synthesizing training pairs..."
python synthesis/synthesize_bulk.py \
    --raw_dir "${DATA_DIR}/raw" \
    --output_dir "${DATA_DIR}" \
    --n_per_template 3

echo "  SFT pairs: $(wc -l < "${DATA_DIR}/sft_pairs.jsonl")"
echo "  DPO pairs: $(wc -l < "${DATA_DIR}/dpo_pairs.jsonl" 2>/dev/null || echo 0)"

# ── Step 4: Stage 1 — SFT ───────────────────────────────────
echo "[4/6] Stage 1: SFT training..."
deepspeed --num_gpus 8 training/train.py \
    --model_name Qwen/Qwen2.5-7B-Coder-Instruct \
    --data_path "${DATA_DIR}/sft_pairs.jsonl" \
    --output_dir "${CHECKPOINT_DIR}/sft" \
    --deepspeed training/configs/ds_config.json \
    --num_train_epochs 3

echo "  SFT checkpoint: ${CHECKPOINT_DIR}/sft"

# ── Step 5: Stage 2 — RL ────────────────────────────────────
echo "[5/6] Stage 2: RL training..."
deepspeed --num_gpus 8 training/train_rl.py \
    --model_path "${CHECKPOINT_DIR}/sft" \
    --output_dir "${CHECKPOINT_DIR}/rl" \
    --drill_set "${DATA_DIR}/drills/compassbench_v1.jsonl" \
    --num_train_epochs 2

echo "  RL checkpoint: ${CHECKPOINT_DIR}/rl"

# ── Step 6: CompassBench evaluation ──────────────────────────
echo "[6/6] Running CompassBench evaluation..."
${EVAL_GPUS:+CUDA_VISIBLE_DEVICES=$EVAL_GPUS} python evaluation/compassbench.py \
    --model_path "${CHECKPOINT_DIR}/rl" \
    --drill_set "${DATA_DIR}/drills/compassbench_v1.jsonl" \
    --output_path "results/${RUN_NAME}_compassbench.json"

echo ""
echo "OncallCompass pipeline complete."
echo "Final model: ${CHECKPOINT_DIR}/rl"
echo "Results:     results/${RUN_NAME}_compassbench.json"
