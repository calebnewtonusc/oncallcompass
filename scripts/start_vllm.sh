#!/usr/bin/env bash
# start_vllm.sh — Launch 4 vLLM instances across 18x A6000 GPUs (16 used for synthesis)
# Each instance uses 4 GPUs with tensor parallel for Qwen2.5-72B-Instruct.
# Ports: 8001, 8002, 8003, 8004

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
API_KEY="${VLLM_API_KEY:-synthesis}"
LOG_DIR="${VLLM_LOG_DIR:-logs/vllm}"

mkdir -p "${LOG_DIR}"

echo "Starting 4x vLLM instances of ${MODEL}"
echo "Ports: 8001-8004 | Tensor parallel: 4 | GPUs per instance: 4"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "${MODEL}" \
	--tensor-parallel-size 4 \
	--port 8001 \
	--api-key "${API_KEY}" \
	--gpu-memory-utilization 0.90 \
	--max-model-len 32768 \
	--dtype bfloat16 \
	>"${LOG_DIR}/vllm_8001.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "${MODEL}" \
	--tensor-parallel-size 4 \
	--port 8002 \
	--api-key "${API_KEY}" \
	--gpu-memory-utilization 0.90 \
	--max-model-len 32768 \
	--dtype bfloat16 \
	>"${LOG_DIR}/vllm_8002.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=8,9,10,11 vllm serve "${MODEL}" \
	--tensor-parallel-size 4 \
	--port 8003 \
	--api-key "${API_KEY}" \
	--gpu-memory-utilization 0.90 \
	--max-model-len 32768 \
	--dtype bfloat16 \
	>"${LOG_DIR}/vllm_8003.log" 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=12,13,14,15 vllm serve "${MODEL}" \
	--tensor-parallel-size 4 \
	--port 8004 \
	--api-key "${API_KEY}" \
	--gpu-memory-utilization 0.90 \
	--max-model-len 32768 \
	--dtype bfloat16 \
	>"${LOG_DIR}/vllm_8004.log" 2>&1 &
PID4=$!

echo "PIDs: ${PID1} ${PID2} ${PID3} ${PID4}"
echo "Waiting 60s for models to load..."
sleep 60

# Health check all 4 instances
ALL_READY=true
for PORT in 8001 8002 8003 8004; do
	if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
		echo "[OK] vLLM on port ${PORT} is ready"
	else
		echo "[FAIL] vLLM on port ${PORT} did not start (check ${LOG_DIR}/vllm_${PORT}.log)"
		ALL_READY=false
	fi
done

if $ALL_READY; then
	echo ""
	echo "All 4 vLLM instances ready on ports 8001-8004"
	echo "Export: export VLLM_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004"
else
	echo ""
	echo "One or more instances failed. Check logs in ${LOG_DIR}/"
	exit 1
fi
