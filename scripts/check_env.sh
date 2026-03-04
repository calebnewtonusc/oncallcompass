#!/usr/bin/env bash
# check_env.sh — Verify OncallCompass environment before training

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ERRORS=$((ERRORS+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

ERRORS=0

echo "OncallCompass Environment Check"
echo "================================"
echo ""

# ── Python packages ──────────────────────────────────────────
echo "Checking Python packages..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pass "PyTorch + CUDA"
else
    fail "PyTorch CUDA not available"
fi
if python -c "import transformers" 2>/dev/null; then
    pass "transformers"
else
    fail "transformers not installed"
fi
if python -c "import trl" 2>/dev/null; then
    pass "trl"
else
    fail "trl not installed"
fi
if python -c "import peft" 2>/dev/null; then
    pass "peft"
else
    fail "peft not installed"
fi
if python -c "import deepspeed" 2>/dev/null; then
    pass "deepspeed"
else
    fail "deepspeed not installed"
fi
if python -c "import anthropic" 2>/dev/null; then
    pass "anthropic SDK"
else
    warn "anthropic not installed (synthesis fallback disabled)"
fi
if python -c "import rich" 2>/dev/null; then
    pass "rich"
else
    fail "rich not installed"
fi
if python -c "import fastapi" 2>/dev/null; then
    pass "fastapi"
else
    warn "fastapi not installed (HTTP server disabled)"
fi
echo ""

# ── GPU availability ─────────────────────────────────────────
echo "Checking GPU availability..."
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -ge 8 ]; then
    pass "GPU count: $GPU_COUNT (sufficient for training)"
elif [ "$GPU_COUNT" -ge 4 ]; then
    warn "GPU count: $GPU_COUNT (use --num_gpus $GPU_COUNT)"
elif [ "$GPU_COUNT" -ge 1 ]; then
    warn "GPU count: $GPU_COUNT (single-GPU mode only)"
else
    fail "No GPUs detected"
fi
echo ""

# ── Environment variables ─────────────────────────────────────
echo "Checking environment variables..."
[ -n "${ANTHROPIC_API_KEY:-}" ] && pass "ANTHROPIC_API_KEY set" || warn "ANTHROPIC_API_KEY not set (synthesis disabled)"
[ -n "${WANDB_API_KEY:-}" ] && pass "WANDB_API_KEY set" || warn "WANDB_API_KEY not set (W&B disabled)"
[ -n "${HF_TOKEN:-}" ] && pass "HF_TOKEN set" || warn "HF_TOKEN not set"
[ -n "${GITHUB_TOKEN:-}" ] && pass "GITHUB_TOKEN set" || warn "GITHUB_TOKEN not set (GitHub crawl disabled)"
echo ""

# ── Data directories ─────────────────────────────────────────
echo "Checking data directories..."
[ -d "data" ] && pass "data/ directory exists" || warn "data/ not found (run discovery scripts first)"
[ -d "checkpoints" ] && pass "checkpoints/ directory exists" || warn "checkpoints/ not found (will be created on first run)"
echo ""

# ── Storage ──────────────────────────────────────────────────
echo "Checking storage..."
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -ge 100 ]; then
    pass "Free disk space: ${FREE_GB}GB"
elif [ "$FREE_GB" -ge 50 ]; then
    warn "Free disk space: ${FREE_GB}GB (minimum)"
else
    fail "Free disk space: ${FREE_GB}GB (need ≥100GB)"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed. OncallCompass ready.${NC}"
else
    echo -e "${RED}$ERRORS check(s) failed. Fix above issues before training.${NC}"
    exit 1
fi
