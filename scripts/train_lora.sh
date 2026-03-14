#!/bin/bash
# LoRA fine-tuning (single-phase)
# Usage: bash scripts/train_lora.sh [config_path]

set -e

CONFIG="${1:-configs/training_config.yaml}"

echo "Starting LoRA training with config: $CONFIG"

accelerate launch src/train_lora.py \
    --config "$CONFIG"
