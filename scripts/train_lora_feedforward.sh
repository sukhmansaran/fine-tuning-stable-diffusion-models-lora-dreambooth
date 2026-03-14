#!/bin/bash
# LoRA fine-tuning with attention + feedforward layers
# Usage: bash scripts/train_lora_feedforward.sh [config_path]

set -e

CONFIG="${1:-configs/training_config_feedforward.yaml}"

echo "Starting attention + feedforward LoRA training with config: $CONFIG"

accelerate launch src/train_lora_feedforward.py \
    --config "$CONFIG"
