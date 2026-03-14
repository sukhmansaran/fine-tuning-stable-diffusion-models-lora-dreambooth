#!/bin/bash
# DreamBooth dual-phase LoRA training
# Usage: bash scripts/train_dreambooth.sh [config_path]
#
# Ensure phase2.enabled is true and phase1 paths are set in config.

set -e

CONFIG="${1:-configs/training_config.yaml}"

echo "Starting DreamBooth phase 2 training with config: $CONFIG"

accelerate launch src/train_dreambooth.py \
    --config "$CONFIG"
