#!/bin/bash
# Image generation with trained LoRA weights
# Usage: bash scripts/generate_images.sh

set -e

python src/inference.py \
    --model_path ./models/base \
    --lora_path ./outputs/lora/step_3330/lora_weights.safetensors \
    --prompt "sks, portrait photo, natural lighting" \
    --negative_prompt "blurry, low resolution, cartoon, anime" \
    --num_images 4 \
    --height 768 \
    --width 768 \
    --steps 30 \
    --guidance_scale 6.5 \
    --seed 151101 \
    --trigger_word sks \
    --lora_rank 4 \
    --lora_alpha 8 \
    --output_dir results/sample_outputs
