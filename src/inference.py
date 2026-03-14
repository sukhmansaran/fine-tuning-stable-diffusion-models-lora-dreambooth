#!/usr/bin/env python3
"""Inference script for generating images with trained LoRA weights.

Supports single-phase and dual-phase LoRA weight loading.

Usage:
    python src/inference.py \
        --model_path ./models/base \
        --lora_path ./outputs/lora/step_3330/lora_weights.safetensors \
        --prompt "sks, cyberpunk astronaut" \
        --num_images 4

    # Dual-phase inference:
    python src/inference.py \
        --model_path ./models/base \
        --lora_path ./outputs/lora/step_3330/lora_weights.safetensors \
        --dual_phase \
        --lora_rank 4 --lora_alpha 8 \
        --phase2_weight 0.3 \
        --prompt "sks, portrait photo" \
        --num_images 4
"""

import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

from src.pipeline import (
    apply_lora_weights,
    patch_text_encoder,
    patch_unet_attention,
    set_phase_weight,
    setup_trigger_token,
)
from src.utils import disable_safety_checker, seed_everything


DEFAULT_NEGATIVE = (
    "blurry, low resolution, grainy, overexposed, underexposed, bad lighting, "
    "jpeg artifacts, glitch, cropped, out of frame, watermark, duplicate, "
    "poorly drawn face, asymmetrical face, deformed features, bad skin texture, "
    "doll-like face, bad eyes, mutated hands, extra fingers, unrealistic proportions, "
    "cartoon, anime, illustration, painting, horror, morbid"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    p = argparse.ArgumentParser(description="Generate images with LoRA-finetuned SD")
    p.add_argument("--model_path", type=str, required=True, help="Base model path (diffusers format)")
    p.add_argument("--lora_path", type=str, required=True, help="Path to LoRA .safetensors weights")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="results/sample_outputs")
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=151101)
    p.add_argument("--trigger_word", type=str, default=None, help="Trigger word to register")

    # LoRA config
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)

    # Dual-phase options
    p.add_argument("--dual_phase", action="store_true", help="Use dual-phase LoRA loading")
    p.add_argument("--phase2_weight", type=float, default=0.3, help="Phase 2 blend weight (0-1)")

    return p.parse_args()


def load_pipeline(args: argparse.Namespace) -> StableDiffusionPipeline:
    """Load SD pipeline and apply LoRA weights.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Ready-to-use StableDiffusionPipeline.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    disable_safety_checker(pipe)

    # Register trigger word if provided
    if args.trigger_word:
        setup_trigger_token(pipe, args.trigger_word)

    if args.dual_phase:
        from src.pipeline import (
            patch_text_encoder_dual_phase,
            patch_unet_dual_phase,
        )
        r, a = args.lora_rank, args.lora_alpha
        patch_unet_dual_phase(pipe.unet, r, a, r, a)
        patch_text_encoder_dual_phase(pipe.text_encoder, r, a, r, a)
        apply_lora_weights(pipe.unet, args.lora_path)
        apply_lora_weights(pipe.text_encoder, args.lora_path)
        set_phase_weight(pipe.unet, args.phase2_weight)
        set_phase_weight(pipe.text_encoder, args.phase2_weight)
    else:
        patch_unet_attention(pipe.unet, args.lora_rank, args.lora_alpha)
        patch_text_encoder(pipe.text_encoder, args.lora_rank, args.lora_alpha)
        apply_lora_weights(pipe.unet, args.lora_path)
        apply_lora_weights(pipe.text_encoder, args.lora_path)

    return pipe


def generate(args: argparse.Namespace) -> None:
    """Generate and save images using the LoRA-finetuned model."""
    seed_everything(args.seed)
    pipe = load_pipeline(args)

    generator = torch.Generator("cuda").manual_seed(args.seed)

    print(f"Generating {args.num_images} image(s)...")
    result = pipe(
        prompt=[args.prompt] * args.num_images,
        negative_prompt=[args.negative_prompt] * args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for i, image in enumerate(result.images):
        path = os.path.join(args.output_dir, f"output_{i + 1}.png")
        image.save(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    args = parse_args()
    generate(args)
