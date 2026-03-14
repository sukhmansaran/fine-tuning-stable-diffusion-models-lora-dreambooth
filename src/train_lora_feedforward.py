#!/usr/bin/env python3
"""LoRA fine-tuning script — attention AND feedforward layers.

Patches self/cross-attention layers AND feedforward (MLP) layers on both
UNet and text encoder with LoRA adapters. More expressive than
attention-only training, at the cost of higher VRAM usage and more
trainable parameters.

UNet targets:  to_q, to_k, to_v, to_out, GEGLU.proj, MLP output
Text encoder:  q_proj, k_proj, v_proj, out_proj, fc1, fc2

Corresponds to: notebooks/fine_tuning_with_feedforward_layers.ipynb

Usage:
    python src/train_lora_feedforward.py --config configs/training_config_feedforward.yaml
"""

import argparse
import os
from typing import Any

import torch
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from safetensors.torch import save_file
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from src.dataset import ImageCaptionDataset
from src.pipeline import (
    patch_text_encoder_with_feedforward,
    patch_unet_attention,
    patch_unet_feedforward,
    setup_trigger_token,
)
from src.utils import (
    check_nan_params,
    enable_tf32,
    extract_lora_state_dict,
    load_config,
    save_config_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning (attention + feedforward layers)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config_feedforward.yaml",
        help="Path to YAML training config",
    )
    return parser.parse_args()


def setup_pipeline(cfg: dict[str, Any], accelerator: Accelerator) -> tuple:
    """Load base model, tokenizer, UNet, and apply attention + feedforward LoRA patches.

    Returns:
        Tuple of (pipe, unet, lora_params, tokenizer).
    """
    device = accelerator.device

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_path"], torch_dtype=torch.float16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], subfolder="tokenizer")
    pipe.tokenizer = tokenizer
    setup_trigger_token(pipe, cfg["trigger_word"], cfg.get("trigger_init_word", "person"))

    unet = UNet2DConditionModel.from_pretrained(
        cfg["model_path"], subfolder="unet", torch_dtype=torch.float16,
    ).to(device)

    # Freeze all base parameters
    for p in unet.parameters():
        p.requires_grad = False
    for p in pipe.text_encoder.parameters():
        p.requires_grad = False

    # Patch attention layers
    lora_params = patch_unet_attention(unet, cfg["lora_r"], cfg["lora_alpha"])

    # Patch feedforward layers
    lora_params += patch_unet_feedforward(unet, cfg["lora_r"], cfg["lora_alpha"])

    # Patch text encoder (attention + feedforward)
    if cfg.get("train_text_encoder", False):
        lora_params += patch_text_encoder_with_feedforward(
            pipe.text_encoder, cfg["lora_r"], cfg["lora_alpha"],
        )

    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters collected.")

    total = sum(p.numel() for p in lora_params)
    print(f"Total trainable LoRA parameters: {total:,}")

    vae_id = cfg.get("vae_model", "stabilityai/sd-vae-ft-mse")
    pipe.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float32)
    pipe.vae.to(device, dtype=torch.float32)
    pipe.text_encoder.to(device, dtype=torch.float16)

    return pipe, unet, lora_params, tokenizer


def train(cfg: dict[str, Any]) -> None:
    """Run the attention + feedforward LoRA training loop."""
    enable_tf32()
    seed_everything(cfg.get("seed", 42))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("gradient_accumulation", 1),
        mixed_precision=cfg.get("mixed_precision", "fp16"),
    )

    pipe, unet, lora_params, tokenizer = setup_pipeline(cfg, accelerator)
    save_config_json(cfg, cfg["output_dir"])

    optimizer = torch.optim.AdamW(lora_params, lr=cfg["learning_rate"])
    lr_scheduler = get_scheduler(
        cfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 0),
        num_training_steps=cfg["max_train_steps"],
    )

    dataset = ImageCaptionDataset(cfg["dataset_dir"], tokenizer, cfg.get("resolution", 512))
    dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 1), shuffle=True)

    unet.train()
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    check_nan_params(unet, "UNet (pre-training)")

    global_step = 0
    max_steps = cfg["max_train_steps"]
    save_every = cfg.get("save_every_n_steps", 500)
    log_every = cfg.get("log_every_n_steps", 100)

    print(f"Starting attention + feedforward LoRA training for {max_steps} steps...")

    for epoch in range(9999):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                input_ids = batch["input_ids"].to(accelerator.device)

                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.clamp(-10, 10) * 0.18215
                    latents = latents.to(dtype=torch.float16)

                noise = 0.9 * torch.randn_like(latents)
                max_t = 300 if global_step < 100 else pipe.scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_t, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden = pipe.text_encoder(input_ids)[0]
                    encoder_hidden = encoder_hidden.to(dtype=torch.float16)

                with autocast("cuda", dtype=torch.float32):
                    pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden).sample

                if torch.isnan(pred).any():
                    print(f"NaN in prediction at step {global_step}, skipping")
                    continue

                noise = noise.to(pred.dtype)
                loss = torch.nn.functional.l1_loss(pred, noise)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss at step {global_step}, skipping")
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_every == 0:
                        print(f"Step {global_step}/{max_steps} | Loss: {loss.item():.4f}")

                    if global_step % save_every == 0:
                        _save_checkpoint(cfg, accelerator, unet, pipe, global_step)

                    if global_step >= max_steps:
                        break

        if global_step >= max_steps:
            break

    _save_checkpoint(cfg, accelerator, unet, pipe, global_step)
    print(f"Training complete at step {global_step}.")


def _save_checkpoint(
    cfg: dict[str, Any],
    accelerator: Accelerator,
    unet: torch.nn.Module,
    pipe: Any,
    step: int,
) -> None:
    """Save LoRA weights checkpoint."""
    save_dir = os.path.join(cfg["output_dir"], f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)

    unet_lora = extract_lora_state_dict(accelerator.unwrap_model(unet))
    te_lora = extract_lora_state_dict(accelerator.unwrap_model(pipe.text_encoder))
    combined = {**unet_lora, **te_lora}

    save_file(combined, os.path.join(save_dir, "lora_weights.safetensors"))
    print(f"Checkpoint saved: {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
