"""Shared utilities for training and inference."""

import os
import json
import random
from typing import Any

import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all backends."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_tf32() -> None:
    """Enable TF32 on Ampere+ GPUs for faster matmuls."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def save_config_json(config: dict[str, Any], output_dir: str) -> None:
    """Save training config as JSON alongside model weights."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {path}")


def disable_safety_checker(pipe: Any) -> None:
    """Disable the NSFW safety checker for local inference."""
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = lambda images, **kwargs: (
            images,
            [False] * len(images),
        )


def check_nan_params(model: torch.nn.Module, label: str = "model") -> bool:
    """Check model parameters for NaN/Inf values. Returns True if clean."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[NaN/Inf detected] {label}: {name}")
            return False
    return True


def extract_lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA-related keys from a model state dict."""
    return {
        k: v for k, v in model.state_dict().items() if "lora" in k.lower()
    }
