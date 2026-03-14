"""Image-caption dataset for DreamBooth + LoRA training."""

import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer


class ImageCaptionDataset(Dataset):
    """Dataset that pairs images with text captions for SD training.

    Expects a directory containing image files (.png/.jpg) with
    matching .txt caption files sharing the same base name.

    Args:
        image_dir: Path to directory containing images and captions.
        tokenizer: HuggingFace tokenizer for encoding captions.
        resolution: Target image resolution (square crop).
    """

    def __init__(
        self,
        image_dir: str,
        tokenizer: PreTrainedTokenizer,
        resolution: int = 512,
    ) -> None:
        self.image_paths: list[str] = []
        self.caption_paths: list[str] = []
        self.tokenizer = tokenizer

        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_dir, fname)
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    self.image_paths.append(img_path)
                    self.caption_paths.append(txt_path)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image/caption pairs found in {image_dir}. "
                "Ensure each image has a matching .txt file."
            )

        self.image_transforms = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if image.getbbox() is None:
            raise ValueError(f"Empty image: {self.image_paths[idx]}")

        pixel_values = self.image_transforms(image)

        with open(self.caption_paths[idx], "r") as f:
            caption = f.read().strip()

        inputs = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
        }
