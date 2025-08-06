# Fine-Tuning Stable Diffusion with DreamBooth and LoRA  
**Project: Fine-Tuning Realistic Vision for a Single Character**

## Overview

This repository documents my experimental journey in fine-tuning Stable Diffusion models by **combining the strengths of DreamBooth and LoRA** techniques. The project is built to help understand how SD models work and how they can be adapted efficiently for custom concepts (like a single character) while preserving base model capabilities.

The base model used for fine-tuning is **Realistic Vision**. My approach includes different phases of training, from fine-tuning attention projections to including feedforward layers, as well as generating images using LoRA weights alone or in dual-phase training. These experiments are shared through a set of clean and modular notebooks (still a work in progress — suggestions and contributions are welcome!).

## Contents

| File Name                            | Description                                                                                                      |
|-------------------------------------- |------------------------------------------------------------------------------------------------------------------|
| **1. fine tuning models with feedforward layers.ipynb** | Fine-tuning with self/cross-attention and feedforward layers (UNet & text encoder), for maximum adaptation.      |
| **2. fine tuning phase 1 code.ipynb**      | Phase 1 training: fine-tune self/cross-attention layers only (UNet & text encoder). Use as standalone or part 1. |
| **3. fine tuning phase 2 code.ipynb**      | Phase 2 training: to be used as part 2 for dual-phase approach.                                                  |
| **4. lora image generation.ipynb**   | Image generation using LoRA weights — for models trained via phase 1.                                            |
| **5. dual phase lora image generation.ipynb**    | Image generation for dual-phase-trained models (phase 1 + phase 2 LoRA weights).                                 |

## What are DreamBooth and LoRA?

- **DreamBooth**:  
  A fine-tuning method that enables Stable Diffusion to learn new concepts from a few example images, allowing you to personalize the model for specific subjects or styles.

- **LoRA (Low-Rank Adaptation)**:  
  A lightweight technique that adds trainable “adapters” to key layers of large models (like attention projections), enabling rapid, memory-efficient fine-tuning for new tasks, personalities, or styles—often with less risk of overfitting or damaging core knowledge.

## How These Notebooks Help

- **DreamBooth meets LoRA**:  
  By combining DreamBooth (for learning new concepts accurately) and LoRA (for efficient, modular adaptation), this repo demonstrates flexible strategies for tuning SD models. You can choose to fine-tune just the attention layers, expand to feedforward layers for more expressiveness, or experiment with phased, stepwise training.
- **Real-World Application**:  
  Perfect for creators and researchers who want to adapt Stable Diffusion (e.g., Realistic Vision) for a single character, artistic style, or proprietary dataset without retraining from scratch.
- **Modular Experiments**:  
  Use any phase/train pipeline separately, or in sequence (dual-phase), and generate images with the corresponding adapters.

## Notes & Contributions

- All notebooks are a work in progress: the code, method, and explanations can be improved!
- **Feedback, suggestions, or pull requests are highly welcome**—this project is for learning, sharing, and potentially pushing the boundaries of model fine-tuning.

## Why This Project?

I started this experiment to overcome technical frustrations and incompatibilities with popular libraries, and to get a first-hand understanding of what actually matters when fine-tuning large diffusion models for personalized use. By dissecting and recombining the best parts of DreamBooth and LoRA, I hope this repo helps others navigate the world of model adaptation with greater clarity and flexibility.

## Note
- The dataset used for training is private and will not be shared publicly.

- Sample images provided (if any) do not contain or reveal sensitive information.

- Users wishing to replicate your results must use their own similar data.

**Thank you for visiting — explore, fork, contribute, and let’s learn how to shape AI models more creatively and efficiently!**
