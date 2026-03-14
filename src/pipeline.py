"""LoRA layer implementations and model patching utilities.

Contains custom LoRA wrappers for nn.Linear layers and functions
to patch UNet and text encoder attention/feedforward layers.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


# ──────────────────────────────────────────────────────────
#  Single-Phase LoRA
# ──────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds trainable LoRA matrices on top of a frozen linear layer.
    Output = original(x) + lora_up(lora_down(x)) * scaling

    Args:
        linear: The original nn.Linear layer to wrap.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA output.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"LoRALinear wraps nn.Linear, got {type(linear)}"
            )

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_down = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling


# ──────────────────────────────────────────────────────────
#  Dual-Phase LoRA (for DreamBooth phase 2)
# ──────────────────────────────────────────────────────────

class LoRALinearDualPhase(nn.Module):
    """Dual-phase LoRA wrapper supporting two independent adapter sets.

    Phase 1 adapters are frozen; phase 2 adapters are trainable.
    During inference, outputs are blended via ``phase2_weight``.

    Args:
        linear: The original nn.Linear layer.
        rank1: Phase 1 LoRA rank.
        alpha1: Phase 1 scaling factor.
        rank2: Phase 2 LoRA rank.
        alpha2: Phase 2 scaling factor.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank1: int,
        alpha1: float,
        rank2: int,
        alpha2: float,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.scale1 = alpha1 / rank1
        self.scale2 = alpha2 / rank2
        self.phase2_weight: float = 1.0

        # Phase 1 (frozen after loading)
        self.lora_down = nn.Linear(linear.in_features, rank1, bias=False)
        self.lora_up = nn.Linear(rank1, linear.out_features, bias=False)

        # Phase 2 (trainable)
        self.lora2_down = nn.Linear(linear.in_features, rank2, bias=False)
        self.lora2_up = nn.Linear(rank2, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora2_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora2_up.weight)

        # Freeze phase 1
        for p in self.lora_down.parameters():
            p.requires_grad = False
        for p in self.lora_up.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora1 = self.lora_up(self.lora_down(x)) * self.scale1
        lora2 = self.lora2_up(self.lora2_down(x)) * self.scale2
        blended = (1 - self.phase2_weight) * lora1 + self.phase2_weight * lora2
        return base + blended


# ──────────────────────────────────────────────────────────
#  UNet Patching
# ──────────────────────────────────────────────────────────

def patch_unet_attention(
    unet: nn.Module, rank: int, alpha: float
) -> list[nn.Parameter]:
    """Patch UNet cross/self-attention layers with LoRA adapters.

    Targets: to_q, to_k, to_v, to_out projections.

    Returns:
        List of trainable LoRA parameters.
    """
    lora_params: list[nn.Parameter] = []
    attn_attrs = ["to_q", "to_k", "to_v", "to_out"]

    for module in unet.modules():
        for attr in attn_attrs:
            if not hasattr(module, attr):
                continue
            original = getattr(module, attr)

            if isinstance(original, nn.Linear) and not isinstance(original, LoRALinear):
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                setattr(module, attr, lora_layer)
                lora_params.extend(lora_layer.lora_down.parameters())
                lora_params.extend(lora_layer.lora_up.parameters())

            elif isinstance(original, nn.ModuleList):
                for i, sublayer in enumerate(original):
                    if isinstance(sublayer, nn.Linear) and not isinstance(sublayer, LoRALinear):
                        lora_layer = LoRALinear(sublayer, rank=rank, alpha=alpha)
                        original[i] = lora_layer
                        lora_params.extend(lora_layer.lora_down.parameters())
                        lora_params.extend(lora_layer.lora_up.parameters())

    print(f"UNet attention LoRA params: {sum(p.numel() for p in lora_params):,}")
    return lora_params


def patch_unet_feedforward(
    unet: nn.Module, rank: int, alpha: float
) -> list[nn.Parameter]:
    """Patch UNet feedforward (MLP) layers with LoRA adapters.

    Targets GEGLU proj layers and MLP output projections.

    Returns:
        List of trainable LoRA parameters.
    """
    lora_params: list[nn.Parameter] = []

    for _name, module in unet.named_modules():
        if hasattr(module, "proj") and isinstance(module.proj, nn.Linear):
            if not isinstance(module.proj, LoRALinear):
                lora_layer = LoRALinear(module.proj, rank=rank, alpha=alpha)
                module.proj = lora_layer
                lora_params.extend(lora_layer.lora_down.parameters())
                lora_params.extend(lora_layer.lora_up.parameters())

        if isinstance(module, nn.ModuleList):
            for idx, submodule in enumerate(module):
                if idx == 2 and isinstance(submodule, nn.Linear):
                    if not isinstance(submodule, LoRALinear):
                        lora_layer = LoRALinear(submodule, rank=rank, alpha=alpha)
                        module[idx] = lora_layer
                        lora_params.extend(lora_layer.lora_down.parameters())
                        lora_params.extend(lora_layer.lora_up.parameters())

    print(f"UNet feedforward LoRA params: {sum(p.numel() for p in lora_params):,}")
    return lora_params


# ──────────────────────────────────────────────────────────
#  Text Encoder Patching
# ──────────────────────────────────────────────────────────

def patch_text_encoder(
    text_encoder: nn.Module,
    rank: int,
    alpha: float,
) -> list[nn.Parameter]:
    """Patch CLIP text encoder attention layers with LoRA adapters.

    Targets: q_proj, k_proj, v_proj, out_proj.

    Args:
        text_encoder: The CLIP text encoder module.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.

    Returns:
        List of trainable LoRA parameters.
    """
    targets = {"q_proj", "k_proj", "v_proj", "out_proj"}
    lora_params: list[nn.Parameter] = []

    for module in text_encoder.modules():
        for name in targets:
            if not hasattr(module, name):
                continue
            proj = getattr(module, name)
            if isinstance(proj, nn.Linear) and not isinstance(proj, LoRALinear):
                lora_layer = LoRALinear(proj, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
                lora_params.extend(lora_layer.lora_down.parameters())
                lora_params.extend(lora_layer.lora_up.parameters())

    print(f"Text encoder LoRA params: {sum(p.numel() for p in lora_params):,}")
    return lora_params


def patch_text_encoder_with_feedforward(
    text_encoder: nn.Module,
    rank: int,
    alpha: float,
) -> list[nn.Parameter]:
    """Patch CLIP text encoder attention AND feedforward layers with LoRA.

    Targets: q_proj, k_proj, v_proj, out_proj, fc1, fc2.

    Args:
        text_encoder: The CLIP text encoder module.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.

    Returns:
        List of trainable LoRA parameters.
    """
    targets = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    lora_params: list[nn.Parameter] = []

    for module in text_encoder.modules():
        for name in targets:
            if not hasattr(module, name):
                continue
            proj = getattr(module, name)
            if isinstance(proj, nn.Linear) and not isinstance(proj, LoRALinear):
                lora_layer = LoRALinear(proj, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
                lora_params.extend(lora_layer.lora_down.parameters())
                lora_params.extend(lora_layer.lora_up.parameters())

    print(f"Text encoder LoRA params (with feedforward): {sum(p.numel() for p in lora_params):,}")
    return lora_params


# ──────────────────────────────────────────────────────────
#  Dual-Phase Patching (DreamBooth Phase 2)
# ──────────────────────────────────────────────────────────

def patch_unet_dual_phase(
    unet: nn.Module,
    rank1: int, alpha1: float,
    rank2: int, alpha2: float,
) -> list[nn.Parameter]:
    """Patch UNet with dual-phase LoRA for DreamBooth phase 2.

    Phase 1 adapters are frozen; only phase 2 params are returned.
    """
    trainable: list[nn.Parameter] = []

    for module in unet.modules():
        for attr in ["to_q", "to_k", "to_v", "to_out"]:
            if not hasattr(module, attr):
                continue
            orig = getattr(module, attr)

            if isinstance(orig, nn.Linear):
                dual = LoRALinearDualPhase(orig, rank1, alpha1, rank2, alpha2)
                setattr(module, attr, dual)
                trainable.extend(dual.lora2_down.parameters())
                trainable.extend(dual.lora2_up.parameters())
            elif isinstance(orig, nn.ModuleList):
                for i, sub in enumerate(orig):
                    if isinstance(sub, nn.Linear):
                        dual = LoRALinearDualPhase(sub, rank1, alpha1, rank2, alpha2)
                        orig[i] = dual
                        trainable.extend(dual.lora2_down.parameters())
                        trainable.extend(dual.lora2_up.parameters())

    print(f"UNet dual-phase trainable params: {sum(p.numel() for p in trainable):,}")
    return trainable


def patch_text_encoder_dual_phase(
    text_encoder: nn.Module,
    rank1: int, alpha1: float,
    rank2: int, alpha2: float,
) -> list[nn.Parameter]:
    """Patch text encoder with dual-phase LoRA for DreamBooth phase 2."""
    trainable: list[nn.Parameter] = []
    targets = {"q_proj", "k_proj", "v_proj", "out_proj"}

    for module in text_encoder.modules():
        for name in targets:
            if not hasattr(module, name):
                continue
            orig = getattr(module, name)
            if isinstance(orig, nn.Linear):
                dual = LoRALinearDualPhase(orig, rank1, alpha1, rank2, alpha2)
                setattr(module, name, dual)
                trainable.extend(dual.lora2_down.parameters())
                trainable.extend(dual.lora2_up.parameters())

    print(f"Text encoder dual-phase trainable params: {sum(p.numel() for p in trainable):,}")
    return trainable


def load_phase1_weights(
    unet: nn.Module,
    text_encoder: nn.Module,
    weights_path: str,
) -> None:
    """Load phase 1 LoRA weights into dual-phase patched models."""
    state_dict = load_file(weights_path)
    loaded = 0

    for name, module in list(unet.named_modules()) + list(text_encoder.named_modules()):
        if not isinstance(module, LoRALinearDualPhase):
            continue
        for part_name, part in [("lora_down", module.lora_down), ("lora_up", module.lora_up)]:
            key = f"{name}.{part_name}.weight"
            if key in state_dict:
                with torch.no_grad():
                    part.weight.copy_(state_dict[key])
                    loaded += 1

    print(f"Loaded {loaded} phase 1 LoRA weight tensors")


def set_phase_weight(model: nn.Module, phase2_weight: float) -> None:
    """Set blending factor between phase 1 and phase 2 LoRA outputs.

    Args:
        model: Model patched with LoRALinearDualPhase layers.
        phase2_weight: 0.0 = phase 1 only, 1.0 = phase 2 only.
    """
    for module in model.modules():
        if isinstance(module, LoRALinearDualPhase):
            module.phase2_weight = phase2_weight


# ──────────────────────────────────────────────────────────
#  Trigger Token Setup
# ──────────────────────────────────────────────────────────

def setup_trigger_token(
    pipe: object,
    trigger_word: str,
    init_word: str = "person",
) -> None:
    """Add trigger token to tokenizer and initialize its embedding.

    If the trigger word is already a single token, no action is taken.

    Args:
        pipe: StableDiffusionPipeline instance.
        trigger_word: The identity token (e.g. "sks").
        init_word: Word to copy embedding from for initialization.
    """
    tokenizer = pipe.tokenizer
    if len(tokenizer.tokenize(trigger_word)) <= 1:
        print(f"'{trigger_word}' is already a single token")
        return

    tokenizer.add_tokens([trigger_word])
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = pipe.text_encoder.get_input_embeddings()
        new_id = tokenizer.convert_tokens_to_ids(trigger_word)
        base_id = tokenizer.convert_tokens_to_ids(init_word)
        emb.weight[new_id] = emb.weight[base_id].clone()

    print(f"Added trigger token '{trigger_word}' (id {new_id})")


# ──────────────────────────────────────────────────────────
#  Weight Loading for Inference
# ──────────────────────────────────────────────────────────

def apply_lora_weights(model: nn.Module, weights_path: str) -> None:
    """Load LoRA weights from safetensors file into a patched model."""
    state_dict = load_file(weights_path, device="cuda")
    missing = []
    for name, param in model.named_parameters():
        if "lora" in name:
            if name in state_dict:
                param.data.copy_(state_dict[name])
            else:
                missing.append(name)
    print(f"LoRA weights loaded from {weights_path}")
    if missing:
        print(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
