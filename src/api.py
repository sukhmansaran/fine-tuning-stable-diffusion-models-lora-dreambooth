"""FastAPI server — path-based dataset, fine-tune LoRA, generate images."""

import base64
import glob
import io
import os
import threading
from typing import Any, Optional

import secrets

import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.pipeline import (
    apply_lora_weights,
    patch_text_encoder,
    patch_unet_attention,
    set_phase_weight,
    setup_trigger_token,
)
from src.utils import disable_safety_checker, seed_everything

# ── Defaults from env ─────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/base")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("API_KEY", "")  # empty string = auth disabled
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str = Security(_api_key_header)) -> None:
    """Dependency that enforces API key auth when API_KEY env var is set."""
    if not _API_KEY:
        return  # auth disabled — local dev mode
    if not key or not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SD LoRA Studio", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_pipe: Optional[StableDiffusionPipeline] = None
_train_state: dict[str, Any] = {
    "status": "idle", "step": 0, "total": 0,
    "loss": None, "error": None, "lora_path": None,
    "trigger_word": None,
}
_train_lock = threading.Lock()

DEFAULT_NEGATIVE = (
    "blurry, low resolution, grainy, overexposed, underexposed, bad lighting, "
    "jpeg artifacts, glitch, cropped, out of frame, watermark, duplicate, "
    "poorly drawn face, asymmetrical face, deformed features"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ── Pydantic models ───────────────────────────────────────────────────────────
class DatasetScanRequest(BaseModel):
    dataset_dir: str


class TrainRequest(BaseModel):
    dataset_dir: str                    # local path the user provides
    output_dir: str = OUTPUT_DIR        # where to save checkpoints
    trigger_word: str = "sks"
    trigger_init_word: str = "person"
    max_train_steps: int = 500
    learning_rate: float = 4e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    lora_r: int = 4
    lora_alpha: int = 8
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation: int = 4
    mixed_precision: str = "fp16"
    train_text_encoder: bool = True
    train_feedforward: bool = False
    save_every_n_steps: int = 250
    seed: int = 42


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    num_images: int = 1
    height: int = 512
    width: int = 512
    steps: int = 30
    guidance_scale: float = 6.5
    seed: int = 42
    phase2_weight: float = 0.3


class GenerateResponse(BaseModel):
    images: list[str]


# ── System info ───────────────────────────────────────────────────────────────
@app.get("/system")
def system_info():
    has_cuda = torch.cuda.is_available()
    info: dict[str, Any] = {"has_cuda": has_cuda, "device": "cuda" if has_cuda else "cpu"}
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]  = props.name
        info["vram_gb"]   = round(props.total_memory / 1024 ** 3, 1)
        info["bf16"]      = torch.cuda.is_bf16_supported()
        info["recommended_precision"] = "bf16" if info["bf16"] and info["vram_gb"] >= 24 else "fp16"
        info["recommended_steps"]     = 500 if info["vram_gb"] >= 16 else 300
        info["recommended_batch"]     = 1
    else:
        info["recommended_precision"] = "no"
        info["recommended_steps"]     = 100
        info["recommended_batch"]     = 1
    return info


# ── Health / config ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.get("/config")
def config():
    return {
        "trigger_word": os.environ.get("TRIGGER_WORD", "") or None,
        "dual_phase":   os.environ.get("DUAL_PHASE", "false").lower() == "true",
    }


# ── Dataset scan ──────────────────────────────────────────────────────────────
@app.post("/dataset/scan")
def scan_dataset(req: DatasetScanRequest, _: None = Depends(_require_api_key)):
    """
    Scan a local directory for image + .txt caption pairs.
    Returns a preview list so the user can confirm before training.
    """
    d = req.dataset_dir.strip()
    if not d:
        raise HTTPException(400, "dataset_dir is required.")
    if not os.path.isdir(d):
        raise HTTPException(404, f"Directory not found: {d}")

    pairs, missing_captions = [], []
    for fname in sorted(os.listdir(d)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        img_path = os.path.join(d, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            pairs.append({"image": fname, "caption": caption})
        else:
            missing_captions.append(fname)

    return {
        "directory": d,
        "found": len(pairs),
        "missing_captions": missing_captions,
        "pairs": pairs,          # full list for preview
    }


# ── Training ──────────────────────────────────────────────────────────────────
def _run_training(cfg: dict[str, Any]) -> None:
    global _train_state
    try:
        from accelerate import Accelerator
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, get_scheduler

        from src.dataset import ImageCaptionDataset
        from src.pipeline import (
            patch_text_encoder,
            patch_text_encoder_with_feedforward,
            patch_unet_attention,
            patch_unet_feedforward,
            setup_trigger_token,
        )
        from src.utils import enable_tf32, extract_lora_state_dict, seed_everything
        enable_tf32()
        seed_everything(cfg.get("seed", 42))

        mixed = cfg.get("mixed_precision", "fp16")
        if not torch.cuda.is_available():
            mixed = "no"

        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.get("gradient_accumulation", 1),
            mixed_precision=mixed,
        )
        device = accelerator.device
        dtype = (torch.float16 if mixed == "fp16"
                 else torch.bfloat16 if mixed == "bf16"
                 else torch.float32)

        _train_state.update(status="loading_model", step=0)

        pipe = StableDiffusionPipeline.from_pretrained(cfg["model_path"], torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], subfolder="tokenizer")
        pipe.tokenizer = tokenizer
        setup_trigger_token(pipe, cfg["trigger_word"], cfg.get("trigger_init_word", "person"))

        unet = UNet2DConditionModel.from_pretrained(
            cfg["model_path"], subfolder="unet", torch_dtype=dtype
        ).to(device)
        for p in unet.parameters():            p.requires_grad = False
        for p in pipe.text_encoder.parameters(): p.requires_grad = False

        lora_params = patch_unet_attention(unet, cfg["lora_r"], cfg["lora_alpha"])
        if cfg.get("train_feedforward", False):
            lora_params += patch_unet_feedforward(unet, cfg["lora_r"], cfg["lora_alpha"])
        if cfg.get("train_text_encoder", True):
            te_patch = patch_text_encoder_with_feedforward if cfg.get("train_feedforward", False) else patch_text_encoder
            lora_params += te_patch(pipe.text_encoder, cfg["lora_r"], cfg["lora_alpha"])

        pipe.vae = AutoencoderKL.from_pretrained(
            cfg.get("vae_model", "stabilityai/sd-vae-ft-mse"), torch_dtype=torch.float32
        ).to(device)
        pipe.text_encoder.to(device, dtype=dtype)

        optimizer = torch.optim.AdamW(lora_params, lr=cfg["learning_rate"])
        lr_sched  = get_scheduler(
            cfg.get("lr_scheduler", "cosine"), optimizer=optimizer,
            num_warmup_steps=cfg.get("lr_warmup_steps", 0),
            num_training_steps=cfg["max_train_steps"],
        )

        dataset    = ImageCaptionDataset(cfg["dataset_dir"], tokenizer, cfg.get("resolution", 512))
        dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 1), shuffle=True)
        unet.train()
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

        max_steps  = cfg["max_train_steps"]
        save_every = cfg.get("save_every_n_steps", 250)
        _train_state.update(status="training", total=max_steps)

        global_step = 0
        for _ in range(99999):
            for batch in dataloader:
                with accelerator.accumulate(unet):
                    pv  = batch["pixel_values"].to(device, dtype=torch.float32)
                    ids = batch["input_ids"].to(device)

                    with torch.no_grad():
                        latents = pipe.vae.encode(pv).latent_dist.sample()
                        latents = latents.clamp(-10, 10) * 0.18215
                        latents = latents.to(dtype=dtype)

                    noise     = 0.9 * torch.randn_like(latents)
                    max_t     = 300 if global_step < 100 else pipe.scheduler.config.num_train_timesteps
                    timesteps = torch.randint(0, max_t, (latents.shape[0],), device=device).long()
                    noisy     = pipe.scheduler.add_noise(latents, noise, timesteps)

                    with torch.no_grad():
                        enc = pipe.text_encoder(ids)[0].to(dtype=dtype)

                    pred = unet(noisy, timesteps, encoder_hidden_states=enc).sample
                    if torch.isnan(pred).any():
                        continue

                    loss = torch.nn.functional.l1_loss(pred.float(), noise.float())
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lora_params, 1.0)
                        optimizer.step()
                        lr_sched.step()
                        optimizer.zero_grad()
                        global_step += 1
                        _train_state.update(step=global_step, loss=round(loss.item(), 4))

                        if global_step % save_every == 0 or global_step >= max_steps:
                            save_dir = os.path.join(cfg["output_dir"], f"step_{global_step}")
                            os.makedirs(save_dir, exist_ok=True)
                            combined = {
                                **extract_lora_state_dict(accelerator.unwrap_model(unet)),
                                **extract_lora_state_dict(accelerator.unwrap_model(pipe.text_encoder)),
                            }
                            lora_out = os.path.join(save_dir, "lora_weights.safetensors")
                            save_file(combined, lora_out)
                            _train_state["lora_path"] = lora_out

                        if global_step >= max_steps:
                            break
            if global_step >= max_steps:
                break

        _train_state.update(status="done")

    except Exception as exc:
        import traceback
        _train_state.update(status="error", error=traceback.format_exc())


@app.post("/train")
def start_training(req: TrainRequest, _: None = Depends(_require_api_key)):
    global _train_state
    with _train_lock:
        if _train_state["status"] in ("loading_model", "training"):
            raise HTTPException(409, "Training already in progress.")

        d = req.dataset_dir.strip()
        if not os.path.isdir(d):
            raise HTTPException(404, f"Dataset directory not found: {d}")

        # Verify at least one image+caption pair exists
        pairs = [
            f for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            and os.path.exists(os.path.join(d, os.path.splitext(f)[0] + ".txt"))
        ]
        if not pairs:
            raise HTTPException(400, "No image+caption pairs found in that directory.")

        _train_state = {
            "status": "queued", "step": 0, "total": req.max_train_steps,
            "loss": None, "error": None, "lora_path": None,
            "trigger_word": req.trigger_word,
        }

        cfg = req.model_dump()
        cfg["model_path"] = MODEL_PATH
        cfg["vae_model"]  = "stabilityai/sd-vae-ft-mse"

        threading.Thread(target=_run_training, args=(cfg,), daemon=True).start()

    return {"started": True}


@app.get("/train/status")
def train_status():
    return dict(_train_state)


@app.post("/train/load")
def load_trained_model(_: None = Depends(_require_api_key)):
    """Hot-reload the inference pipeline with the latest trained LoRA weights."""
    global _pipe
    lora_path = _train_state.get("lora_path")
    if not lora_path or not os.path.exists(lora_path):
        raise HTTPException(404, "No trained weights found. Run training first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    disable_safety_checker(pipe)

    trigger = _train_state.get("trigger_word") or os.environ.get("TRIGGER_WORD", "")
    lora_r  = int(os.environ.get("LORA_RANK", "4"))
    lora_a  = int(os.environ.get("LORA_ALPHA", "8"))

    if trigger:
        setup_trigger_token(pipe, trigger)

    patch_unet_attention(pipe.unet, rank=lora_r, alpha=lora_a)
    patch_text_encoder(pipe.text_encoder, rank=lora_r, alpha=lora_a)
    apply_lora_weights(pipe, lora_path)

    _pipe = pipe
    return {"loaded": True, "lora_path": lora_path}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global _pipe
    lora_path  = os.environ.get("LORA_PATH", "")
    dual_phase = os.environ.get("DUAL_PHASE", "false").lower() == "true"
    lora_rank  = int(os.environ.get("LORA_RANK", "4"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "8"))
    trigger    = os.environ.get("TRIGGER_WORD", None)

    if not os.path.exists(MODEL_PATH):
        return  # no base model yet — user will configure and train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    disable_safety_checker(pipe)

    if trigger:
        setup_trigger_token(pipe, trigger)

    if lora_path and os.path.exists(lora_path):
        if dual_phase:
            patch_unet_attention(pipe.unet, rank=lora_rank, alpha=lora_alpha, dual_phase=True)
            patch_text_encoder(pipe.text_encoder, rank=lora_rank, alpha=lora_alpha, dual_phase=True)
        else:
            patch_unet_attention(pipe.unet, rank=lora_rank, alpha=lora_alpha)
            patch_text_encoder(pipe.text_encoder, rank=lora_rank, alpha=lora_alpha)
        apply_lora_weights(pipe, lora_path)

    _pipe = pipe


# ── Generate ──────────────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, _: None = Depends(_require_api_key)):
    if _pipe is None:
        raise HTTPException(503, "Model not loaded. Complete training and click 'Load Model'.")

    seed_everything(req.seed)

    if os.environ.get("DUAL_PHASE", "false").lower() == "true":
        set_phase_weight(_pipe, req.phase2_weight)

    output = _pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_images_per_prompt=req.num_images,
        height=req.height,
        width=req.width,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
    )

    encoded = []
    for img in output.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())

    return GenerateResponse(images=encoded)
