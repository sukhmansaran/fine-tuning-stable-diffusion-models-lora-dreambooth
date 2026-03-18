"""Smoke tests for the FastAPI endpoints (no GPU required)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Client with auth disabled (no API_KEY set)."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_pretrained:

        mock_pipe = MagicMock()
        mock_pretrained.return_value = mock_pipe

        import src.api as api_module
        api_module._pipe = mock_pipe
        api_module._API_KEY = ""  # auth disabled

        yield TestClient(api_module.app)


@pytest.fixture
def authed_client():
    """Client with API_KEY enforcement enabled."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("diffusers.StableDiffusionPipeline.from_pretrained") as mock_pretrained:

        mock_pipe = MagicMock()
        mock_pretrained.return_value = mock_pipe

        import src.api as api_module
        api_module._pipe = mock_pipe
        api_module._API_KEY = "test-secret-key"

        yield TestClient(api_module.app)


# ── Health / config / system ──────────────────────────────────────────────────

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_system(client):
    res = client.get("/system")
    assert res.status_code == 200
    data = res.json()
    assert "has_cuda" in data
    assert "device" in data
    # CPU path: no GPU fields, safe defaults
    assert data["has_cuda"] is False
    assert data["recommended_precision"] == "no"


def test_config(client):
    res = client.get("/config")
    assert res.status_code == 200
    data = res.json()
    assert "trigger_word" in data
    assert "dual_phase" in data


# ── Dataset scan ──────────────────────────────────────────────────────────────

def test_dataset_scan_missing_dir(client):
    res = client.post("/dataset/scan", json={"dataset_dir": "/nonexistent/path/xyz"})
    assert res.status_code == 404


def test_dataset_scan_empty_dir_param(client):
    res = client.post("/dataset/scan", json={"dataset_dir": "   "})
    assert res.status_code == 400


def test_dataset_scan_valid_dir(client):
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "001.jpg")
        txt_path = os.path.join(tmpdir, "001.txt")
        open(img_path, "wb").close()
        with open(txt_path, "w") as f:
            f.write("sks, test caption")

        res = client.post("/dataset/scan", json={"dataset_dir": tmpdir})
        assert res.status_code == 200
        data = res.json()
        assert data["found"] == 1
        assert data["pairs"][0]["caption"] == "sks, test caption"


def test_dataset_scan_missing_caption_reported(client):
    with tempfile.TemporaryDirectory() as tmpdir:
        # image without a matching .txt
        open(os.path.join(tmpdir, "no_caption.png"), "wb").close()

        res = client.post("/dataset/scan", json={"dataset_dir": tmpdir})
        assert res.status_code == 200
        data = res.json()
        assert data["found"] == 0
        assert "no_caption.png" in data["missing_captions"]


# ── Train status ──────────────────────────────────────────────────────────────

def test_train_status(client):
    res = client.get("/train/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_train_rejects_missing_dataset(client):
    res = client.post("/train", json={
        "dataset_dir": "/nonexistent/dataset",
        "output_dir": "./outputs/lora",
        "trigger_word": "sks",
        "max_train_steps": 10,
        "learning_rate": 4e-5,
    })
    assert res.status_code == 404


# ── Generate ──────────────────────────────────────────────────────────────────

def test_generate_no_model(client):
    import src.api as api_module
    api_module._pipe = None
    res = client.post("/generate", json={"prompt": "test prompt"})
    assert res.status_code == 503


# ── Authentication ────────────────────────────────────────────────────────────

def test_auth_disabled_allows_requests(client):
    """When API_KEY is empty, all requests pass through."""
    res = client.post("/dataset/scan", json={"dataset_dir": "/nonexistent"})
    # 404 (not found dir) means auth passed, endpoint was reached
    assert res.status_code == 404


def test_auth_rejects_missing_key(authed_client):
    res = authed_client.post("/dataset/scan", json={"dataset_dir": "/any"})
    assert res.status_code == 401


def test_auth_rejects_wrong_key(authed_client):
    res = authed_client.post(
        "/dataset/scan",
        json={"dataset_dir": "/any"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res.status_code == 401


def test_auth_accepts_correct_key(authed_client):
    res = authed_client.post(
        "/dataset/scan",
        json={"dataset_dir": "/nonexistent"},
        headers={"X-API-Key": "test-secret-key"},
    )
    # 404 = auth passed, endpoint reached
    assert res.status_code == 404


def test_auth_readonly_endpoints_unprotected(authed_client):
    """GET endpoints (health, system, config, train/status) are always public."""
    for path in ["/health", "/system", "/config", "/train/status"]:
        res = authed_client.get(path)
        assert res.status_code == 200, f"{path} should be public"


# ── pipeline.py unit tests ────────────────────────────────────────────────────

def test_apply_lora_weights_cpu(tmp_path):
    """apply_lora_weights must not hardcode cuda — should work on CPU."""
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file
    from src.pipeline import LoRALinear, apply_lora_weights

    linear = nn.Linear(8, 8)
    lora = LoRALinear(linear, rank=2, alpha=4)

    # Save a minimal lora state dict
    weights = {k: v for k, v in lora.state_dict().items() if "lora" in k}
    weights_file = tmp_path / "lora_weights.safetensors"
    save_file(weights, str(weights_file))

    with patch("torch.cuda.is_available", return_value=False):
        # Should not raise — previously crashed with device="cuda" on CPU
        apply_lora_weights(lora, str(weights_file))


def test_patch_unet_attention_accepts_dual_phase_kwarg():
    """patch_unet_attention must accept dual_phase kwarg without TypeError."""
    import torch.nn as nn
    from src.pipeline import patch_unet_attention

    # Minimal stand-in for a UNet module with attention attrs
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(16, 16)

    class FakeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FakeAttn()

    unet = FakeUNet()
    # Must not raise TypeError
    params = patch_unet_attention(unet, rank=2, alpha=4, dual_phase=True)
    assert isinstance(params, list)


def test_patch_text_encoder_accepts_dual_phase_kwarg():
    """patch_text_encoder must accept dual_phase kwarg without TypeError."""
    import torch.nn as nn
    from src.pipeline import patch_text_encoder

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)

    enc = FakeEncoder()
    params = patch_text_encoder(enc, rank=2, alpha=4, dual_phase=True)
    assert isinstance(params, list)
