# Quickstart Guide

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA 12.1+ (CPU works but training is very slow)
- ~10 GB free disk space (for base model)

---

## 1. Clone the repo

```bash
git clone https://github.com/sukhmansaran/fine-tuning-stable-diffusion-models-lora-dreambooth.git
cd fine-tuning-stable-diffusion-models-lora-dreambooth
```

---

## 2. Set up Python environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

Install dependencies — install torch first so xformers picks the right CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install xformers
pip install -r requirements.txt
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce ...
```

---

## 3. Download the base model

You need a free Hugging Face account. Log in once:

```bash
huggingface-cli login
```

Then download (~7 GB):

```bash
huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE --local-dir ./models/base
```

---

## 4. Prepare your dataset

Create a folder anywhere on your machine with images and matching caption files:

```
my_dataset/
├── 001.jpg
├── 001.txt    ← "sks, portrait photo, natural lighting"
├── 002.jpg
├── 002.txt    ← "sks, full body shot, outdoor background"
└── ...
```

Rules:
- Every image needs a `.txt` file with the **same base name**
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- Include your trigger word in every caption (e.g. `sks`)
- 10–30 images is enough for good results

---

## 5. Start the API server

In your Python virtual environment:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Leave this terminal running. API docs available at http://localhost:8000/docs

---

## 6. Start the frontend

Open a **second terminal**:

```bash
cd frontend
npm install
npm start
```

Opens at **http://localhost:3000**

---

## 7. Use the UI

### Tab 1 — Dataset

1. Type the full path to your dataset folder, e.g.:
   - Windows: `D:\my_dataset`
   - Mac/Linux: `/home/user/my_dataset`
2. Click **Scan** — previews all found image/caption pairs and warns about any missing captions
3. Click **Use this dataset** to proceed

### Tab 2 — Fine-Tune

1. Your dataset path is pre-filled
2. Set a **trigger word** — a short unique token like `sks` that represents your subject
3. Set **output directory** — where checkpoints will be saved (default: `./outputs/lora`)
4. Adjust training metrics if needed (GPU defaults are auto-detected):

   | Setting | Default | Notes |
   |---|---|---|
   | Train Steps | 500 | ~20–30× your image count is a good rule |
   | Learning Rate | 4e-5 | Lower = more stable, slower |
   | LR Scheduler | cosine | cosine works well for most cases |
   | LoRA Rank | 4 | Higher = more expressive, more VRAM |
   | LoRA Alpha | 8 | Usually 2× rank |
   | Resolution | 512 | Use 768 if you have 16 GB+ VRAM |
   | Batch Size | 1 | Keep at 1 for ≤16 GB VRAM |
   | Grad Accumulation | 4 | Effective batch = batch × accum |
   | Mixed Precision | fp16 | Auto-set based on your GPU |

5. Click **Start Training** — watch the live progress bar and loss value
6. When status shows **done**, click **Load Model → Generate**

### Tab 3 — Generate

1. Write a prompt using your trigger word, e.g.:
   ```
   sks, cyberpunk portrait, cinematic lighting, 8k
   ```
2. Optionally click **+ inject "sks"** to prepend the trigger word automatically
3. Adjust generation settings (size, steps, guidance scale, seed)
4. Click **Generate** — images appear below with download links

---

## Troubleshooting

**`react-scripts` not found**
```bash
cd frontend
Remove-Item -Recurse -Force node_modules   # Windows
rm -rf node_modules                         # Mac/Linux
npm install
npm start
```

**CUDA out of memory**
- Lower resolution to 512
- Keep batch size at 1
- Reduce LoRA rank to 2 or 4

**Model not loading on startup**
- The API skips model loading if `./models/base` doesn't exist yet — this is normal
- After training completes, use the "Load Model" button in the UI

**Dataset scan returns 0 pairs**
- Make sure each image has a `.txt` file with the exact same base name
- Check the path is correct and accessible
