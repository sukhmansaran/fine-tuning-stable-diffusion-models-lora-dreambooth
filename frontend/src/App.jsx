import { useState, useEffect, useCallback, useRef } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "/api";
const post = (url, body) =>
  fetch(`${API}${url}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const DEFAULT_TRAIN = {
  dataset_dir: "",
  output_dir: "./outputs/lora",
  trigger_word: "sks",
  trigger_init_word: "person",
  max_train_steps: 500,
  learning_rate: 0.00004,
  lr_scheduler: "cosine",
  lr_warmup_steps: 50,
  lora_r: 4,
  lora_alpha: 8,
  resolution: 512,
  batch_size: 1,
  gradient_accumulation: 4,
  mixed_precision: "fp16",
  train_text_encoder: true,
  train_feedforward: false,
  save_every_n_steps: 250,
  seed: 42,
};

const DEFAULT_GEN = {
  prompt: "", negative_prompt: "", num_images: 1,
  height: 512, width: 512, steps: 30,
  guidance_scale: 6.5, seed: 42, phase2_weight: 0.3,
};

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ health }) {
  const labels = { ok: "Model ready", loading: "Model loading…", error: "API unreachable" };
  return (
    <div className={`status status--${health}`}>
      <span className="status-dot" />
      {labels[health] ?? "Checking…"}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 1 — Dataset
// ══════════════════════════════════════════════════════════════════════════════
function DatasetTab({ datasetDir, setDatasetDir, onConfirmed }) {
  const [scanning, setScanning]   = useState(false);
  const [scanResult, setScanResult] = useState(null); // {found, missing_captions, pairs}
  const [error, setError]         = useState(null);

  const scan = async () => {
    if (!datasetDir.trim()) return;
    setScanning(true); setError(null); setScanResult(null);
    try {
      const res = await post("/dataset/scan", { dataset_dir: datasetDir.trim() });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setScanResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setScanning(false); }
  };

  const handleKey = (e) => { if (e.key === "Enter") scan(); };

  return (
    <div className="tab-content">
      <p className="tab-desc">
        Point to a local folder that contains your training images and matching <code>.txt</code> caption files.
        Each image must have a <code>.txt</code> file with the same base name.
      </p>

      <div className="path-row">
        <div className="field full">
          <label>Dataset Directory (local path)</label>
          <input
            type="text"
            placeholder="e.g. /home/user/my_dataset  or  C:\datasets\my_subject"
            value={datasetDir}
            onChange={e => { setDatasetDir(e.target.value); setScanResult(null); }}
            onKeyDown={handleKey}
          />
        </div>
        <button className="btn" onClick={scan} disabled={scanning || !datasetDir.trim()}>
          {scanning ? "Scanning…" : "Scan"}
        </button>
      </div>

      {error && <div className="error" role="alert">{error}</div>}

      {scanResult && (
        <>
          <div className="scan-summary">
            <span className="scan-ok">✓ {scanResult.found} image/caption pair{scanResult.found !== 1 ? "s" : ""} found</span>
            {scanResult.missing_captions.length > 0 && (
              <span className="scan-warn">
                ⚠ {scanResult.missing_captions.length} image{scanResult.missing_captions.length !== 1 ? "s" : ""} missing captions (will be skipped)
              </span>
            )}
          </div>

          {scanResult.missing_captions.length > 0 && (
            <div className="missing-list">
              <div className="section-header small">Images without captions</div>
              {scanResult.missing_captions.map(f => (
                <div key={f} className="missing-item">
                  <span className="warn-icon">⚠</span> {f}
                  <span className="hint"> — create a matching <code>{f.replace(/\.[^.]+$/, ".txt")}</code></span>
                </div>
              ))}
            </div>
          )}

          {scanResult.pairs.length > 0 && (
            <>
              <div className="section-header">Preview <span className="muted">({scanResult.pairs.length} pairs)</span></div>
              <div className="preview-list">
                {scanResult.pairs.map((p, i) => (
                  <div key={i} className="preview-row">
                    <span className="preview-filename">{p.image}</span>
                    <span className="preview-caption">{p.caption}</span>
                  </div>
                ))}
              </div>

              <button
                className="btn"
                onClick={() => onConfirmed(datasetDir.trim())}
                disabled={scanResult.found === 0}
              >
                Use this dataset → go to Train
              </button>
            </>
          )}
        </>
      )}

      <div className="format-hint">
        <div className="section-header small">Expected folder structure</div>
        <pre className="code-block">{`my_dataset/
├── 001.jpg
├── 001.txt   ← "sks, portrait photo, natural lighting"
├── 002.png
├── 002.txt   ← "sks, full body shot, outdoor"
└── ...`}</pre>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 2 — Train
// ══════════════════════════════════════════════════════════════════════════════
function TrainTab({ datasetDir, onModelLoaded }) {
  const [cfg, setCfg]         = useState({ ...DEFAULT_TRAIN, dataset_dir: datasetDir });
  const [sysInfo, setSysInfo] = useState(null);
  const [status, setStatus]   = useState(null);
  const [error, setError]     = useState(null);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef(null);

  // Keep dataset_dir in sync if user goes back and changes it
  useEffect(() => {
    setCfg(p => ({ ...p, dataset_dir: datasetDir }));
  }, [datasetDir]);

  useEffect(() => {
    fetch(`${API}/system`).then(r => r.json()).then(info => {
      setSysInfo(info);
      setCfg(p => ({
        ...p,
        mixed_precision: info.recommended_precision,
        max_train_steps: info.recommended_steps,
        batch_size: info.recommended_batch,
      }));
    }).catch(() => {});
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const d = await fetch(`${API}/train/status`).then(r => r.json());
        setStatus(d);
        if (d.status === "done" || d.status === "error") {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {}
    }, 1500);
  }, []);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const set = (key) => (e) => {
    const val = e.target.type === "checkbox" ? e.target.checked
      : e.target.type === "number" ? Number(e.target.value)
      : e.target.value;
    setCfg(p => ({ ...p, [key]: val }));
  };

  const startTrain = async () => {
    setError(null); setLoading(true);
    try {
      const res = await post("/train", cfg);
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setStatus({ status: "queued", step: 0, total: cfg.max_train_steps });
      startPolling();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const loadModel = async () => {
    setError(null);
    try {
      const res = await post("/train/load", {});
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      onModelLoaded();
    } catch (e) { setError(e.message); }
  };

  const isRunning = ["queued", "loading_model", "training"].includes(status?.status);
  const isDone    = status?.status === "done";
  const progress  = status?.total ? Math.round((status.step / status.total) * 100) : 0;

  return (
    <div className="tab-content">
      {sysInfo && (
        <div className="sys-banner">
          {sysInfo.has_cuda
            ? `GPU: ${sysInfo.gpu_name} · ${sysInfo.vram_gb} GB VRAM · ${sysInfo.recommended_precision} precision`
            : "No GPU detected — training will run on CPU (slow)"}
        </div>
      )}

      {/* Paths */}
      <div className="section-header">Paths</div>
      <div className="grid grid-2">
        <div className="field">
          <label>Dataset Directory</label>
          <input type="text" value={cfg.dataset_dir} onChange={set("dataset_dir")}
            placeholder="/path/to/your/images" />
          <span className="hint">Folder with images + .txt captions</span>
        </div>
        <div className="field">
          <label>Output Directory</label>
          <input type="text" value={cfg.output_dir} onChange={set("output_dir")}
            placeholder="./outputs/lora" />
          <span className="hint">Where checkpoints will be saved</span>
        </div>
      </div>

      {/* Subject */}
      <div className="section-header">Subject</div>
      <div className="grid grid-2">
        <div className="field">
          <label>Trigger Word</label>
          <input type="text" value={cfg.trigger_word} onChange={set("trigger_word")} />
          <span className="hint">Unique token for your subject (e.g. sks)</span>
        </div>
        <div className="field">
          <label>Init Word</label>
          <input type="text" value={cfg.trigger_init_word} onChange={set("trigger_init_word")} />
          <span className="hint">Initialise trigger from this word (e.g. person)</span>
        </div>
      </div>

      {/* Core metrics */}
      <div className="section-header">Core Metrics</div>
      <div className="grid">
        <div className="field">
          <label>Train Steps</label>
          <input type="number" min={50} max={10000} value={cfg.max_train_steps} onChange={set("max_train_steps")} />
        </div>
        <div className="field">
          <label>Learning Rate</label>
          <input type="number" step={0.000001} min={0.000001} max={0.01} value={cfg.learning_rate} onChange={set("learning_rate")} />
        </div>
        <div className="field">
          <label>LR Scheduler</label>
          <select value={cfg.lr_scheduler} onChange={set("lr_scheduler")}>
            <option value="cosine">cosine</option>
            <option value="linear">linear</option>
            <option value="constant">constant</option>
            <option value="cosine_with_restarts">cosine restarts</option>
          </select>
        </div>
        <div className="field">
          <label>Warmup Steps</label>
          <input type="number" min={0} max={1000} value={cfg.lr_warmup_steps} onChange={set("lr_warmup_steps")} />
        </div>
      </div>

      {/* LoRA */}
      <div className="section-header">LoRA Settings</div>
      <div className="grid">
        <div className="field">
          <label>LoRA Rank (r)</label>
          <input type="number" min={1} max={64} value={cfg.lora_r} onChange={set("lora_r")} />
          <span className="hint">Higher = more capacity, more VRAM</span>
        </div>
        <div className="field">
          <label>LoRA Alpha</label>
          <input type="number" min={1} max={128} value={cfg.lora_alpha} onChange={set("lora_alpha")} />
          <span className="hint">Scaling (usually 2× rank)</span>
        </div>
        <div className="field">
          <label>Resolution</label>
          <select value={cfg.resolution} onChange={set("resolution")}>
            <option value={256}>256</option>
            <option value={512}>512</option>
            <option value={768}>768</option>
          </select>
        </div>
        <div className="field">
          <label>Save Every N Steps</label>
          <input type="number" min={50} max={2000} value={cfg.save_every_n_steps} onChange={set("save_every_n_steps")} />
        </div>
      </div>

      {/* System */}
      <div className="section-header">System</div>
      <div className="grid">
        <div className="field">
          <label>Batch Size</label>
          <input type="number" min={1} max={8} value={cfg.batch_size} onChange={set("batch_size")} />
        </div>
        <div className="field">
          <label>Grad Accumulation</label>
          <input type="number" min={1} max={16} value={cfg.gradient_accumulation} onChange={set("gradient_accumulation")} />
          <span className="hint">Effective batch = batch × accum</span>
        </div>
        <div className="field">
          <label>Mixed Precision</label>
          <select value={cfg.mixed_precision} onChange={set("mixed_precision")} disabled={!sysInfo?.has_cuda}>
            <option value="fp16">fp16</option>
            <option value="bf16" disabled={!sysInfo?.bf16}>bf16</option>
            <option value="no">none (fp32)</option>
          </select>
        </div>
        <div className="field">
          <label>Seed</label>
          <input type="number" value={cfg.seed} onChange={set("seed")} />
        </div>
      </div>

      <div className="checkbox-row">
        <label>
          <input type="checkbox" checked={cfg.train_text_encoder} onChange={set("train_text_encoder")} />
          Also fine-tune text encoder
        </label>
      </div>
      <div className="checkbox-row">
        <label>
          <input type="checkbox" checked={cfg.train_feedforward} onChange={set("train_feedforward")} />
          Also train feedforward (MLP) layers
        </label>
        {cfg.train_feedforward && (
          <span className="hint"> — more expressive, ~2 GB extra VRAM</span>
        )}
      </div>

      <div className="action-row">
        <button className="btn" onClick={startTrain} disabled={isRunning || loading || !cfg.dataset_dir.trim()}>
          {isRunning ? "Training…" : "Start Training"}
        </button>
        {isDone && (
          <button className="btn btn-secondary" onClick={loadModel}>
            Load Model → Generate
          </button>
        )}
      </div>

      {status && status.status !== "idle" && (
        <div className="train-status">
          <div className="status-row">
            <span className={`train-badge train-badge--${status.status}`}>{status.status}</span>
            {status.status === "training" && (
              <span className="step-info">
                Step {status.step} / {status.total}
                {status.loss !== null && ` · loss ${status.loss}`}
              </span>
            )}
          </div>
          {status.status === "training" && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          )}
          {isDone    && <div className="success">✓ Training complete. Click "Load Model → Generate" above.</div>}
          {status.status === "error" && <div className="error" role="alert"><pre>{status.error}</pre></div>}
        </div>
      )}

      {error && <div className="error" role="alert">{error}</div>}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 3 — Generate
// ══════════════════════════════════════════════════════════════════════════════
function GenerateTab({ health, modelReady }) {
  const [params, setParams]   = useState(DEFAULT_GEN);
  const [images, setImages]   = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [triggerWord, setTriggerWord] = useState("");
  const [dualPhase, setDualPhase]     = useState(false);

  useEffect(() => {
    fetch(`${API}/config`).then(r => r.json()).then(d => {
      if (d.trigger_word) setTriggerWord(d.trigger_word);
      if (d.dual_phase)   setDualPhase(true);
    }).catch(() => {});
  }, []);

  const set = (key) => (e) =>
    setParams(p => ({ ...p, [key]: e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const randomSeed = () => setParams(p => ({ ...p, seed: Math.floor(Math.random() * 2 ** 32) }));

  const injectTrigger = useCallback(() => {
    if (!triggerWord) return;
    setParams(p => ({
      ...p,
      prompt: p.prompt.startsWith(triggerWord)
        ? p.prompt
        : `${triggerWord}, ${p.prompt}`.trim().replace(/^,\s*/, ""),
    }));
  }, [triggerWord]);

  const generate = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setImages([]);
    try {
      const res = await post("/generate", params);
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setImages((await res.json()).images);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="tab-content">
      {!modelReady && (
        <div className="info-banner">
          Complete training and click "Load Model → Generate" in the Train tab first.
        </div>
      )}

      <form className="form" onSubmit={generate}>
        <div className="field full">
          <div className="label-row">
            <label htmlFor="prompt">Prompt</label>
            {triggerWord && (
              <button type="button" className="pill-btn" onClick={injectTrigger}>
                + inject "{triggerWord}"
              </button>
            )}
          </div>
          <textarea id="prompt" rows={3}
            placeholder={triggerWord ? `${triggerWord}, portrait in cyberpunk style…` : "portrait in cyberpunk style…"}
            value={params.prompt} onChange={set("prompt")} required />
        </div>

        <div className="field full">
          <label htmlFor="neg">Negative Prompt</label>
          <textarea id="neg" rows={2} placeholder="blurry, low quality…"
            value={params.negative_prompt} onChange={set("negative_prompt")} />
        </div>

        <div className="grid">
          {[
            ["Images",         "num_images",    {min:1,max:8}],
            ["Width",          "width",          {min:256,max:1024,step:64}],
            ["Height",         "height",         {min:256,max:1024,step:64}],
            ["Steps",          "steps",          {min:10,max:100}],
            ["Guidance Scale", "guidance_scale", {min:1,max:20,step:0.5}],
          ].map(([label, key, attrs]) => (
            <div className="field" key={key}>
              <label>{label}</label>
              <input type="number" {...attrs} value={params[key]} onChange={set(key)} />
            </div>
          ))}
          <div className="field">
            <label>Seed</label>
            <div className="seed-row">
              <input type="number" value={params.seed} onChange={set("seed")} />
              <button type="button" className="icon-btn" onClick={randomSeed} title="Random seed">⟳</button>
            </div>
          </div>
          {dualPhase && (
            <div className="field">
              <label>Phase 2 Weight</label>
              <input type="number" step={0.05} min={0} max={1} value={params.phase2_weight} onChange={set("phase2_weight")} />
              <span className="hint">0 = face · 1 = body</span>
            </div>
          )}
        </div>

        <button className="btn" type="submit" disabled={loading || !modelReady}>
          {loading ? "Generating…" : "Generate"}
        </button>
      </form>

      {error && <div className="error" role="alert">{error}</div>}

      {images.length > 0 && (
        <div className="gallery">
          {images.map((b64, i) => (
            <div key={i} className="img-card">
              <img src={`data:image/png;base64,${b64}`} alt={`Generated ${i + 1}`} />
              <a className="download" href={`data:image/png;base64,${b64}`} download={`generated_${i + 1}.png`}>
                ↓ Download
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Root
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab]               = useState("dataset");
  const [health, setHealth]         = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [datasetDir, setDatasetDir] = useState("");

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => { setHealth(d.model_loaded ? "ok" : "loading"); if (d.model_loaded) setModelReady(true); })
      .catch(() => setHealth("error"));
  }, []);

  const TABS = [
    { id: "dataset",  label: "1 · Dataset" },
    { id: "train",    label: "2 · Fine-Tune" },
    { id: "generate", label: "3 · Generate" },
  ];

  return (
    <div className="app">
      <header className="header">
        <h1>SD LoRA Studio</h1>
        <p>Point to your dataset · fine-tune · generate</p>
        <StatusBadge health={health} />
      </header>

      <nav className="tabs">
        {TABS.map(t => (
          <button key={t.id} className={`tab-btn ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "dataset"  && (
        <DatasetTab
          datasetDir={datasetDir}
          setDatasetDir={setDatasetDir}
          onConfirmed={(dir) => { setDatasetDir(dir); setTab("train"); }}
        />
      )}
      {tab === "train"    && (
        <TrainTab
          datasetDir={datasetDir}
          onModelLoaded={() => { setModelReady(true); setHealth("ok"); setTab("generate"); }}
        />
      )}
      {tab === "generate" && <GenerateTab health={health} modelReady={modelReady} />}
    </div>
  );
}
