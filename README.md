# Plant disease classification (disease-only)

**Task:** classify **disease** from a leaf/plant image (same disease on different crops maps to one label, e.g. “late blight”), using a **YOLO classification** model trained with Ultralytics — **local inference only** (no paid vision APIs).

Weights are loaded **only from Hugging Face** (downloaded once into the HF cache on your machine or server).

## Repo layout

```
├── app.py              # FastAPI app (/health, /predict)
├── inference.py        # HF download + YOLO cls + top-k disease names
├── requirements.txt
├── Procfile            # Render / similar — uvicorn
├── scripts/
│   └── prefetch_weights.py   # optional: run at Render build to cache HF weights
├── notebooks/
│   └── Yolo_on_metric_dataset.ipynb
└── README.md
```

## Setup

1. Python 3.10+
2. `pip install -r requirements.txt`
3. Set **Hugging Face** env vars (required):

```powershell
$env:HF_MODEL_REPO = "YourUser/YourModel"
$env:HF_MODEL_FILENAME = "best YOLO.pt"
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Use your real repo id and the **exact** filename from the model’s **Files** tab. Private repo: set `HF_TOKEN`.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `HF_MODEL_REPO` | **Required.** Hugging Face model id (`user/model`). |
| `HF_MODEL_FILENAME` | Filename in the repo (default `best.pt`). |
| `HF_MODEL_REVISION` | Optional branch/commit. |
| `HF_TOKEN` | Optional; for private HF repos. |
| `YOLO_LABEL_MAP_PATH` | Optional path to `label_map.json` (default `label_map.json` in project root if present). |

## API

- **Swagger:** `http://127.0.0.1:8000/docs`
- **GET `/health`**
- **POST `/predict`** — image upload; returns `predicted_disease` and `top_predictions`

## Deploy (e.g. Render)

1. **Environment:** **`HF_MODEL_REPO`**, **`HF_MODEL_FILENAME`**, and **`HF_TOKEN`** (HF token = faster downloads, fewer 502s — create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
2. **Build command** (important — downloads weights *during build* so `/predict` is not doing a long download behind a short gateway timeout):

   `pip install -r requirements.txt && python scripts/prefetch_weights.py`

3. **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT` or use the **`Procfile`**.

If **`HF_MODEL_REPO`** is not set during build, prefetch skips (no error); for production you should have the same env vars for **build** and **runtime** on Render.

### If you see “Application startup failed” on Render

- Scroll **up** in the log for the Python error (not only the last line).
- **Missing env:** `Set HF_MODEL_REPO...` / ValueError → add variables in the Render **Environment** section and **Clear build cache & deploy** if needed.
- **404 from Hugging Face:** wrong repo id or filename → match the **Files** tab exactly (e.g. `best YOLO.pt`).
- **401 / private repo:** add **`HF_TOKEN`** with a read token from Hugging Face.
- **Out of memory:** PyTorch + YOLO on the smallest free instance can OOM → upgrade instance or use a smaller model later.

### “No open ports detected”

This app **does not** load the heavy model at process start; the port opens quickly. Use the **build command** above so the weight file is cached before runtime.

### 502 on `/predict` (keeps happening)

| Cause | What to do |
|--------|------------|
| **Gateway timeout** while downloading weights on first request | Use **`HF_TOKEN`**, set **build command** with **`prefetch_weights.py`**, redeploy. |
| **Out of memory** (PyTorch + YOLO) | Check logs for `Killed` / OOM → **upgrade** Render instance (free 512 MB is often too small). |
| **Cold sleep** | First request after idle can be slow — wait and retry **`/health`** then **`/predict`**. |

---

## Submission checklist (course)

| Deliverable | Notes |
|-------------|--------|
| **GitHub** | Invite evaluator if private. |
| **Weights** | Hosted on Hugging Face — link in README/report. |
| **W&B** | Public project URL in README + report. |
| **Report (PDF)** | Method, ablations, metrics. |
| **Live API** | Public `/docs` URL. |

**Next steps:** ensure weights are on HF → set env on the host → deploy → add W&B + API URLs to README.
