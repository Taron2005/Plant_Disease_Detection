---
title: Plant Disease API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Plant disease classifier (API)

This repo is the **production side** of a plant-disease project: a **FastAPI** service that runs **Ultralytics YOLO in classification mode** and returns a **disease label** (and top‑k scores) for an uploaded leaf or plant image.

Training used a folder structure where each class came from *plant × disease* combinations; labels were **collapsed to disease-only** so the same pathology maps to one class whether it showed up on different crops. The exported checkpoint (`best.pt` or similar) is **not stored in Git** — it is loaded at runtime from a **Hugging Face model repo** you point to with environment variables.

If you only care about **calling the model**, use **`/docs`** on the deployed Space or run locally after setting `HF_MODEL_REPO`. If you care about **how the model was trained**, see `notebooks/Yolo_on_metric_dataset.ipynb` (data prep, splits, metrics).

## What you get

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Tiny JSON pointer to `/docs`, `/health`, `/predict` |
| `GET /health` | Liveness check |
| `POST /predict` | Multipart file upload → JSON with `predicted_disease` and `top_predictions` |
| `GET /docs` | Swagger UI (OpenAPI) |

Example JSON from `POST /predict`:

```json
{
  "predicted_disease": "late_blight",
  "top_predictions": [
    {"disease": "late_blight", "confidence": 0.91},
    {"disease": "early_blight", "confidence": 0.04},
    {"disease": "healthy", "confidence": 0.02}
  ]
}
```

Strings match whatever class names the checkpoint was trained with (and optional `label_map.json` if you ship one next to the app).

## Repo layout

| Path | Role |
|------|------|
| `app.py` | FastAPI routes, image validation |
| `inference.py` | Download weights from the Hub, load YOLO once, run top‑k classification |
| `requirements.txt` | Python deps for the API |
| `Dockerfile` | Image for **Hugging Face Spaces** (listens on **7860**) |
| `notebooks/Yolo_on_metric_dataset.ipynb` | Training / evaluation notebook (not required to run the API) |

## Environment variables

| Variable | Required | Notes |
|----------|----------|--------|
| `HF_MODEL_REPO` | Yes | Model id, e.g. `YourName/your-weights-repo` |
| `HF_MODEL_FILENAME` | No | File in that repo (default `best.pt`) |
| `HF_MODEL_REVISION` | No | Branch or commit |
| `HF_TOKEN` | For private or gated repos | Also helps with rate limits on downloads |
| `YOLO_LABEL_MAP_PATH` | No | Override path to `label_map.json` for display names |

## Run locally

```bash
pip install -r requirements.txt
export HF_MODEL_REPO="YourName/your-model"
export HF_MODEL_FILENAME="best.pt"   # if different
export HF_TOKEN="hf_..."             # if needed
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Windows (PowerShell):

```powershell
$env:HF_MODEL_REPO = "YourName/your-model"
$env:HF_MODEL_FILENAME = "best.pt"
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs**.

## Deploy on Hugging Face Spaces (Docker)

1. Create a Space with the **Docker** SDK (this repo’s README front matter already says `sdk: docker`).
2. Connect the Space to this GitHub repo **or** push the same tree to the Space’s Git remote.
3. Under **Settings → Variables and secrets**, add at least `HF_MODEL_REPO`, plus `HF_MODEL_FILENAME` / `HF_TOKEN` as needed.
4. Build and open **`https://<user>-<space>.hf.space/docs`**.

The `Dockerfile` runs `uvicorn` on port **7860**, which is what Spaces expects. First request may be slow while weights download into the Hub cache inside the container.

## Troubleshooting

| What you see | What usually fixes it |
|--------------|------------------------|
| Missing `HF_MODEL_REPO` | Set it in the Space / host env, not only locally |
| 404 from Hub | Filename must match the **Files** tab exactly |
| 401 on download | Set `HF_TOKEN` |
| Wrong class names | Add a `label_map.json` in the app working dir or set `YOLO_LABEL_MAP_PATH` |
| OOM on small hardware | Smaller model or a Space with more RAM |

Spaces config reference: [https://huggingface.co/docs/hub/spaces-config-reference](https://huggingface.co/docs/hub/spaces-config-reference)
