<<<<<<< HEAD
# Plant disease classification API

REST API for **disease-only** plant image classification using an **Ultralytics YOLO classification** model. Inference runs on your own hardware or cloud instance; weights are loaded from **Hugging Face** (cached after the first download).

## Features

- **POST `/predict`** — upload an image; receive the top predicted disease label and top-3 candidates with confidence scores  
- **GET `/health`** — service health check  
- **OpenAPI / Swagger UI** at **`/docs`**

## Repository layout

| Path | Description |
|------|-------------|
| `app.py` | FastAPI application |
| `inference.py` | Model loading (Hugging Face) and inference |
| `requirements.txt` | Python dependencies |
| `Procfile` | Process definition for platforms like Render (`uvicorn` on `$PORT`) |
| `Dockerfile` | Container image for Hugging Face Spaces or other Docker hosts |
| `scripts/prefetch_weights.py` | Optional build step to download weights before the web server starts |
| `notebooks/` | Training and dataset preparation notebook |

## Requirements

- Python 3.10 or newer  
- A Hugging Face **model repository** containing the weight file (e.g. `best.pt` or a custom filename)

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_MODEL_REPO` | Yes | Hugging Face model id: `organization/model-name` |
| `HF_MODEL_FILENAME` | No | File name in that repo (default: `best.pt`) |
| `HF_MODEL_REVISION` | No | Branch or commit hash |
| `HF_TOKEN` | If the repo is private | [Access token](https://huggingface.co/settings/tokens) with read access; also improves download reliability on rate limits |
| `YOLO_LABEL_MAP_PATH` | No | Path to optional `label_map.json` for class names (defaults to `label_map.json` in the working directory if present) |

## Local run

Install dependencies, set the variables, then start the server from the project root:

```bash
pip install -r requirements.txt
export HF_MODEL_REPO="your-org/your-model"
export HF_MODEL_FILENAME="your-weights.pt"
export HF_TOKEN="hf_..."   # if needed
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

On Windows (PowerShell):

```powershell
$env:HF_MODEL_REPO = "your-org/your-model"
$env:HF_MODEL_FILENAME = "your-weights.pt"
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** to use the interactive API.

## Example response (`POST /predict`)

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

Label strings match the classes used when the model was trained.

## Deployment (e.g. Render)

1. Connect this repository to the hosting provider.  
2. Set the **same** environment variables in the service settings (`HF_MODEL_REPO`, `HF_MODEL_FILENAME`, and `HF_TOKEN` when applicable).  
3. **Build command** (recommended so weights are downloaded during build, not on the first user request):

   ```bash
   pip install -r requirements.txt && python scripts/prefetch_weights.py
   ```

4. **Start command:**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

   Or rely on the included **`Procfile`** if the platform supports it.

5. After deploy, the interactive documentation is at **`https://<your-host>/docs`**.

### Hugging Face Spaces (Docker)

Free CPU Spaces often give **more RAM** than small free tiers on some other hosts, which helps PyTorch + YOLO. Limits depend on Hugging Face’s current policy.

1. Create a **new Space** → choose **Docker** (not Gradio).
2. Point the Space at this GitHub repository (or push this repo to Hugging Face).
3. Open **Settings → Variables and secrets** and add:
   - `HF_MODEL_REPO`
   - `HF_MODEL_FILENAME`
   - `HF_TOKEN` (recommended for reliable downloads; required if the weight file is in a private repo)
4. Build and run. The `Dockerfile` serves the API on port **7860** (Spaces default).
5. Open **`https://<your-username>-<space-name>.hf.space/docs`** for Swagger.

If the Space still reports **out of memory**, try a **CPU upgrade** in the Space hardware settings (paid) or reduce model size.

## Troubleshooting

| Symptom | Suggestion |
|---------|------------|
| Error about missing `HF_MODEL_REPO` | Define all required variables in the **runtime** environment of the host. |
| 404 from Hugging Face | Use the exact file name shown under the model’s **Files** tab (including spaces or capitalization). |
| 401 / authentication | Set `HF_TOKEN` for private repositories. |
| 502 or timeout on first prediction | Add `HF_TOKEN`; use the build command with `prefetch_weights.py`; allow extra time after cold start on free tiers. |
| Process killed / out of memory | PyTorch and the model may exceed small instances; use a plan with more RAM or a smaller model. |
| Ultralytics config directory warning | Harmless fallback to `/tmp`; optionally set `YOLO_CONFIG_DIR` to a writable path (e.g. `/tmp/Ultralytics`). |
=======
---
title: Plant Disease Api
emoji: 📉
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> cff3695ec2b2caefd50e9a55116a0ea21bb46c51
