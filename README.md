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

## Troubleshooting

| Symptom | Suggestion |
|---------|------------|
| Error about missing `HF_MODEL_REPO` | Define all required variables in the **runtime** environment of the host. |
| 404 from Hugging Face | Use the exact file name shown under the model’s **Files** tab (including spaces or capitalization). |
| 401 / authentication | Set `HF_TOKEN` for private repositories. |
| 502 or timeout on first prediction | Add `HF_TOKEN`; use the build command with `prefetch_weights.py`; allow extra time after cold start on free tiers. |
| Process killed / out of memory | PyTorch and the model may exceed small instances; use a plan with more RAM or a smaller model. |
| Ultralytics config directory warning | Harmless fallback to `/tmp`; optionally set `YOLO_CONFIG_DIR` to a writable path (e.g. `/tmp/Ultralytics`). |
