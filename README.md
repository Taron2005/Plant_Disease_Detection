# Plant disease classification (disease-only)

**Task:** classify **disease** from a leaf/plant image (same disease on different crops maps to one label, e.g. “late blight”), using a **YOLO classification** model trained with Ultralytics — **local inference only** (no paid vision APIs).

Weights are loaded **only from Hugging Face** (downloaded once into the HF cache on your machine or server).

## Repo layout

```
├── app.py              # FastAPI app (/health, /predict)
├── inference.py        # HF download + YOLO cls + top-k disease names
├── requirements.txt
├── Procfile            # Render / similar — uvicorn
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

Set `HF_MODEL_REPO`, `HF_MODEL_FILENAME`, and `HF_TOKEN` if needed. Start command: `Procfile` or `uvicorn app:app --host 0.0.0.0 --port $PORT`.

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
