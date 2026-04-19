---
title: Plant Disease API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Plant Disease API

Small **FastAPI** service for plant **disease** labels from a photo. It uses a **YOLO classification** checkpoint (Ultralytics). Weights are **not** in this repo: you point the app at a file on the **Hugging Face Hub** with env vars.

What it returns: the best disease class plus a short list of top scores (`POST /predict`). Swagger is at **`/docs`**.

## Endpoints

| Method | Path | What it does |
|--------|------|----------------|
| GET | `/` | Basic info + where `/docs` is |
| GET | `/health` | Health check |
| POST | `/predict` | Upload an image file, get JSON |
| GET | `/docs` | Swagger UI |

Example response:

```json
{
  "predicted_disease": "mosaic",
  "top_predictions": [
    {"disease": "mosaic", "confidence": 0.88},
    {"disease": "late_blight", "confidence": 0.07},
    {"disease": "early_blight", "confidence": 0.03}
  ]
}
```

Class names come from the **model** by default. If you add a **`label_map.json`** next to the app (or set `YOLO_LABEL_MAP_PATH`), that file overrides names. Nothing downloads `label_map.json` from the Hub; only the `.pt` file is downloaded from the model repo.

## Reports (PDFs on GitHub)

This Space git copy **does not** include PDFs (Hub storage rules). The course reports are on **GitHub** (`main` branch):

**https://github.com/Taron2005/Plant_Disease_Detection/tree/main/Reports**

| File | What |
|------|------|
| `plant_disease_full_report.pdf` | Full technical write-up |
| `plant_disease_technical_report_archived.pdf` | Technical report (archived copy) |

## Notebooks (`notebooks/`)

Jupyter notebooks where **experiments for this task** were run: dataset handling, training runs, and alternative setups (e.g. different model families). They are **not** needed to run the deployed API; they document what was tried and how the final model was produced.

## Git branches

- **`main` (GitHub)** — full project: API code + `Reports/` PDFs + everything else.
- **`hf-deploy`** — **this** branch: same API as `main`, but **no** `Reports/` folder, so pushes to the Space git succeed.
- **`experiments`** — large Colab notebooks and extra experiment artifacts (see GitHub).

To work with the full notebooks locally:

```bash
git fetch origin
git checkout experiments
```

Or copy only the notebooks folder from that branch without switching branches:

```bash
git checkout experiments -- notebooks/
```

After editing on `experiments`, run `git push origin experiments`. To update **this Space** after you change code on `main`: merge or cherry-pick into `hf-deploy`, drop `Reports/` if present, then run `git push hf hf-deploy:main`.

## Files in this repo (API)

| File | What |
|------|------|
| `app.py` | Routes and upload checks |
| `inference.py` | Load weights from Hub, run YOLO, top-k output |
| `requirements.txt` | Dependencies |
| `Dockerfile` | For **Hugging Face Spaces** (Docker), port **7860** |
| `Reports/` | (GitHub `main` only) PDF reports — link in section above |
| `notebooks/` | Experiment notebooks for the task (see above) |

## Env vars

| Variable | Required | Meaning |
|----------|----------|---------|
| `HF_MODEL_REPO` | yes | Hub model id, e.g. `user/model` |
| `HF_MODEL_FILENAME` | no | Weight file name (default `best.pt`) |
| `HF_MODEL_REVISION` | no | Branch or commit |
| `HF_TOKEN` | if private | Read token for the Hub |
| `YOLO_LABEL_MAP_PATH` | no | Path to optional local `label_map.json` |

## Run locally

```bash
pip install -r requirements.txt
export HF_MODEL_REPO="user/model"
export HF_MODEL_FILENAME="best.pt"
export HF_TOKEN="hf_..."   # only if the repo is private
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Windows (PowerShell):

```powershell
$env:HF_MODEL_REPO = "user/model"
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/docs**.

## Hugging Face Space (Docker)

1. Create a Space with **Docker** (this README header uses `sdk: docker`).
2. Connect the Space to this repo’s **`hf-deploy`** branch (or push `hf-deploy` to the Space git remote as `main`; see Reports section).
3. In **Settings → Variables and secrets**, set `HF_MODEL_REPO` and anything else you need from the table above.
4. After build, open **`https://<your-username>-<space-name>.hf.space/docs`**.

The `Dockerfile` runs uvicorn on **7860**. First request can be slow while weights download.

## If something breaks

- **Missing `HF_MODEL_REPO`** — set it in the Space settings, not only on your laptop.
- **404 from Hub** — filename must match the model repo **Files** tab exactly.
- **401** — set `HF_TOKEN` for private repos.
- **Out of memory** — smaller model or a Space with more RAM.

Hub Spaces docs: [Spaces config](https://huggingface.co/docs/hub/spaces-config-reference)
