import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import inference

app = FastAPI(title="Plant Disease API", version="1.0")


@app.get("/")
def root():
    # some hosts ping "/" before anything else
    return {
        "service": "plant-disease-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart image)",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image file")

    # PIL verify() can leave things in a weird state; open again for real work
    try:
        Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    try:
        predicted_disease, top_predictions = inference.predict_top_k(raw, k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "predicted_disease": predicted_disease,
        "top_predictions": top_predictions,
    }
