import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import inference

app = FastAPI(title="Plant disease classifier", version="1.0")


@app.on_event("startup")
def startup():
    # load model once when app starts
    inference.load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # make sure uploaded file is an image
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image file")

    # reopen after verify() — verify() leaves the buffer in a bad state for some formats
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
