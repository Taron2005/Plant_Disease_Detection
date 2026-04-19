import os

# Cheap hosts + PyTorch: cap BLAS threads so one request doesn't spawn a thread army
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import json
import sys
import threading
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO


def _patch_main_for_old_checkpoints():
    # Old Colab runs pickled PadToSquareResize on __main__ — without this, load can blow up
    try:
        _bilinear = Image.Resampling.BILINEAR
    except AttributeError:
        _bilinear = Image.BILINEAR

    main = sys.modules["__main__"]
    if not hasattr(main, "PadToSquareResize"):

        class PadToSquareResize:
            def __init__(self, size, fill=(114, 114, 114)):
                self.size = size
                self.fill = fill

            def __call__(self, img):
                return ImageOps.pad(
                    img, (self.size, self.size), color=self.fill, method=_bilinear
                )

        main.PadToSquareResize = PadToSquareResize


LABEL_MAP_PATH = os.environ.get("YOLO_LABEL_MAP_PATH", "label_map.json")

_model = None
_class_id_to_name = None
_load_lock = threading.Lock()


def resolve_weights_path() -> str:
    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo:
        raise ValueError(
            "Set HF_MODEL_REPO to your Hugging Face model id (e.g. User/Model). "
            "Weights are pulled from the Hub, not bundled in the repo."
        )

    if "your-username" in repo or repo == "your-username/your-model-name":
        raise ValueError(
            "HF_MODEL_REPO still looks like a placeholder — use the id from the model page URL."
        )

    from huggingface_hub import hf_hub_download

    filename = (os.environ.get("HF_MODEL_FILENAME") or "best.pt").strip() or "best.pt"
    revision = (os.environ.get("HF_MODEL_REVISION") or "").strip() or None

    download_args = {"repo_id": repo, "filename": filename}
    if revision:
        download_args["revision"] = revision
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        download_args["token"] = token

    try:
        return hf_hub_download(**download_args)
    except Exception as e:
        raise ValueError(
            f'Could not download "{filename}" from "{repo}". '
            "Check the Files tab name, branch, and HF_TOKEN if private."
        ) from e


def _load_label_map(path: str, model_names) -> dict:
    p = Path(path)
    if not p.is_file():
        return dict(model_names)

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {i: str(name) for i, name in enumerate(data)}
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            out[int(k)] = str(v)
        return out

    raise ValueError("label_map.json should be a list or dict index -> name")


def load_model(label_map_path: str | None = None):
    global _model, _class_id_to_name

    wpath = resolve_weights_path()
    lpath = label_map_path or LABEL_MAP_PATH

    _patch_main_for_old_checkpoints()
    _model = YOLO(wpath)
    _class_id_to_name = _load_label_map(lpath, _model.names)


def get_model():
    # Lazy load: uvicorn binds the port before we drag in torch + weights
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                load_model()
    return _model


def get_class_names():
    get_model()
    return _class_id_to_name


def predict_top_k(image_bytes: bytes, k: int = 3) -> tuple[str, list[dict]]:
    model = get_model()
    names = get_class_names()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=img, verbose=False)
    r = results[0]
    if r.probs is None:
        raise ValueError("Not a classification checkpoint — need a YOLO cls model (best.pt)")

    probs = r.probs
    idx = probs.top5
    conf = probs.top5conf
    if hasattr(idx, "tolist"):
        idx = idx.tolist()
    if hasattr(conf, "tolist"):
        conf = conf.tolist()

    k = min(k, len(idx))
    top_list = []
    for i in range(k):
        class_id = int(idx[i])
        label = names.get(class_id, str(class_id))
        score = float(conf[i])
        top_list.append({"disease": label, "confidence": round(score, 4)})

    best = top_list[0]["disease"]
    return best, top_list
