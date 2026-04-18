"""
Local YOLO classification inference. Keeps disease labels only (from the model / label map).
"""

import os

# Small hosts (e.g. Render free) behave better with fewer BLAS threads
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
    # checkpoints trained in Colab pickle __main__.PadToSquareResize — keep same as training
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

# optional: path to label_map.json if you want names different from the checkpoint (file can sit in repo root)
LABEL_MAP_PATH = os.environ.get("YOLO_LABEL_MAP_PATH", "label_map.json")

_model = None
_class_id_to_name = None
_load_lock = threading.Lock()


def resolve_weights_path() -> str:
    # Weights always come from Hugging Face (cached under ~/.cache/huggingface/hub after first download).

    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo:
        raise ValueError(
            "Set HF_MODEL_REPO to your Hugging Face model id (e.g. User/Model). "
            "This app does not load weights from a local folder."
        )

    if "your-username" in repo or repo == "your-username/your-model-name":
        raise ValueError(
            "HF_MODEL_REPO still looks like the README placeholder. "
            'Use your real id from the model page URL (e.g. "SomeUser/SomeModel").'
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
            f'Could not download "{filename}" from Hugging Face repo "{repo}". '
            "Check repo id and filename (exact match on the Files tab). "
            "Private or gated repo? Set HF_TOKEN in the host environment."
        ) from e


def _load_label_map(path: str, model_names) -> dict:
    """Build index -> disease name. Prefer label_map.json if present."""
    p = Path(path)
    if not p.is_file():
        return dict(model_names)

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {i: str(name) for i, name in enumerate(data)}
    if isinstance(data, dict):
        # keys might be strings "0", "1", ...
        out = {}
        for k, v in data.items():
            out[int(k)] = str(v)
        return out

    raise ValueError("label_map.json should be a list or a dict of index -> name")


def load_model(label_map_path: str | None = None):
    """Load weights once and set up class names."""
    global _model, _class_id_to_name

    wpath = resolve_weights_path()
    lpath = label_map_path or LABEL_MAP_PATH

    _patch_main_for_old_checkpoints()
    _model = YOLO(wpath)
    _class_id_to_name = _load_label_map(lpath, _model.names)


def get_model():
    # Load on first use so the web server binds its port right away (Render checks for an open port quickly).
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
    """
    Run inference on raw image bytes.
    Returns (top_label, top_k list of {disease, confidence}).
    """
    model = get_model()
    names = get_class_names()

    # PIL verifies we have a real image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=img, verbose=False)
    r = results[0]
    if r.probs is None:
        raise ValueError("Model output is not classification — check that best.pt is a cls model")

    probs = r.probs
    # top5 is available on classification probs in ultralytics
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
