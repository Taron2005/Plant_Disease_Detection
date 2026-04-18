"""
Run in Render *build* so weights are already in the HF cache before the web process starts.
Cuts first /predict time and avoids gateway timeouts on slow downloads.
"""
import os
import sys


def main() -> int:
    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo:
        print("prefetch_weights: HF_MODEL_REPO not set — skipping (set it on Render).")
        return 0

    filename = (os.environ.get("HF_MODEL_FILENAME") or "best.pt").strip() or "best.pt"
    revision = (os.environ.get("HF_MODEL_REVISION") or "").strip() or None

    print(f"prefetch_weights: downloading {repo} / {filename} ...")
    from huggingface_hub import hf_hub_download

    kwargs = {"repo_id": repo, "filename": filename}
    if revision:
        kwargs["revision"] = revision
    hf_hub_download(**kwargs)
    print("prefetch_weights: done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("prefetch_weights failed:", e, file=sys.stderr)
        raise
