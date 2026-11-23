"""Download a pinned pretrained checkpoint into the Docker image at build time."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from api_server import MODEL_ID

MODEL_REVISION = "8196688df56a8d08ccfa186505343adff412ab6b"


def prepare_model(destination: str) -> Path:
    model_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=destination,
            allow_patterns=["config.json", "model.pth", "voices/*.pt", "README.md"],
        )
    )
    required = [model_dir / "config.json", model_dir / "model.pth"]
    if not all(path.is_file() for path in required) or len(list((model_dir / "voices").glob("*.pt"))) != 12:
        raise RuntimeError("Incomplete Wayu checkpoint: expected config, weights and 12 voices")
    print(f"Pretrained model {MODEL_ID}@{MODEL_REVISION} saved to {model_dir}")
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination model directory")
    args = parser.parse_args()
    prepare_model(args.output)
