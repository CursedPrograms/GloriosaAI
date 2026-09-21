"""Shared helpers: project paths, settings and image discovery (no TensorFlow import)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
IMAGE_SIZE = 128
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

DEFAULT_SETTINGS = {
    "epochs": 10000,
    "batch_size": 16,
    "latent_dim": 128,
    "generation_interval": 25,
    "checkpoint_interval": 500,
    "learning_rate": 0.0002,
    "use_learning_rate_scheduler": False,
    "random_seed": 0,
    "directories": {
        "model_checkpoints": "./output/model_checkpoints",
        "training_images": "./output/training_images",
        "training_models": "./output/models",
        "video": "./output/video",
        "video_frames": "./output/video_frames",
        "training_data": "training_data/",
    },
}


def load_settings(path=SETTINGS_FILE):
    """Load settings.json, filling any missing key with its default."""
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
    except FileNotFoundError:
        print(f"{path} not found, using defaults.")
        return settings
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")
    directories = user.pop("directories", {})
    settings.update(user)
    settings["directories"].update(directories)
    return settings


def resolve_dir(path):
    """Resolve a directory from settings relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def list_images(directory):
    """Recursively list image files below `directory`, sorted for determinism."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
