"""Generate images from trained generators found in the input-models folder.

Supports the .keras files written by the trainer and the older
generator_architecture_N.json + generator_weights_N.h5 pairs.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import ROOT


def load_model(path, compile=False):
    from tensorflow.keras.models import load_model as keras_load
    return keras_load(path, compile=compile)


def find_generators(model_dir):
    """Return [(label, loader)] for every generator in `model_dir`."""
    model_dir = Path(model_dir)
    found = []
    for path in sorted(model_dir.glob("generator_*.keras")):
        found.append((path.stem, lambda p=path: load_model(str(p), compile=False)))
    for arch in sorted(model_dir.glob("generator_architecture_*.json")):
        epoch = re.fullmatch(r"generator_architecture_(\d+)", arch.stem)
        weights = model_dir / f"generator_weights_{epoch.group(1)}.h5" if epoch else None
        if weights and weights.exists():
            found.append((f"generator_{epoch.group(1)}", lambda a=arch, w=weights: load_legacy(a, w)))
        else:
            print(f"Skipping {arch.name}: matching weights file not found.")
    return found


def load_legacy(architecture, weights):
    """Load an old architecture JSON + .h5 pair.

    JSON saved by Keras 2 often can't be deserialised by Keras 3, so on failure
    fall back to rebuilding the original generator and loading the weights into it.
    """
    from tensorflow.keras.models import model_from_json
    text = architecture.read_text()
    try:
        model = model_from_json(text)
    except Exception:
        from models import build_legacy_generator
        shape = re.search(r'"batch_input_shape":\s*\[\s*null,\s*(\d+)\s*\]', text)
        model = build_legacy_generator(int(shape.group(1)) if shape else 128)
    model.load_weights(str(weights))
    return model


def to_uint8(images, model):
    """Map model output to 0-255; tanh generators emit [-1, 1], sigmoid (legacy) ones [0, 1]."""
    activation = getattr(model.layers[-1], "activation", None)
    if getattr(activation, "__name__", "") == "tanh":
        images = (images + 1.0) / 2.0
    return np.clip(images * 255, 0, 255).astype(np.uint8)


def save_grid(images, path, columns=4):
    rows = [np.concatenate(list(images[i:i + columns]), axis=1)
            for i in range(0, len(images) - columns + 1, columns)]
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(ROOT / "input" / "input_models"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "output_model_images"))
    parser.add_argument("--count", type=int, default=16, help="images per model")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    generators = find_generators(args.model_dir)
    if not generators:
        print(f"No generator models found in {args.model_dir}.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for label, load in generators:
        try:
            model = load()
        except Exception as e:  # a bad file shouldn't stop the other models
            print(f"Could not load {label}: {str(e)[:200]}")
            continue
        noise = rng.normal(size=(args.count, *model.input_shape[1:])).astype("float32")
        images = to_uint8(model.predict(noise, verbose=0), model)
        for i, img in enumerate(images):
            Image.fromarray(img).save(out_dir / f"{label}_sample_{i}.png")
        save_grid(images, out_dir / f"{label}_grid.png")
        print(f"{label}: wrote {len(images)} images")
    print(f"Images saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
