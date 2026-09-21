"""Turn raw images into augmented 128x128 RGB training images."""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image

from common import IMAGE_SIZE, ROOT, list_images


def process_image(image, size=IMAGE_SIZE, min_zoom=0.8, flip_vertical=False, rng=random):
    """Random square crop (keeping `min_zoom`..1 of the short side), random flips, resize."""
    image = image.convert("RGB")
    width, height = image.size
    side = int(min(width, height) * rng.uniform(min_zoom, 1.0))
    side = max(side, 1)
    left = rng.randint(0, width - side)
    top = rng.randint(0, height - side)
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_vertical and rng.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(ROOT / "unprocessed_images"))
    parser.add_argument("--output-dir", default=str(ROOT / "training_data" / "processed_images"))
    parser.add_argument("--variations", type=int, default=3, help="augmented copies per image")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--min-zoom", type=float, default=0.8)
    parser.add_argument("--vflip", action="store_true", help="also flip vertically at random")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    files = list_images(args.input_dir)
    if not files:
        print(f"No images found in {args.input_dir}.")
        return 1

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in files:
        try:
            with Image.open(path) as image:
                image.load()
                for variation in range(args.variations):
                    result = process_image(image, args.size, args.min_zoom, args.vflip, rng)
                    result.save(out_dir / f"processed_{path.stem}_{variation}.png")
                    written += 1
        except OSError as e:
            print(f"Skipping {path.name}: {e}")
    print(f"Wrote {written} images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
