"""Encode the PNG frames in the video-frames folder into an mp4."""
import argparse
import sys
from pathlib import Path

from common import load_settings, resolve_dir


def encode_video(frames_dir, output_dir, fps=30.0, keep=5, base_name="output_video"):
    """Write frames (sorted by name) to the next free `<base_name>_N.mp4`; return its path."""
    import cv2  # imported lazily so the module is cheap to import

    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        print(f"No PNG frames found in {frames_dir}.")
        return None

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"Could not read {frames[0]}.")
        return None
    height, width = first.shape[:2]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while (output_dir / f"{base_name}_{number}.mp4").exists():
        number += 1
    output_path = output_dir / f"{base_name}_{number}.mp4"

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open a video writer for {output_path}.")
        return None
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Skipping unreadable frame {frame_path.name}")
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    print(f"Video saved to {output_path} ({len(frames)} frames)")

    # Keep only the newest `keep` videos, ordered by their numeric suffix.
    def number_of(path):
        return int(path.stem.rsplit("_", 1)[1])

    videos = sorted(output_dir.glob(f"{base_name}_*.mp4"), key=number_of)
    if keep > 0:
        for old in videos[:-keep]:
            old.unlink()
    return output_path


def main(argv=None):
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", default=settings["directories"]["video_frames"])
    parser.add_argument("--output-dir", default=settings["directories"]["video"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--keep", type=int, default=5, help="videos to keep (0 = keep all)")
    args = parser.parse_args(argv)
    result = encode_video(resolve_dir(args.frames_dir), resolve_dir(args.output_dir), args.fps, args.keep)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
