import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "1": ("Trainer", "Train a Generative Adversarial Network (GAN)", ["scripts/trainer.py", "--interactive"]),
    "2": ("Video encoder", "Encode a video from saved sample images", ["scripts/video_encoder.py"]),
    "3": ("Model output", "Generate images from trained models", ["scripts/modelout.py"]),
    "4": ("Image processor", "Prepare images for training", ["scripts/image_processor.py"]),
    "00": ("Install dependencies", "Install the packages GloriosaAI needs", ["scripts/install_dependencies.py"]),
}


def app_name():
    try:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        return config.get("Config", {}).get("AppName", "GloriosaAI")
    except (OSError, json.JSONDecodeError):
        return "GloriosaAI"


def run_script(command):
    script = ROOT / command[0]
    if not script.exists():
        print(f"Script '{command[0]}' does not exist.")
        return
    try:
        subprocess.run([sys.executable, str(script), *command[1:]], cwd=ROOT)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    print(app_name())
    while True:
        print("\nAvailable scripts:")
        for key, (name, description, _) in SCRIPTS.items():
            print(f"  {key}: {name} - {description}")
        try:
            choice = input("Enter a number (or 'q' to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in ("q", "quit", "exit"):
            break
        if choice in SCRIPTS:
            run_script(SCRIPTS[choice][2])
        else:
            print("Invalid choice. Please select a valid script number.")


if __name__ == "__main__":
    main()
