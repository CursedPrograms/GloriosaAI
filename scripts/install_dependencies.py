"""Install requirements.txt into the Python environment running this script."""
import subprocess
import sys

from common import ROOT


def main():
    requirements = ROOT / "requirements.txt"
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    if result.returncode == 0:
        print("Dependencies installed successfully.")
    else:
        print(f"pip failed with exit code {result.returncode}.")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
