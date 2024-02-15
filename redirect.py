import subprocess
import os

if __name__ == "__main__":
    script_path = os.path.join("scripts", "transformer.py")
    subprocess.run(["python", script_path])