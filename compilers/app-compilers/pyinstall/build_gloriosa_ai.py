import subprocess

pyinstaller_command = [
    "python",
    "-m",
    "PyInstaller",
    "--name",
    "run_gloriosa_ai",
    "--onefile",
    "run_gloriosa_ai.py"
]

try:
    subprocess.run(pyinstaller_command, check=True)
    print("PyInstaller completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error running PyInstaller: {e}")