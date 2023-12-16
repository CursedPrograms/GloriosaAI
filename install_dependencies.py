import subprocess

dependencies = [
    "tensorflow",
    "numpy",
    "matplotlib",
    "Pillow",
    "opencv-python",
    "pyinstaller",       
]

for package in dependencies:
    subprocess.call(["pip", "install", package])

if __name__ == "__main__":
    subprocess.run(["python", "main.py"])