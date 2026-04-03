[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<br>
<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>
<br>

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
</div>

<div align="center">
  <img alt="TensorFlow" src="https://img.shields.io/badge/tensorflow%20-%23323330.svg?&style=for-the-badge&logo=tensorflow&logoColor=white"/>
   <img alt="OpenCV" src="https://img.shields.io/badge/opencv-%23323330.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
</div>
<div align="center">
    <img alt="Git" src="https://img.shields.io/badge/git%20-%23323330.svg?&style=for-the-badge&logo=git&logoColor=white"/>
  <img alt="PowerShell" src="https://img.shields.io/badge/PowerShell-%23323330.svg?&style=for-the-badge&logo=powershell&logoColor=white"/>
  <img alt="Shell" src="https://img.shields.io/badge/Shell-%23323330.svg?&style=for-the-badge&logo=gnu-bash&logoColor=white"/>
  <img alt="Batch" src="https://img.shields.io/badge/Batch-%23323330.svg?&style=for-the-badge&logo=windows&logoColor=white"/>
  </div>
<br>

## GloriosaAI

<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="GloriosaAI" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/demo_images/gloriosa_cover.png">
</a>
</div>
<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>

## Scripts:

- **main.py:** The selection menu for GloriosaAI


### /scripts/

- **trainer.py:** Runs GloriosaAI trainer
- **modelout.py:** Output images from trained models with GloriosaAI
- **video_encoder.py:** Encode a video using GloriosaAI
- **image-processor.py:** Prepare images for GloriosaAI
- **preprocessor_data.py:** Dependency for GloriosaAI
- **install_dependencies.py:** Install dependencies

### Prerequisite Folders:

### Trainer
- `training_data/class`

### Video Encoder
- `output/video_frames` (Also created by the Trainer)

### ModelOut
- `input/input_models` (Copy the architecture and weights of both the discriminator and generator.)

### Image-Processor
- `unprocessed_images`

<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>

### trainer.py Hyperparameters:

- **Epochs**:
  - Controls the number of training iterations.

- **Batch Size**:
  - Determines the number of data samples processed in each training step.

- **Latent Dimension**:
  - Defines the size of the latent space in the generative model.

- **Generation Interval**:
  - Sets how often generated images are saved during training.

- **Learning Rate**:
  - Governs the step size during gradient descent optimization.

- **Use Learning Rate Scheduler**:
  - Specifies whether to use a learning rate scheduler during training.

- **Random Seed**:
  - Seeds the random number generator for reproducibility.
</p>
<br>

<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>

<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank" align="center">
  <img alt="GloriosaAI" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/demo_images/gloriosa.gif">
</a>
</div>
<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>

### Prerequisites:

Package            Version
------------------ -----------
absl-py            2.4.0
astunparse         1.6.3
certifi            2026.2.25
charset-normalizer 3.4.7
click              8.3.1
contourpy          1.3.3
cycler             0.12.1
Flask              2.1.1
flatbuffers        25.12.19
fonttools          4.62.1
gast               0.7.0
google-pasta       0.2.0
grpcio             1.80.0
h5py               3.14.0
idna               3.11
itsdangerous       2.2.0
Jinja2             3.1.6
keras              3.14.0
kiwisolver         1.5.0
libclang           18.1.1
markdown-it-py     4.0.0
MarkupSafe         3.0.3
matplotlib         3.8.2
mdurl              0.1.2
ml_dtypes          0.5.4
namex              0.1.0
numpy              1.26.2
opencv-python      4.8.1.78
opt_einsum         3.4.0
optree             0.19.0
packaging          26.0
Pillow             10.1.0
pip                26.0.1
protobuf           7.34.1
Pygments           2.20.0
pyparsing          3.3.2
python-dateutil    2.9.0.post0
requests           2.33.1
rich               14.3.3
setuptools         82.0.1
six                1.17.0
tensorflow         2.21.0
termcolor          3.3.0
typing_extensions  4.15.0
urllib3            2.6.3
Werkzeug           3.1.8
wheel              0.46.3
wrapt              2.1.2


### Compiler:

- PyInstaller

### Optional:

- A dataset of images for training (128 x 128 resolution, RGB format)

[GloriosaAI - Art Showcase](https://www.youtube.com/watch?v=0XxlTf5EoUs)

<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>

## How to Run:
```bash
pip install -r requirements.txt
```
```bash
pip install opencv-contrib-python
```
For GPU
```bash
nvidia-smi
pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
pip install -U tensorflow[and-cuda]
```
For CPU
```bash
pip install tensorflow
```
Additional Packages
```bash
pip install numpy
pip install matplotlib
pip install Pillow
pip install opencv-python
pip install pyinstaller
pip install flask
```
```bash
python main.py
```

To make the setup script executable, run the following command in your terminal:

```bash
chmod +x setup.sh
```
<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="Gloriosa Icon" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/icons/icon.ico">
</a>
</div>
<br>
<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="GloriosaAI" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/demo_images/gloriosa.jpg">
</a>
</div>

- [Gender-Age-ID Repository](https://github.com/CursedPrograms/Gender-Age-ID)
- [Detect-Face Repository](https://github.com/CursedPrograms/Detect-Face)
- [Cursed GPT](https://github.com/CursedPrograms/Cursed-GPT)
- [Image-Generator](https://github.com/CursedPrograms/Image-Generator)

<br>
<div align="center">
Cursed Entertainment 2024
</div>
<br>
<div align="center">
<a href="https://cursed-entertainment.itch.io/" target="_blank">
    <img src="https://github.com/CursedPrograms/cursedentertainment/raw/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>



