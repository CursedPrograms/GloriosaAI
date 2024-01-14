<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
</div>

## GloriosaAI
![GloriosaAI](https://github.com/CursedPrograms/GloriosaAI/raw/main/Gloriosa.jpg)
### Prerequisite Folders:

### Trainer
- `training_data/class`

### Video Encoder
- `video_frames` (Also created by the Trainer)

### ModelOut
- `input_models` (Copy the architecture and weights of both the discriminator and generator.)

### Styles
- `image_style`
- `style_edit`

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

### Prerequisites:

- TensorFlow 2.14.0
- Numpy 1.26.2
- Matplotlib 3.8.2
- Pillow 10.1.0
- OpenCV-Python 4.8.1.78
- Flask 3.0.0

### Compiler:

- PyInstaller

### Optional:

- A dataset of images for training (128 x 128 resolution, RGB format)

[GloriosaAI - Art Showcase](https://www.youtube.com/watch?v=0XxlTf5EoUs)

### Installation:

You can install the required Python packages using pip:
```bash
pip install -r requirements.txt
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
```



