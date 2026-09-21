[![Twitter: @NorowaretaGemu](https://img.shields.io/badge/X-@NorowaretaGemu-blue.svg?style=flat)](https://x.com/NorowaretaGemu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <a href="https://ko-fi.com/cursedentertainment">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi" style="width: 20%;"/>
  </a>
</div>

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%23323330.svg?&style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/tensorflow%20-%23323330.svg?&style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img alt="OpenCV" src="https://img.shields.io/badge/opencv-%23323330.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
</div>

# GloriosaAI

Train a generative adversarial network (GAN) on your own images, watch it learn, and turn the progress into a video.

<div align="center">
<a href="https://cursedprograms.github.io/gloriosa-ai-pr/" target="_blank">
  <img alt="GloriosaAI" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/demo_images/gloriosa_cover.png">
</a>
</div>

<div align="center">
  <img alt="GloriosaAI demo" src="https://github.com/CursedPrograms/GloriosaAI/raw/main/demo_images/gloriosa.gif">
</div>

[Art showcase on YouTube](https://www.youtube.com/watch?v=0XxlTf5EoUs)

## Quick start

Requires Python 3.9+ (3.10–3.12 recommended).

| Platform | Setup | Run |
| --- | --- | --- |
| Linux / macOS | `./setup.sh` | `./run.sh` |
| Windows (cmd) | `setup.bat` | `run.bat` |
| Windows (PowerShell) | `.\setup.ps1` | `.\run.ps1` |

Setup creates a `psdenv` virtual environment and installs `requirements.txt`. `run` does that automatically if needed, then opens the menu (`main.py`).
For GPU training see the [TensorFlow install guide](https://www.tensorflow.org/install/pip); exact versions from a known-good environment are in `requirements-lock.txt`.

## Workflow

1. **Prepare data.** Put raw images in `unprocessed_images/` and run *Image processor* (menu option 4). It writes randomly cropped, flipped, 128×128 RGB variants to `training_data/processed_images/`. Or place your own 128×128 images anywhere under `training_data/` (subfolders are fine, they are not treated as classes).
2. **Train.** Run *Trainer* (option 1). A grid of samples from fixed noise is saved every `generation_interval` epochs, so you can see the same latent points improve over time.
3. **Make a video.** The trainer offers to encode the sample grids at the end, or run *Video encoder* (option 2) on `output/video_frames/`.
4. **Generate.** Copy generator models into `input/input_models/` and run *Model output* (option 3).

Every script also works standalone with flags: `python scripts/trainer.py --help`.

### Training

```bash
python scripts/trainer.py                          # settings from settings.json
python scripts/trainer.py --epochs 20000 --batch-size 32 --learning-rate 0.0001
python scripts/trainer.py --interactive            # prompt for each value
python scripts/trainer.py --fresh                  # ignore the existing checkpoint
```

Training resumes automatically from `output/model_checkpoints/` (Ctrl+C saves a checkpoint first). A checkpoint can only be resumed with the same `latent_dim`.
Loss curves are written to `losses.csv` next to each run's samples.

| Setting (`settings.json`) | Flag | Meaning |
| --- | --- | --- |
| `epochs` | `--epochs` | Total optimisation steps. One epoch = one batch. |
| `batch_size` | `--batch-size` | Images per step. Use 8 or more; batch norm behaves badly at 1. |
| `latent_dim` | `--latent-dim` | Size of the generator's noise input. Keep it consistent across resumed runs. |
| `generation_interval` | `--generation-interval` | Save a sample grid every N epochs. |
| `checkpoint_interval` | `--checkpoint-interval` | Save a checkpoint and a `.keras` model backup every N epochs. |
| `learning_rate` | `--learning-rate` | Adam learning rate. |
| `use_learning_rate_scheduler` | `--lr-scheduler` | Exponential decay of the learning rate. |
| `random_seed` | `--seed` | Seed for reproducibility. |
| `directories` | — | Output and data locations, relative to the project root. |

### Output

```
output/
  training_images/output_image_<run>/   sample grids + losses.csv
  models/output_model_<run>/            generator_<epoch>.keras, discriminator_<epoch>.keras
  model_checkpoints/                    latest weights for resuming
  video_frames/  video/                 frames and encoded mp4s
  output_model_images/                  images from Model output
```

Model output reads both the new `.keras` generators and the older `generator_architecture_N.json` + `generator_weights_N.h5` pairs (see `input/input_models/`).

## Project layout

```
main.py                      interactive menu
settings.json                training defaults and directories
scripts/trainer.py           GAN training loop
scripts/models.py            generator and discriminator
scripts/modelout.py          sample images from saved generators
scripts/video_encoder.py     frames -> mp4
scripts/image_processor.py   dataset preparation and augmentation
scripts/common.py            settings and path helpers
tests/                       pytest suite (no TensorFlow needed)
```

## Related projects

- [Gender-Age-ID](https://github.com/CursedPrograms/Gender-Age-ID)
- [Detect-Face](https://github.com/CursedPrograms/Detect-Face)
- [Cursed GPT](https://github.com/CursedPrograms/Cursed-GPT)
- [Image-Generator](https://github.com/CursedPrograms/Image-Generator)

<div align="center">
Cursed Entertainment 2024
<br><br>
<a href="https://cursed-entertainment.itch.io/" target="_blank">
  <img src="https://github.com/CursedPrograms/cursedentertainment/raw/main/images/logos/logo-wide-grey.png"
       alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
