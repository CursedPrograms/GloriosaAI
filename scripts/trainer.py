"""Train a DCGAN on the images in the training-data folder.

One "epoch" here is one optimisation step on one batch (as in earlier versions).
Defaults come from settings.json; command-line flags override them, and
--interactive prompts for each value instead.
"""
import argparse
import csv
import json
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from common import IMAGE_SHAPE, list_images, load_settings, resolve_dir
from models import build_discriminator, build_generator
from video_encoder import encode_video

GRID = 4  # samples are saved as a GRID x GRID contact sheet from fixed noise
REAL_LABEL = 0.9  # one-sided label smoothing


def parse_args(settings):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=settings["epochs"])
    p.add_argument("--batch-size", type=int, default=settings["batch_size"])
    p.add_argument("--latent-dim", type=int, default=settings["latent_dim"])
    p.add_argument("--generation-interval", type=int, default=settings["generation_interval"],
                   help="save a sample grid every N epochs")
    p.add_argument("--checkpoint-interval", type=int, default=settings["checkpoint_interval"],
                   help="save a checkpoint and model backup every N epochs")
    p.add_argument("--learning-rate", type=float, default=settings["learning_rate"])
    p.add_argument("--lr-scheduler", action="store_true", default=settings["use_learning_rate_scheduler"],
                   help="exponentially decay the learning rate")
    p.add_argument("--seed", type=int, default=settings["random_seed"])
    p.add_argument("--fresh", action="store_true", help="ignore existing checkpoints and start from scratch")
    p.add_argument("--video", choices=["yes", "no", "ask"], default="ask",
                   help="encode a video of the samples when training ends")
    p.add_argument("--interactive", action="store_true", help="prompt for each setting")
    return p.parse_args()


def ask(prompt, default, cast):
    while True:
        raw = input(f"{prompt} (default: {default}): ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"Please enter a valid {cast.__name__}.")


def prompt_for_settings(args):
    args.epochs = ask("Number of epochs", args.epochs, int)
    args.batch_size = ask("Batch size", args.batch_size, int)
    print("Note: keep the latent dimension the same when resuming from a checkpoint.")
    args.latent_dim = ask("Latent dimension", args.latent_dim, int)
    args.generation_interval = ask("Sample interval (epochs)", args.generation_interval, int)
    args.learning_rate = ask("Learning rate", args.learning_rate, float)
    args.lr_scheduler = input("Use learning rate scheduler (y/N)? ").strip().lower() == "y"
    args.seed = ask("Random seed", args.seed, int)


def validate(args):
    for name in ("epochs", "batch_size", "latent_dim", "generation_interval", "checkpoint_interval"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus if gpus else 'none (training on CPU)'}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_image(path):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SHAPE[:2])
    return tf.cast(image, tf.float32) / 127.5 - 1.0  # [-1, 1] to match the generator's tanh


def make_dataset(files, batch_size, seed):
    ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files])
    ds = ds.shuffle(len(files), seed=seed).repeat()
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def to_uint8(images):
    return np.clip((np.asarray(images) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_grid(images, path):
    images = to_uint8(images)
    rows = [np.concatenate(list(images[r * GRID:(r + 1) * GRID]), axis=1) for r in range(GRID)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def save_checkpoint(generator, discriminator, epoch, latent_dim, checkpoint_dir):
    generator.save_weights(str(checkpoint_dir / "generator.weights.h5"))
    discriminator.save_weights(str(checkpoint_dir / "discriminator.weights.h5"))
    (checkpoint_dir / "state.json").write_text(json.dumps({"epoch": epoch, "latent_dim": latent_dim}))


def save_models(generator, discriminator, epoch, model_dir):
    generator.save(str(model_dir / f"generator_{epoch}.keras"))
    discriminator.save(str(model_dir / f"discriminator_{epoch}.keras"))
    print(f"Models saved to {model_dir}")


def try_resume(generator, discriminator, latent_dim, checkpoint_dir):
    """Load a checkpoint if one exists and matches; return the epoch to continue from."""
    state_file = checkpoint_dir / "state.json"
    gen_file = checkpoint_dir / "generator.weights.h5"
    disc_file = checkpoint_dir / "discriminator.weights.h5"
    if not (state_file.exists() and gen_file.exists() and disc_file.exists()):
        return 0
    state = json.loads(state_file.read_text())
    if state.get("latent_dim") != latent_dim:
        raise SystemExit(
            f"Checkpoint was trained with latent_dim={state.get('latent_dim')}, not {latent_dim}. "
            "Use the same value or pass --fresh."
        )
    generator.load_weights(str(gen_file))
    discriminator.load_weights(str(disc_file))
    print(f"Resumed from checkpoint at epoch {state['epoch']}.")
    return int(state["epoch"])


def main():
    settings = load_settings()
    args = parse_args(settings)
    if args.interactive:
        prompt_for_settings(args)
    validate(args)

    enable_gpu_memory_growth()
    tf.keras.utils.set_random_seed(args.seed)

    dirs = settings["directories"]
    data_dir = resolve_dir(dirs["training_data"])
    files = list_images(data_dir)
    if not files:
        raise SystemExit(f"No training images found in {data_dir}. Add images there or run the "
                         "image processor first (see README).")
    print(f"Found {len(files)} training images.")

    run_id = int(time.time())
    sample_dir = resolve_dir(dirs["training_images"]) / f"output_image_{run_id}"
    model_dir = resolve_dir(dirs["training_models"]) / f"output_model_{run_id}"
    checkpoint_dir = resolve_dir(dirs["model_checkpoints"])
    for d in (sample_dir, model_dir, checkpoint_dir):
        d.mkdir(parents=True, exist_ok=True)

    generator = build_generator(args.latent_dim)
    discriminator = build_discriminator(IMAGE_SHAPE)
    start_epoch = 0 if args.fresh else try_resume(generator, discriminator, args.latent_dim, checkpoint_dir)

    lr = args.learning_rate
    if args.lr_scheduler:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=1000, decay_rate=0.9)
    g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(real):
        noise = tf.random.normal((tf.shape(real)[0], args.latent_dim))
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake = generator(noise, training=True)
            real_logits = discriminator(real, training=True)
            fake_logits = discriminator(fake, training=True)
            d_loss = (bce(tf.ones_like(real_logits) * REAL_LABEL, real_logits)
                      + bce(tf.zeros_like(fake_logits), fake_logits))
            g_loss = bce(tf.ones_like(fake_logits), fake_logits)
        g_opt.apply_gradients(zip(g_tape.gradient(g_loss, generator.trainable_variables),
                                  generator.trainable_variables))
        d_opt.apply_gradients(zip(d_tape.gradient(d_loss, discriminator.trainable_variables),
                                  discriminator.trainable_variables))
        return d_loss, g_loss

    fixed_noise = np.random.default_rng(args.seed).normal(size=(GRID * GRID, args.latent_dim)).astype("float32")
    dataset = iter(make_dataset(files, args.batch_size, args.seed))

    log_path = sample_dir / "losses.csv"
    epoch = start_epoch
    d_sum = g_sum = 0.0
    count = 0
    started = time.time()
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "d_loss", "g_loss"])
        try:
            while epoch < args.epochs:
                d_loss, g_loss = train_step(next(dataset))
                d_sum += float(d_loss)
                g_sum += float(g_loss)
                count += 1
                epoch += 1

                if epoch % args.generation_interval == 0:
                    d_avg, g_avg = d_sum / count, g_sum / count
                    save_grid(generator(fixed_noise, training=False), sample_dir / f"sample_{epoch:07d}.png")
                    log.writerow([epoch, f"{d_avg:.4f}", f"{g_avg:.4f}"])
                    log_file.flush()
                    print(f"Epoch {epoch}/{args.epochs}  D loss: {d_avg:.4f}  G loss: {g_avg:.4f}  "
                          f"({time.time() - started:.0f}s)")
                    d_sum = g_sum = 0.0
                    count = 0

                if epoch % args.checkpoint_interval == 0:
                    save_checkpoint(generator, discriminator, epoch, args.latent_dim, checkpoint_dir)
                    save_models(generator, discriminator, epoch, model_dir)
        except KeyboardInterrupt:
            print("\nInterrupted, saving progress...")

    if epoch > start_epoch and epoch % args.checkpoint_interval:
        save_checkpoint(generator, discriminator, epoch, args.latent_dim, checkpoint_dir)
        save_models(generator, discriminator, epoch, model_dir)
    print(f"Finished at epoch {epoch}. Samples: {sample_dir}")

    make_video = args.video
    if make_video == "ask":
        make_video = "yes" if input("Create a video of the samples (y/N)? ").strip().lower() in ("y", "yes") else "no"
    if make_video == "yes":
        frames_dir = resolve_dir(dirs["video_frames"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.png"):
            old.unlink()
        for frame in sorted(sample_dir.glob("*.png")):
            shutil.copy(frame, frames_dir / frame.name)
        encode_video(frames_dir, resolve_dir(dirs["video"]))


if __name__ == "__main__":
    sys.exit(main())
