"""GAN architectures. Both networks work on images scaled to [-1, 1]."""
from tensorflow.keras import Model, layers

from common import IMAGE_SHAPE


def build_generator(latent_dim):
    z = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 512, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 512))(x)
    for filters in (256, 128, 64, 32):  # 8 -> 16 -> 32 -> 64 -> 128
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    return Model(z, out, name="generator")


def build_discriminator(input_shape=IMAGE_SHAPE):
    img = layers.Input(shape=input_shape)
    x = img
    for filters in (64, 128, 256, 512):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)  # raw logits; the loss uses from_logits=True
    return Model(img, out, name="discriminator")


def build_legacy_generator(latent_dim=128):
    """The pre-2.0 architecture (ReLU, sigmoid output), for loading old .h5 weights."""
    z = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 1024)(z)
    x = layers.Reshape((8, 8, 1024))(x)
    for filters in (512, 256, 128, 64):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return Model(z, out, name="legacy_generator")
