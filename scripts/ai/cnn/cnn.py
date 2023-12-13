import tensorflow as tf
from tensorflow.keras import layers, models
def build_generator():
    model = models.Sequential()
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model
generator = build_generator()
generator.summary()
