import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam


def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

def build_generator(noise_dim, img_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

def train_gan(generator, discriminator, gan, epochs, batch_size, noise_dim, img_shape, camo_data):
    for epoch in range(epochs):
        for _ in range(camo_data.shape[0] // batch_size):
            noise = generate_noise(batch_size, noise_dim)
            generated_images = generator.predict(noise)
            image_batch = camo_data[np.random.randint(0, camo_data.shape[0], size=batch_size)]

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(image_batch, valid)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = generate_noise(batch_size, noise_dim)
            valid = np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(noise, valid)

        print(f"Epoch {epoch+1}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

def generate_camo(generator, noise_dim, num_samples):
    noise = generate_noise(num_samples, noise_dim)
    generated_images = generator.predict(noise)
    return generated_images

noise_dim = 100
img_shape = (64, 64, 3) 
batch_size = 64
epochs = 10000
camo_data = np.random.rand(1000, *img_shape)  

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

generator = build_generator(noise_dim, img_shape)

gan = build_gan(generator, discriminator)

train_gan(generator, discriminator, gan, epochs, batch_size, noise_dim, img_shape, camo_data)

generated_camo = generate_camo(generator, noise_dim, num_samples=5)

for i in range(generated_camo.shape[0]):
    plt.imshow(generated_camo[i])
    plt.title(f"Generated Camouflage {i+1}")
    plt.show()
