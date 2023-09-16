import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import shutil         
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

epochs = int(input("Enter the number of epochs (recommended: 10,000): "))
batch_size = int(input("Enter the batch size (recommended: 1): "))
latent_dim = int(input("Enter the latent dimension (recommended: 100): "))
generation_interval = int(input("Enter the generation interval (recommended: 100) (e.g., every X epochs): "))

output_base_dir = "output_images"
model_save_base_dir = "output_models"
unique_identifier = int(time.time())
output_dir = os.path.join(output_base_dir, f"output_image_{unique_identifier}")
model_save_dir = os.path.join(model_save_base_dir, f"output_model_{unique_identifier}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

image_shape = (128, 128, 3)

def build_generator(latent_dim):
    generator = Sequential()
    generator.add(Dense(8 * 8 * 1024, input_dim=latent_dim))
    generator.add(Reshape((8, 8, 1024)))
    generator.add(Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))
    return generator

def build_discriminator(input_shape):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape, activation='relu'))
    discriminator.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator

def load_and_preprocess_dataset(target_size, batch_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "training_data")

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )
    dataset = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='input'
    )
    return dataset

dataset = load_and_preprocess_dataset(image_shape[:2], batch_size)
generator = build_generator(latent_dim)
discriminator = build_discriminator(image_shape)

discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)

gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

def generate_and_save_images(generator, epoch, output_dir, num_examples=1):
    if epoch > 0 and epoch % 100 == 0:
        noise = np.random.normal(0, 1, (num_examples, latent_dim))
        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5

        for i in range(num_examples):
            plt.figure(figsize=(1, 1))
            plt.imshow(generated_images[i])
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"generated_image_epoch_{epoch}_sample_{i}.png"), bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()

for epoch in range(epochs):
    real_batch = next(dataset)
    real_images = real_batch[0]

    noise = np.random.normal(0, 1, (real_images.shape[0], latent_dim))
    generated_images = generator.predict(noise)

    generated_images = [tf.image.resize(image, (128, 128)) for image in generated_images]
    generated_images = np.array(generated_images)

    real_labels = np.ones((real_images.shape[0], 1))
    fake_labels = np.zeros((generated_images.shape[0], 1))

    merged_images = np.concatenate([real_images, generated_images], axis=0)
    merged_labels = np.concatenate([real_labels, fake_labels], axis=0)

    indices = np.arange(merged_images.shape[0])
    np.random.shuffle(indices)
    merged_images = merged_images[indices]
    merged_labels = merged_labels[indices]

    d_loss = discriminator.train_on_batch(merged_images, merged_labels)

    noise = np.random.normal(0, 1, (real_images.shape[0], latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % generation_interval == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]} | G Loss: {g_loss}")

        for i in range(len(generated_images)):
            generated_image = generated_images[i]
            generated_image = (generated_image * 255).astype(np.uint8)
            generated_image = Image.fromarray(generated_image)
            generated_image.save(os.path.join(output_dir, f"generated_image_epoch_{epoch}_sample_{i}.png"))

    if epoch % 500 == 0:
        generator_model_save_path = os.path.join(model_save_dir, f"gan_generator_weights_epoch_{epoch}.h5")
        generator_architecture_path = generator_model_save_path.replace(".h5", "_architecture.json")

        with open(generator_architecture_path, "w") as json_file:
            json_file.write(generator.to_json())

        discriminator_model_save_path = os.path.join(model_save_dir, f"gan_discriminator_weights_epoch_{epoch}.h5")
        discriminator_architecture_path = discriminator_model_save_path.replace(".h5", "_architecture.json")

        with open(discriminator_architecture_path, "w") as json_file:
            json_file.write(discriminator.to_json())

        generator.save_weights(generator_model_save_path)
        discriminator.save_weights(discriminator_model_save_path)

        print(f"Saved generator model weights to {generator_model_save_path}")
        print(f"Saved discriminator model weights to {discriminator_model_save_path}")

    if epoch == epochs - 1:
        user_input = input("Training is complete. Do you want to create a video (yes/no)? ").strip().lower()

        if user_input == "yes":
            video_frames_dir = os.path.join(os.getcwd(), "video_frames")
            os.makedirs(video_frames_dir, exist_ok=True)

            for existing_file in os.listdir(video_frames_dir):
                if existing_file.endswith(".png"):
                    existing_path = os.path.join(video_frames_dir, existing_file)
                    os.remove(existing_path)

            for image_file in os.listdir(output_dir):
                if image_file.endswith(".png"):
                    src_path = os.path.join(output_dir, image_file)
                    dst_path = os.path.join(video_frames_dir, image_file)
                    shutil.move(src_path, dst_path)

            print("Files moved to 'video_frames' folder.")

            os.system("python dyslexai-video_encoder.py")

        print(f"Exiting the program at epoch {epoch}.")