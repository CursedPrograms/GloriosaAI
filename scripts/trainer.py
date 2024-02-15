import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import subprocess
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from preprocess_data import image_paths, labels 
import json

# Check available devices
print(tf.config.experimental.list_physical_devices('GPU'))

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with open("settings.json", "r") as settings_file:
    settings = json.load(settings_file)

data_directory = settings["directories"]["training_data"]
output_base_dir = settings["directories"]["training_images"]
model_save_base_dir = settings["directories"]["training_models"]
checkpoint_dir = settings["directories"]["model_checkpoints"]
unique_identifier = int(time.time())
output_dir = os.path.join(output_base_dir, f"output_image_{unique_identifier}")
model_save_dir = os.path.join(model_save_base_dir, f"output_model_{unique_identifier}")

for directory in [output_dir, model_save_dir, checkpoint_dir]:
    os.makedirs(directory, exist_ok=True)

# Hyperparameters
def get_integer_input(prompt, recommended):
    while True:
        try:
            value = int(input(prompt))
            if value < 0: 
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a valid integer. (Recommended: {recommended})")

def get_float_input(prompt, recommended):
    while True:
        try:
            value = float(input(prompt))
            if value < 0.0: 
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a valid float. (Recommended: {recommended})")

def get_integer_input(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return int(user_input) if user_input else default

# Function to get float input with a default value
def get_float_input(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return float(user_input) if user_input else default

# Function to load settings from a JSON file
def load_settings_from_json(filename):
    try:
        with open(filename) as json_file:
            settings = json.load(json_file)
        return settings
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Get user choice
choice = input("Do you want to enter settings interactively (I) or load from settings.json file (F)? ").strip().lower()

if choice == "i":
    # Interactive input
    epochs = get_integer_input("Enter the number of epochs (recommended: 10,000)", 10000)
    batch_size = get_integer_input("Enter the batch size (recommended: 1)", 1)
    print("Note: It is essential to keep the latent dimension consistent throughout training.")
    latent_dim = get_integer_input("Enter the latent dimension (recommended: 128)", 128)
    print(f"Latent dimension set to: {latent_dim}")
    generation_interval = get_integer_input("Enter the image generation interval (recommended: 25) (e.g., every X epochs)", 25)
    learning_rate = get_float_input("Enter the learning rate (recommended: 0.0002)", 0.0002)
    use_learning_rate_scheduler = input("Use learning rate scheduler (y/n)? ").strip().lower() == "y"
    random_seed = get_integer_input("Enter the random seed", 0)
elif choice == "f":
    # Load from settings.json file
    settings = load_settings_from_json("settings.json")
    if settings:
        epochs = settings.get("epochs", 10000)
        batch_size = settings.get("batch_size", 1)
        latent_dim = settings.get("latent_dim", 128)
        generation_interval = settings.get("generation_interval", 25)
        learning_rate = settings.get("learning_rate", 0.0002)
        use_learning_rate_scheduler = settings.get("use_learning_rate_scheduler", False)
        random_seed = settings.get("random_seed", 0)
    else:
        print("Error loading settings from settings.json. Using default values.")
else:
    print("Invalid choice. Exiting.")


# Image shape
image_shape = (128, 128, 3)
initial_epoch = 0

# Function to copy checkpoints
def copy_checkpoint(original_path, new_name, model_name):
    checkpoint_path = f"model_checkpoints/gan_{model_name}_weights_epoch_0.h5"
    new_checkpoint_path = f"model_checkpoints/gan_{model_name}_weights_epoch_1.h5"

    if os.path.exists(new_checkpoint_path):
        os.remove(new_checkpoint_path)

    if os.path.exists(checkpoint_path):
        os.rename(checkpoint_path, new_checkpoint_path)
        print(f"Renamed {checkpoint_path} to {new_checkpoint_path}")
    else:
        print(f"{model_name} checkpoint file not found at: {checkpoint_path}.")

# Copy checkpoints
copy_checkpoint("generator", "gan_generator", "generator")
copy_checkpoint("discriminator", "gan_discriminator", "discriminator")

# Define the learning rate scheduler
if use_learning_rate_scheduler:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )
else:
    lr_schedule = learning_rate

# Random seeds
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Build generator
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

# Build discriminator
def build_discriminator(input_shape):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape, activation='relu'))
    discriminator.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator

# Initialize models
generator = build_generator(latent_dim)
discriminator = build_discriminator(image_shape)

# Paths for initial weights
generator_weights_path = os.path.join(checkpoint_dir, f"gan_generator_weights_epoch_1.h5")
discriminator_weights_path = os.path.join(checkpoint_dir, f"gan_discriminator_weights_epoch_1.h5")

# Load initial weights if they exist
if not os.path.exists(generator_weights_path) or not os.path.exists(discriminator_weights_path):
    print(f"Weights file not found at: {checkpoint_dir}. Training from scratch.")
else:
    generator.load_weights(generator_weights_path)
    discriminator.load_weights(discriminator_weights_path)
    print("Generator weights loaded successfully.")
    print("Discriminator weights loaded successfully")

    generator_last_epoch = int(generator_weights_path.split("_")[-1].split(".")[0])
    discriminator_last_epoch = int(discriminator_weights_path.split("_")[-1].split(".")[0])

    initial_epoch_path = os.path.join(checkpoint_dir, "current_epoch.txt")
    if os.path.exists(initial_epoch_path):
        with open(initial_epoch_path, "r") as epoch_file:
            epoch_info = epoch_file.read().strip()
            if epoch_info.startswith("initial_epoch: "):
                initial_epoch = int(epoch_info[len("initial_epoch: "):])
            else:
                print("Invalid format in epoch file.")
    else:
        initial_epoch = 0
        print("Epoch file not found. Using the maximum epoch from weights files as the initial epoch.")

# Custom accuracy metric
def custom_accuracy(y_true, y_pred):
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_pred_binary, y_true), tf.float32))
    total_samples = tf.cast(tf.shape(y_true)[0], tf.float32)
    accuracy = correct_predictions / total_samples
    return accuracy

def load_and_preprocess_dataset(target_size, batch_size):
    # Assuming your JSON is stored in a file named 'config.json'
    with open("settings.json", "r") as settings_file:
        settings = json.load(settings_file)

    # Extract the training data directory from the loaded JSON
    training_data_dir = settings["directories"]["training_data"]

    # Construct the full path to the training data directory by going back one folder
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_folder, '..', training_data_dir)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    dataset = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return dataset

# Function to generate and save images
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

# Load and preprocess dataset
dataset = load_and_preprocess_dataset(image_shape[:2], batch_size)

# Build discriminator and GAN
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, 0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)

gan_optimizer = tf.keras.optimizers.Adam(learning_rate)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy', metrics=['accuracy', custom_accuracy])

# Determine initial epoch
initial_epoch_path = os.path.join(checkpoint_dir, "current_epoch.txt")
if os.path.exists(initial_epoch_path):
    with open(initial_epoch_path, "r") as epoch_file:
        epoch_info = epoch_file.read().strip()
        if epoch_info.startswith("initial_epoch: "):
            initial_epoch = int(epoch_info[len("initial_epoch: "):])
        else:
            print("Invalid format in epoch file.")
else:
    initial_epoch = 0

print(f"Initial epoch set to: {initial_epoch}")

# Function to save backup models
def save_models(epoch, generator, discriminator, model_save_dir, initial_epoch):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    generator_weights_path = os.path.join(model_save_dir, f"generator_weights_{epoch}.h5")
    discriminator_weights_path = os.path.join(model_save_dir, f"discriminator_weights_{epoch}.h5")
    generator.save_weights(generator_weights_path)
    discriminator.save_weights(discriminator_weights_path)

    generator_architecture_path = os.path.join(model_save_dir, f"generator_architecture_{epoch}.json")
    with open(generator_architecture_path, "w") as json_file:
        json_file.write(generator.to_json())

    discriminator_architecture_path = os.path.join(model_save_dir, f"discriminator_architecture_{epoch}.json")
    with open(discriminator_architecture_path, "w") as json_file:
        json_file.write(discriminator.to_json())
    print(f"Models saved to {model_save_dir}")


# Function to save checkpoint models and trigger callbacks
def save_and_trigger_callbacks(generator, discriminator, epoch, checkpoint_dir, initial_epoch):
    generator_weights_path = os.path.join(checkpoint_dir, f"gan_generator_weights_epoch_0.h5")
    discriminator_weights_path = os.path.join(checkpoint_dir, f"gan_discriminator_weights_epoch_0.h5")

    generator.save_weights(generator_weights_path)
    discriminator.save_weights(discriminator_weights_path)

    with open(os.path.join(checkpoint_dir, "current_epoch.txt"), "w") as epoch_file:
        epoch_file.write(f"initial_epoch: {epoch}")

    print(f"Models saved to {checkpoint_dir}")

# Training loop
for epoch in range(epochs):
    real_batch = next(dataset)
    real_images = real_batch[0]
    noise = np.random.normal(0, 1, (real_images.shape[0], latent_dim))
    generated_images = generator.predict(noise)

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
        for i in range(len(generated_images)):
            generated_image = generated_images[i]
            generated_image = (generated_image * 255).astype(np.uint8)
            generated_image = Image.fromarray(generated_image)
            generated_image.save(os.path.join(output_dir, f"generated_image_epoch_{epoch}_sample_{i}.png"))
        print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")
        print(f"Image(s) saved to {output_dir}")

    if epoch > 0 and epoch % 500 == 0:
        
        save_and_trigger_callbacks(generator, discriminator, epoch, checkpoint_dir, initial_epoch)
        save_models(epoch, generator, discriminator, model_save_dir, initial_epoch)

user_input = input("Training is complete. Do you want to create a video (yes/no)? ").strip().lower()

if user_input == "yes":
    video_frames_dir = settings["directories"]["video_frames"]
    os.makedirs(video_frames_dir, exist_ok=True)

    for existing_file in os.listdir(video_frames_dir):
        if existing_file.endswith(".png"):
            existing_path = os.path.join(video_frames_dir, existing_file)
            os.remove(existing_path)

    for image_file in os.listdir(output_dir):
        if image_file.endswith(".png"):
            src_path = os.path.join(output_dir, image_file)
            dst_path = os.path.join(video_frames_dir, image_file)
            shutil.copy(src_path, dst_path)

    print("Files copied to 'videoframes' folder.")

    os.system("python scripts/video_encoder.py")

save_and_trigger_callbacks(generator, discriminator, epoch, checkpoint_dir, initial_epoch)
save_models(epoch, generator, discriminator, model_save_dir, initial_epoch)

print(f"Initial epoch set to: {initial_epoch}")
print(f"Exiting the program at epoch {epoch}.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main_script_path = os.path.join(script_dir, "..", "main.py")
    
    subprocess.run(["python", main_script_path])
