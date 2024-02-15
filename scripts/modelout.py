import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Sequential
from PIL import Image
import json

def get_integer_input(prompt, recommended):
    while True:
        try:
            value = int(input(prompt))
            if value < 0: 
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a valid integer. (Recommended: {recommended})")

latent_dim = get_integer_input("Enter the latent dimension (recommended: 128): ", 128)

def build_generator(input_shape):
    generator = Sequential()
    generator.add(Dense(8 * 8 * 1024, input_shape=input_shape))
    generator.add(Reshape((8, 8, 1024)))
    generator.add(Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))
    return generator

def load_models(model_dir):
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".json"):
            model_name, _ = os.path.splitext(filename)
            parts = model_name.split('_')
            if len(parts) == 3 and parts[0] == "generator" and parts[1] == "architecture":
                epoch = int(parts[2])
                weights_name = f"generator_weights_{epoch}.h5"
                architecture_path = os.path.join(model_dir, filename)
                weights_path = os.path.join(model_dir, weights_name)

                with open(architecture_path, "r") as json_file:
                    model = model_from_json(json_file.read())

                model.load_weights(weights_path)
                models.append(model)
    return models

def generate_and_save_images(models, output_dir, input_shape, num_examples=16):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        expected_shape = model.input_shape[1:]          
        noise = np.random.normal(0, 1, (num_examples, *expected_shape))      
        generated_images = model.predict(noise)
        
        for j, img in enumerate(generated_images):
            img = (img * 255).astype(np.uint8)       
            img = Image.fromarray(img)
            save_path = os.path.join(output_dir, f"generated_image_model_{i}_sample_{j}.png")
            img.save(save_path)

with open("settings.json", "r") as settings_file:
    settings = json.load(settings_file)

def main():
    input_shape = (128, 128, 3)          
    model_dir = "./input/input_models"
    output_image_dir = "./output/output_model_images"

    try:
        models = load_models(model_dir)

        if not models:
            print("No models found in the 'input_models' folder.")
            return

        generate_and_save_images(models, output_image_dir, input_shape)
        print("Images generated and saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
