import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Sequential
from PIL import Image

def build_generator(input_shape):
    generator = Sequential()
    generator.add(Dense(7 * 7 * 256, input_shape=input_shape))     
    generator.add(Reshape((7, 7, 256)))
    generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    generator.add(Conv2D(3, (7, 7), padding='same', activation='sigmoid'))
    return generator

def load_models(model_dir):
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith("_architecture.json"):
            model_name = filename.split("_architecture.json")[0]
            architecture_path = os.path.join(model_dir, filename)
            weights_path = os.path.join(model_dir, f"{model_name}.h5")
            
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

def main():
    input_shape = (28, 28, 3)          
    model_dir = "blending_models"
    output_image_dir = "output_blends"

    models = load_models(model_dir)

    if not models:
        print("No models found in the 'blending_models' folder.")
        return

    generate_and_save_images(models, output_image_dir, input_shape)
    print("Images generated and saved successfully!")

if __name__ == "__main__":
    main()
