import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

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

def generate_and_save_images(models, output_dir, input_dim, num_examples=16):
    os.makedirs(output_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (num_examples, input_dim))
    
    for i, model in enumerate(models):
        generated_images = model.predict(noise)
        generated_images = 0.5 * generated_images + 0.5
        
        for j, img in enumerate(generated_images):
            plt.figure(figsize=(1, 1))
            plt.imshow(img)
            plt.axis('off')
            save_path = os.path.join(output_dir, f"generated_image_model_{i}_sample_{j}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

def main():
    input_dim = 100
    model_dir = "models"
    output_image_dir = "image_blends"
    output_model_dir = "model_blends"
    
    models = load_models(model_dir)
    
    if not models:
        print("No models found in the 'models' folder.")
        return
    
    generate_and_save_images(models, output_image_dir, input_dim)
    
    os.makedirs(output_model_dir, exist_ok=True)
    for i, model in enumerate(models):
        model_save_path = os.path.join(output_model_dir, f"blended_model_{i}.h5")
        model.save(model_save_path)
        print(f"Saved blended model {i} to {model_save_path}")

if __name__ == "__main__":
    main()
