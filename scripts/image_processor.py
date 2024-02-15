from PIL import Image
import os
import random

# Set the input and output directories
input_dir = './unprocessed_images'
output_dir = './training_data/processed_images'
os.makedirs(output_dir, exist_ok=True)

# Set the size of the processed images
processed_size = (128, 128)

# Set the range for scaling and zooming
min_scale_factor = 0.8
max_scale_factor = 1.2

# Set the number of variations for each image
num_variations = 3

# Iterate through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Open the image
        input_path = os.path.join(input_dir, filename)
        image = Image.open(input_path)

        for variation in range(num_variations):
            # Create a copy of the original image for each variation
            processed_image = image.copy()

            # Resize the image to the desired size
            processed_image = processed_image.resize(processed_size)

            # Randomly scale the image
            scale_factor = random.uniform(min_scale_factor, max_scale_factor)
            new_size = (int(processed_size[0] * scale_factor), int(processed_size[1] * scale_factor))
            processed_image = processed_image.resize(new_size)

            # Randomly flip the image horizontally
            if random.choice([True, False]):
                processed_image = processed_image.transpose(Image.FLIP_LEFT_RIGHT)

            # Randomly flip the image vertically
            if random.choice([True, False]):
                processed_image = processed_image.transpose(Image.FLIP_TOP_BOTTOM)

            # Save the processed image
            output_path = os.path.join(output_dir, f'processed_{filename.split(".")[0]}_{variation}.png')
            processed_image.save(output_path)

print(f'Images processed and saved in {output_dir}')
