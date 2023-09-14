Prerequisites
Before diving into the code, make sure you have the following prerequisites installed on your system:

Python 3.x
TensorFlow (tested with TensorFlow 2.x)
NumPy
Matplotlib
Pillow (PIL)
Optional: A dataset of images for training (recommended specifications: 128 x 128 resolution, RGB format, .PNG files)
You can install the required Python packages using pip with the following commands:

pip install tensorflow
pip install numpy
pip install matplotlib
pip install Pillow

How to Use
Follow these steps to effectively utilize the provided code:


1. Data Preparation (Optional)
If you have a dataset for training, place it in a directory named training_data within the same location as your script or notebook. The script will utilize an ImageDataGenerator to load and preprocess this data.

2. Configure Training Parameters
Customize the training parameters to suit your specific needs:

epochs: Set the number of training epochs you desire.
batch_size: Define the batch size used during training.
input_dim: Specify the dimension of the random noise input for the generator.
image_shape: Set the shape of the images in your dataset (e.g., (128, 128, 3) for 128x128 resolution RGB images).

3. Training
Run the script, and the GAN will begin training. During training, it will periodically save generated images and model weights in the output_images and saved_models directories, respectively.

4. Monitoring Progress
The script will print important information during training:

The current epoch
Discriminator loss and accuracy
Generator loss
Additionally, you can visually monitor the generated images in the output_images directory.

5. Save Models (Optional)
The generator and discriminator model weights will be automatically saved to the saved_models directory at regular intervals (by default, every 1000 epochs). You can adjust this interval as needed in the code.

6. Post-Training Usage
After training is complete, you can use the trained generator to generate new images by loading the saved generator model weights.
