import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess

output_dim = (128, 128)

output_folder = os.path.join(os.path.dirname(__file__), 'styled_images')
os.makedirs(output_folder, exist_ok=True)

def get_first_png_file(folder):
    for file in os.listdir(folder):
        if file.lower().endswith('.png'):
            return os.path.join(folder, file)
    return None

content_folder = os.path.join(os.path.dirname(__file__), 'image_edit')
style_folder = os.path.join(os.path.dirname(__file__), 'image_style')

content_file = get_first_png_file(content_folder)
style_file = get_first_png_file(style_folder)

if content_file is None or style_file is None:
    print("No PNG files found in the content or style folders.")
    exit()

def load_and_preprocess_image(image_path, output_dim):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, output_dim)
    image = image[tf.newaxis, :]
    return image

def preprocess_image(image):
    image = tf.keras.applications.vgg19.preprocess_input(image * 255)
    return image

def deprocess_image(processed_image):
    x = (processed_image[0] + 1) / 2.0
    x = np.clip(x, 0, 1)
    return x

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
content_layers = ['block5_conv2'] 

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

def vgg_layers(layer_names):
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

style_image = load_and_preprocess_image(os.path.join(style_folder, 'style.png'), output_dim)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

style_features = [tf.reduce_mean(output, axis=(1, 2), keepdims=True) for output in style_outputs]

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers):
        super(StyleModel, self).__init__()
        self.vgg = vgg_layers(style_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = [gram_matrix(output) for output in outputs]

        return style_outputs

style_model = StyleModel(style_layers)

content_image = load_and_preprocess_image(os.path.join(content_folder, 'content.png'), output_dim)
content_target = content_image

style_target = style_model(style_image)

def total_variation_loss(image):
    image_shape = tf.shape(image)
    x_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    y_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    x_loss = tf.reduce_sum(tf.square(x_deltas)) / tf.cast(image_shape[1] - 1, tf.float32)
    y_loss = tf.reduce_sum(tf.square(y_deltas)) / tf.cast(image_shape[2] - 1, tf.float32)
    
    return x_loss + y_loss

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

def calculate_loss(image):
    image = image * 255.0
    
    content_features = vgg_layers(content_layers)(content_image)
    image_features = vgg_layers(content_layers)(image)
    
    content_loss = tf.reduce_mean((image_features - content_features) ** 2)
    
    style_features = style_model(image)
    style_loss = tf.add_n([tf.reduce_mean((style_features[i] - style_target[i]) ** 2)
                           for i in range(num_style_layers)])
    
    tv_loss = total_variation_weight * total_variation_loss(image)
    
    total_loss = content_weight * content_loss + style_weight * style_loss + tv_loss
    
    return total_loss

image = tf.Variable(content_image)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        loss = calculate_loss(image)
    gradients = tape.gradient(loss, image)
    opt.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

num_iterations = 1000

for i in range(num_iterations):
    train_step(image)
    if i % 100 == 0:
        img = deprocess_image(image.numpy())
        plt.imshow(img)
        plt.title(f"Iteration {i}")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'output_{i}.png'))
        plt.show()

final_image = deprocess_image(image.numpy())
plt.imshow(final_image)
plt.axis('off')
plt.savefig(os.path.join(output_folder, 'final_output.png'))
plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main_script_path = os.path.join(script_dir, "..", "main.py")
    
    subprocess.run(["python", main_script_path])
