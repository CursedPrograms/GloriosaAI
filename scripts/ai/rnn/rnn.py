import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Generate random art data (replace this with your own dataset)
def generate_art_data(num_samples, sequence_length, feature_dim):
    return np.random.rand(num_samples, sequence_length, feature_dim)

# Define the RNN model for art generation
def build_art_rnn_model(sequence_length, feature_dim):
    model = models.Sequential()

    # LSTM layer with return_sequences=True to output a sequence
    model.add(layers.LSTM(256, input_shape=(sequence_length, feature_dim), return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Dense layer for outputting features
    model.add(layers.Dense(feature_dim, activation='sigmoid'))

    return model

# Hyperparameters
num_samples = 1000
sequence_length = 50
feature_dim = 3  # For RGB images, use 3; adjust for other types of art data

# Generate synthetic art data
art_data = generate_art_data(num_samples, sequence_length, feature_dim)

# Build and compile the RNN model
art_rnn_model = build_art_rnn_model(sequence_length, feature_dim)
art_rnn_model.compile(optimizer='adam', loss='mse')  # Adjust the loss function as needed

# Train the RNN model
art_rnn_model.fit(art_data, art_data, epochs=50, batch_size=32)  # Adjust epochs and batch_size

# Generate new art sequences
generated_art = art_rnn_model.predict(art_data)

# Display some examples of the generated art
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(art_data[i], cmap='viridis')  # Display original art
    plt.subplot(2, 5, i + 6)
    plt.imshow(generated_art[i], cmap='viridis')  # Display generated art

plt.show()
