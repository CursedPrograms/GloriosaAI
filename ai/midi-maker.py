import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation

def generate_music_data(length=100, num_notes=88):
    return np.random.randint(0, 2, size=(length, num_notes))

model = Sequential()
model.add(LSTM(256, input_shape=(None, 88), return_sequences=True))
model.add(Dense(88))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

X_train = np.expand_dims(generate_music_data(), axis=0)
y_train = np.roll(X_train, shift=-1, axis=1)

model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes=2), epochs=50, batch_size=1)

def generate_music(seed, length=100):
    generated_music = []
    for _ in range(length):
        next_note_probs = model.predict(np.expand_dims(seed, axis=0))[0][-1]
        next_note = np.random.choice(np.arange(88), p=next_note_probs)
        generated_music.append(next_note)
        seed = np.roll(seed, shift=-1)
        seed[-1] = next_note
    return generated_music

seed_music = generate_music_data(length=50)
generated_music = generate_music(seed_music, length=100)
print("Generated Music:", generated_music)
