import numpy as np
import matplotlib.pyplot as plt

def generate_art_pattern(size):

    art_pattern = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i % 2 == 0 and j % 2 == 0:
                art_pattern[i, j] = 1
            elif i % 3 == 0 or j % 3 == 0:
                art_pattern[i, j] = 0.5
            else:
                art_pattern[i, j] = 0

    return art_pattern

art_pattern = generate_art_pattern(10)

plt.imshow(art_pattern, cmap='viridis')
plt.title('Logic-Based Art Generation')
plt.show()
