import os

data_root = 'training_data'

image_paths = []
labels = []

for subdir in os.listdir(data_root):
    subdir_path = os.path.join(data_root, subdir)
    
    if os.path.isdir(subdir_path):
        files = os.listdir(subdir_path)
        
        for file in files:
            image_path = os.path.join(subdir_path, file)
            image_paths.append(image_path)
            labels.append(subdir)         
