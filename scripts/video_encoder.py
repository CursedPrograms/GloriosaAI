import os
import cv2
import sys
import json
from glob import glob

def display_warning(message):
    print(f"\033[91m{message}\033[0m") 

# Assuming your JSON is stored in a file named 'config.json'
with open("settings.json", "r") as settings_file:
    settings = json.load(settings_file)

input_folder = settings["directories"]["video_frames"]
output_folder = settings["directories"]["video"]
output_base_name = "output_video"
output_extension = ".mp4"
max_videos_to_keep = 5  

if not os.path.exists(input_folder) or not any(file.endswith(".png") for file in os.listdir(input_folder)):
    display_warning("Warning: No PNG files found in the input folder or the input folder does not exist.")
    sys.exit()

os.makedirs(output_folder, exist_ok=True)

image_files = sorted(os.listdir(input_folder))

if not os.path.exists(input_folder):
    display_warning("Warning: The input folder does not exist.")
    sys.exit()

png_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]

if not png_files:
    display_warning("Warning: No PNG files found in the input folder.")
    sys.exit()

first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

output_number = 1

while True:
    output_path = os.path.join(output_folder, f"{output_base_name}_{output_number}{output_extension}")
    if not os.path.exists(output_path):
        break
    output_number += 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')         
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

for image_file in image_files:
    if image_file.endswith(".png"):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)

        if not out.isOpened():
            output_path = os.path.join(output_folder, f"{output_base_name}_{output_number}{output_extension}")
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        out.write(frame)

out.release()

print(f"Video saved to {output_path}")

existing_videos = sorted(glob(os.path.join(output_folder, f"{output_base_name}_*{output_extension}")))
videos_to_remove = max(0, len(existing_videos) - max_videos_to_keep)
for video_path in existing_videos[:videos_to_remove]:
    os.remove(video_path)

if __name__ == "__main__":
    sys.exit()
