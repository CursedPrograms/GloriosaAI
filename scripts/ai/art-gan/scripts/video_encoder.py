import os
import cv2

input_folder = "video_frames"
output_folder = "output_video"

os.makedirs(output_folder, exist_ok=True)

image_files = sorted(os.listdir(input_folder))

if not any(image.endswith(".png") for image in image_files):
    print("No PNG files found in the input folder.")
    exit()

first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

output_base_name = "output_video"
output_extension = ".mp4"
output_path = os.path.join(output_folder, output_base_name + output_extension)
output_number = 1

while os.path.exists(output_path):
    output_base_name = f"output_video_{output_number}"
    output_path = os.path.join(output_folder, output_base_name + output_extension)
    output_number += 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')         
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

for image_file in image_files:
    if image_file.endswith(".png"):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

out.release()

print(f"Video saved to {output_path}")

if __name__ == "__main__":
    exit()