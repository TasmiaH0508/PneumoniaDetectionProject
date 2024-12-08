import os
import numpy as np
from PIL import Image # use pillow for image processing

image_folder = "./chest_xray/train/NORMAL"
target_size = (1857, 1317) # resize all the images to be the smallest possible

image_folder_1 = "./chest_xray/train/NORMAL"
image_folder_2 = "./chest_xray/train/PNEUMONIA"

def find_smallest_size(image_folder):
    min_width = float('inf')
    min_height = float('inf')
    min_file = ""
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)
        image_size = img.size
        if image_size[0] < min_height:
            min_file = filename
            min_height = image_size[0]
        if image_size[1] < min_width:
            min_file = filename
            min_width = image_size[1]
    print(min_file)
    return min_width, min_height

print(find_smallest_size(image_folder_1))
print(find_smallest_size(image_folder_2))

def process_images(image_folder, target_size):
    image_features = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).convert("L")  # Convert to grayscale

            # Resize image
            img_resized = img.resize(target_size)

            # Flatten to 1D feature vector
            img_array = np.array(img_resized).flatten()

            # Normalize pixel values to range [0, 1]
            img_array = img_array / 255.0

            # Append to list
            image_features.append(img_array)
    return np.array(image_features)

"""""""""
# Process all images
image_data = process_images(image_folder, target_size)

print(image_data.shape)
print(image_data[0])

print("Processing complete")
"""""