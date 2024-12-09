import os
import numpy as np
from PIL import Image

image_folder = "./Data/COVID"
image_folder_1 = "./Data/NORMAL"
target_size = (256, 256)

def process_images(image_folder, target_size):
    image_features = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
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

"""""""""""
# COVID data has shape (1626, 65536). The rows are the number of data samples and cols are pixels
image_data = process_images(image_folder, target_size)
# If covid is present, 1. If normal, 0
one_col = np.ones((image_data.shape[0], 1))
image_data = np.concatenate((image_data, one_col), axis=1)
print(image_data)
image_data_normal = process_images(image_folder_1, target_size)
zeros_col = np.zeros((image_data_normal.shape[0], 1))
image_data_normal = np.concatenate((image_data_normal, zeros_col), axis=1)
print(image_data_normal)

#np.save('COVIDData', image_data) # saves data

#check_data = np.load('NormalData.npy') # for loading data

"""""

# Reduce Dimensionality of Data
