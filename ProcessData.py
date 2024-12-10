import os
from statistics import covariance

import numpy as np
from PIL import Image

image_folder = "./Raw Data/COVID"
image_folder_1 = "./Raw Data/NORMAL"
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


#np.save('COVIDData', image_data) # saves data

#check_data = np.load('NormalData.npy') # for loading data

# Reduce Dimensionality of Data

#image_data = process_images(image_folder, target_size) # rows are the number of samples
#print(image_data.shape)

def pca(data):
    # data has the shape (number of data points, number of features)
    data = data.T
    num_data_points = data.shape[0]
    num_features = data.shape[1]
    # data now has the shape (number of features, number of data points)
    # -> the rows are the features and the cols are data points
    mean_matrix = np.tile(np.reshape(np.mean(data, axis=1), (num_features, 1)), num_data_points)
    mean_centred_data = data - mean_matrix
    covariance_matrix = 1 / num_data_points * mean_centred_data @ mean_centred_data.T
    # perform svd
    U, S, _ = np.linalg.svd(covariance_matrix)
    sum_of_diagonal_values = np.sum(S, axis=0)
    sum = 0
    r = 0
    while sum / sum_of_diagonal_values < 0.99:
        sum += S[r]
        r += 1
    print(r)
    print(U)
    U_reduced = U[: r] # incorrect
    print(U_reduced)
    # reduce data
    Z = U_reduced.T @ data
    # reconstruct data
    return U_reduced @ Z

data = np.array([[1, 1], [5, 133]])
#data = process_images(image_folder, target_size)
print(pca(data))