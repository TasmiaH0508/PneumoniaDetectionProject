import os

import numpy as np
from PIL import Image

P_folder = "./Raw Data/PNEUMONIA"
NORMAL_folder_1 = "./Raw Data/NORMAL"
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

# Reduce Dimensionality of Data

def pca(data):
    # data has the shape (number of data points, number of features)
    data = data.T
    print("Data has been transposed.")
    # data now has the shape (number of features, number of sata points)
    num_data_points = data.shape[1]
    print(1)
    mean_matrix = np.mean(data, axis=1, keepdims=True)
    print(2)
    mean_centred_data = data - mean_matrix
    print(3)
    covariance_matrix = (mean_centred_data @ mean_centred_data.T) / num_data_points # takes a long time for covariance matrix to be computed -> maybe consider using built in numpy function
    print("Covariance computed")
    # perform svd
    U, S, _ = np.linalg.svd(covariance_matrix)
    explained_variance_ratio = np.cumsum(S) / np.sum(S)
    r = np.searchsorted(explained_variance_ratio, 0.99) + 1
    U_reduced = U[:, : r]
    # reduce data, no reconstruction is performed since the goal here is to reduce dimensions
    Z = U_reduced.T @ data
    Z = Z.T
    print("Z has been computed")
    # change Z so that it has the same shape as original data: (number of data points, number of features)
    return Z

number_of_rows_to_select_randomly = 1 # Need to cut down on data samples as data test set is too large

P_data = process_images(P_folder, target_size)
number_of_data_points_for_P = P_data.shape[0]
random_indices_P = np.random.choice(number_of_data_points_for_P, number_of_rows_to_select_randomly, replace=False)
P_data = P_data[random_indices_P]
print("Pneumonia data has been processed.")

normal_data = process_images(NORMAL_folder_1, target_size)
number_of_data_points_for_NORMAL = normal_data.shape[0]
random_indices_Normal = np.random.choice(number_of_data_points_for_NORMAL, number_of_rows_to_select_randomly, replace=False)
normal_data = normal_data[random_indices_Normal]
print("Normal data has been processed.")

# join the data together by stacking the P_data on top of the normal
combined_data = np.vstack((P_data, normal_data))
print("Data has been combined.")

# perform pca to reduce dimensionality of data
reduced_combined_data = pca(combined_data)
print("PCA complete")

# add in the output column
# 0 if normal and 1 if pneumonia is present
number_of_pneumonia_samples = P_data.shape[0]
ones_col = np.ones(number_of_pneumonia_samples)
number_of_normal_samples = normal_data.shape[0]
total_samples = number_of_pneumonia_samples + number_of_normal_samples
zeros_col = np.zeros(number_of_normal_samples)
output_col = np.hstack((ones_col, zeros_col))
output_col = np.reshape(output_col, (total_samples, 1))
processed_data = np.hstack((reduced_combined_data, output_col))
bias_col = np.ones(total_samples)
bias_col = np.reshape(bias_col, (total_samples, 1))
processed_data = np.hstack((bias_col, processed_data))
print("Data has been processed.")

# shuffle the data
np.random.seed(42)
np.random.shuffle(processed_data)

# save the data that has been processed as csv file
np.save('./Processed Data/PvNormalData', processed_data)
print("Data has been shuffled and saved.")

#np.save('COVIDData', image_data) # saves data

#check_data = np.load('NormalData.npy') # for loading data