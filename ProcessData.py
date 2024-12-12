import os

import numpy as np
from PIL import Image

P_folder = "./Raw Data/PNEUMONIA"
NORMAL_folder_1 = "./Raw Data/NORMAL"
target_size = (256, 256) # The size (64, 64) can still work for a total of 400 samples

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

import torch

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

# utilise batch processing

def pca_with_batch_processing(data, batch_size=10):
    # data has the shape (number of data points, number of features)
    data = data.T
    data = torch.from_numpy(data)
    # data now has the shape (number of features, number of data points)
    num_features = data.shape[0]
    num_data_points = data.shape[1]
    mean_matrix = torch.mean(data, dim=0)
    mean_centred_data = torch.zeros_like(data)
    print(1)
    # batch processing
    for i in range(0, num_data_points, batch_size):
        batch_end = min(i + batch_size, num_data_points)
        mean_centred_data[:, i:batch_end] = data[:, i:batch_end] - mean_matrix
    print(2)
    # do batch processing
    covariance_matrix = torch.zeros((num_features, num_features))
    for i in range(0, num_features, batch_size):
        batch_end = min(i + batch_size, num_data_points)
        batch = mean_centred_data[:, i:batch_end]
        covariance_matrix += batch @ batch.T
    covariance_matrix /= num_data_points
    print(3)
    # perform svd
    U, S, _ = torch.linalg.svd(covariance_matrix)
    explained_variance_ratio = torch.cumsum(S, dim=0)/ torch.sum(S)
    r = torch.searchsorted(explained_variance_ratio, 0.99) + 1
    U_reduced = U[:, : r].float()
    print("SVD performed and U reduced")
    # reduce data, no reconstruction is performed since the goal here is to reduce dimensions
    # take transpose so that the shape is the same as the original data
    # do batch processing
    Z = torch.zeros((num_data_points, r), dtype=torch.float)
    for i in range(0, num_data_points, batch_size):
        batch_end = min(i + batch_size, num_data_points)
        batch = mean_centred_data[:, i:batch_end].float()
        Z[i:batch_end] = (U_reduced.T @ batch).T
    return Z # is a tensor

def process_data():
    # todo: remove if not needed
    number_of_rows_to_select_randomly = 200  # Need to cut down on data samples as data test set is too large

    P_data = process_images(P_folder, target_size)
    number_of_data_points_for_P = P_data.shape[0]
    random_indices_P = np.random.choice(number_of_data_points_for_P, number_of_rows_to_select_randomly, replace=False)
    P_data = P_data[random_indices_P]
    print("Pneumonia data has been processed.")

    normal_data = process_images(NORMAL_folder_1, target_size)
    number_of_data_points_for_NORMAL = normal_data.shape[0]
    random_indices_Normal = np.random.choice(number_of_data_points_for_NORMAL, number_of_rows_to_select_randomly,
                                             replace=False)
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
    output_col = np.vstack((ones_col, zeros_col))
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

    # np.save('COVIDData', image_data) # saves data

    # check_data = np.load('NormalData.npy') # for loading data

def process_data_in_batches(folder_1, folder_2, sample_size=200):
    processed_data_1 = process_images(folder_1, target_size)
    number_of_data_points_data_1 = processed_data_1.shape[0]
    processed_data_2 = process_images(folder_2, target_size)
    number_of_data_points_data_2 = processed_data_2.shape[0]

    number_of_features = 65536

    # reduce data size
    np.random.seed(46)
    random_indices = np.random.choice(number_of_data_points_data_1, sample_size, replace=False)
    data_1 = processed_data_1[random_indices] # has shape: (sample_size, 65536)
    random_indices = np.random.choice(number_of_data_points_data_2, sample_size, replace=False)
    data_2 = processed_data_2[random_indices]
    # stack data_1 on TOP of data_2
    joined_data = np.vstack((data_1, data_2)) # has shape: (2*sample_size, 65536) ~ (400, 65536) if sample size is 200

    # process data_1
    data_chunks = []
    i = 0
    while i < number_of_features:
        # pick 64 * 64 columns at one time
        num_columns_to_pick = 64 * 64
        if i + num_columns_to_pick > number_of_features:
            data_to_add = joined_data[i:number_of_features]
        else:
            data_to_add = joined_data[:, i : i + num_columns_to_pick]
        data_chunks.append(data_to_add)
        i += num_columns_to_pick

    processed_data = None
    for data_chunk in data_chunks:
        processed_data_chunk = pca(data_chunk)
        if processed_data is None:
            processed_data = processed_data_chunk
        else:
            processed_data = torch.hstack((processed_data, processed_data_chunk))

    # add in bias col
    total_samples = number_of_data_points_data_1 + number_of_data_points_data_2
    bias_col = torch.ones(total_samples)
    bias_col = torch.reshape(bias_col, (total_samples, 1))
    processed_data = np.hstack((bias_col, processed_data))

    # add in the labels
    # the label for data 1(TOP) is 0 and for data 2(BOTTOM), it is 1
    zeros_col = torch.zeros(number_of_data_points_data_1)
    zeros_col = torch.reshape(zeros_col, (number_of_data_points_data_1, 1))
    ones_col = torch.ones(number_of_data_points_data_2)
    ones_col = torch.reshape(ones_col, (number_of_data_points_data_2, 1))
    label_col = torch.vstack((zeros_col, ones_col))
    processed_data = np.hstack((processed_data, label_col))

    #todo: save the data


#process_data_in_batches(P_folder, NORMAL_folder_1)

#process_data()