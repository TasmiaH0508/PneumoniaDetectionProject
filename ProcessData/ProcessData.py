import os

import numpy as np
from PIL import Image

import torch

P_folder = "../RawData/PNEUMONIA"
NORMAL_folder_1 = "../RawData/NORMAL"
target_size = (256, 256)

def process_images(image_folder, target_size):
    ''''''''''
    Takes a folder of '.png' images and returns a 2D np array, where the rows are the data points and 
    the columns are the features(ie has shape (num data points, num columns)), each of which maps to a pixel
    '''''''''''
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

def pca(data):
    '''''''''''
    Takes a 2D torch array and returns a 2D torch array, with reduced number of features. The shape of the output 
    is (num data points, reduced number of features). Returns also the indices of the features kept.
    
    Data is expected to be normalised and of the type torch tensor.
    '''''
    data = data.T
    # data now has the shape (number of features, number of data points)
    num_data_points = data.shape[1]
    num_features = data.shape[0]
    mean_matrix = torch.mean(data, dim=1)
    mean_matrix = torch.reshape(mean_matrix, (num_features, 1))
    mean_centred_data = data - mean_matrix
    covariance_matrix = (mean_centred_data @ mean_centred_data.T) / num_data_points
    # perform svd
    U, S, _ = torch.linalg.svd(covariance_matrix)
    explained_variance_ratio = torch.cumsum(S, dim=0) / torch.sum(S)
    r = torch.searchsorted(explained_variance_ratio, 0.99) + 1
    U_reduced = U[:, : r]
    # reduce data
    Z = U_reduced.T @ data
    # perform reconstruction
    reconstructed_data = (U_reduced @ Z).T # transpose so that the data format remains the same
    return reconstructed_data

def pca_with_batch_processing(data, batch_size=400):
    '''''''''''
    Takes a 2D torch array and returns a 2D torch array, with reduced number of features. The shape of the output 
    is (num data points, reduced number of features). Returns also the indices of the features kept.

    Data is expected to be normalised and of the type torch tensor.
    '''''
    data = data.T # has shape (number of data points, number of features)
    num_data_points = data.shape[1]
    num_features = data.shape[0]
    mean_matrix = torch.mean(data, dim=1)
    mean_matrix = torch.reshape(mean_matrix, (num_features, 1))
    mean_centred_data = data - mean_matrix
    # do batch processing as covariance matrix takes a long time to be computed
    covariance_matrix = torch.zeros((num_features, num_features), dtype=torch.float64)
    for i in range(0, num_data_points, batch_size):
        batch_end = min(i + batch_size, num_data_points)
        batch = mean_centred_data[:, i : batch_end]
        covariance_matrix += batch @ batch.T
    print("Covariance computed")
    U, S, _ = torch.linalg.svd(covariance_matrix)
    print("Performed SVD")
    explained_variance_ratio = torch.cumsum(S, dim=0) / torch.sum(S)
    r = torch.searchsorted(explained_variance_ratio, 0.99).item() + 1
    U_reduced = U[:, : r]
    # reduce data
    Z = U_reduced.T @ data
    # perform reconstruction
    reconstructed_data = (U_reduced @ Z).T # transpose so that the data format remains the same
    print("Reconstructed data")
    return reconstructed_data

def normalise_data_min_max(data):
    '''''
    Input is a data matrix of shape (number of observations, number of data points) and is a torch tensor

    Returns data of the same shape after min-max scaling. Data is a tensor
    '''''''''
    min_matrix = torch.min(data, dim=0)[0]
    max_matrix = torch.max(data, dim=0)[0]
    range_matrix = max_matrix - min_matrix
    range_matrix = torch.where(range_matrix == 0, 1, range_matrix)
    normalised_data = (data - min_matrix) / range_matrix
    return normalised_data # is a tensor

def get_features_to_be_kept(data, threshold_var=0.01):
    ''''''''''
    Takes in data that has been reconstructed after PCA and the threshold variance. Is a torch tensor has 
    shape (number of observations, number of features)
    
    Returns the indices of the features kept, based on whether variance of feature >= threshold_var.
    '''
    variance_feature_wise = torch.var(data, dim=0)
    num_features = data.shape[1]
    feature_indices = torch.arange(num_features)
    indices_of_features_to_keep = torch.where(variance_feature_wise >= threshold_var, feature_indices, -1)
    print(indices_of_features_to_keep)
    indices_of_features_to_keep = torch.where(indices_of_features_to_keep != -1)[0]
    return indices_of_features_to_keep

def process_data_in_batches(raw_data_1, raw_data_2, sample_size=200):
    '''''''''
    Takes in 2 matrices, each of which is a numpy array of shape (num data points, num features). Note that num columns 
    must be the same for each matrix.
    
    Note that raw_data_1 will have the label of 0 and raw_data_2 will have the label of 1.
    
    WARNING: Data shape may change with different runs
    
    Procedure:
    1. Pick random rows from raw_data_1 and raw_data_2
    2. Stack the processed(reduced) raw_data_1 on top of raw_data_2
    3. Normalise the data. Perform PCA (on the test set only) and reconstruct the data
    4. Remove the features where there is very little variance and keep track of the indices removed
    '''''''''''
    number_of_data_points_data_1 = raw_data_1.shape[0]
    number_of_data_points_data_2 = raw_data_2.shape[0]

    number_of_features = raw_data_1.shape[1]

    # Prepare data for training
    np.random.seed(46)
    random_indices_1_train = np.random.choice(number_of_data_points_data_1, sample_size, replace=False)
    data_1 = raw_data_1[random_indices_1_train] # has shape: (sample_size, num_features)
    random_indices_2_train = np.random.choice(number_of_data_points_data_2, sample_size, replace=False)
    data_2 = raw_data_2[random_indices_2_train]
    # stack data_1 on TOP of data_2
    joined_training_data = np.vstack((data_1, data_2)) # has shape: (2*sample_size, num_features)
    print("Data joined with shape: ", joined_training_data.shape)

    # Normalisation should be done before PCA as PCA is sensitive to data with large ranges
    joined_training_data = torch.from_numpy(joined_training_data)
    joined_data = normalise_data_min_max(joined_training_data)
    print("Normalised data, min-max scaled.")

    data_chunks = []
    i = 0
    while i < number_of_features:
        # pick 64 * 64 columns at one time
        num_columns_to_pick = 64 * 64
        if i + num_columns_to_pick > number_of_features:
            data_to_add = joined_data[:, i : number_of_features]
        else:
            data_to_add = joined_data[:, i : i + num_columns_to_pick]
        data_chunks.append(data_to_add)
        i += num_columns_to_pick

    processed_training_data = None
    for data_chunk in data_chunks:
        processed_data_chunk = pca_with_batch_processing(data_chunk) # faster with batch processing
        if processed_training_data is None:
            processed_training_data = processed_data_chunk
        else:
            processed_training_data = torch.hstack((processed_training_data, processed_data_chunk))
    print("PCA complete and Data reconstructed")

    # Find the indices of the features that display little variance across samples
    indices_with_sufficiently_large_variance = get_features_to_be_kept(processed_training_data)
    processed_training_data = processed_training_data[:, indices_with_sufficiently_large_variance]

    # add in bias col for the training set
    total_samples = 2 * sample_size
    bias_col = torch.ones(total_samples)
    bias_col = torch.reshape(bias_col, (total_samples, 1))
    processed_training_data = torch.hstack((bias_col, processed_training_data))
    # add in the labels for the training set
    # the label for data 1(TOP) is 0 and for data 2(BOTTOM), it is 1
    zeros_col = torch.zeros(sample_size)
    zeros_col = torch.reshape(zeros_col, (sample_size, 1))
    ones_col = torch.ones(sample_size)
    ones_col = torch.reshape(ones_col, (sample_size, 1))
    label_col = torch.vstack((zeros_col, ones_col))
    processed_training_data = torch.hstack((processed_training_data, label_col))

    # Process data for test set
    unselected_pts_for_data_1 = np.arange(number_of_data_points_data_1)
    unselected_pts_for_data_1 = np.where(unselected_pts_for_data_1 != random_indices_1_train)[0]
    test_data_1 = raw_data_1[unselected_pts_for_data_1] # select the rows/observations first
    test_data_1 = test_data_1[:, indices_with_sufficiently_large_variance] # select the cols/features
    unselected_pts_for_data_2 = np.arange(number_of_data_points_data_2)
    unselected_pts_for_data_2 = np.where(unselected_pts_for_data_2 != random_indices_2_train)[0]
    test_data_2 = raw_data_2[unselected_pts_for_data_2] # select the rows/observations first
    test_data_2 = test_data_2[:, indices_with_sufficiently_large_variance] # select the cols/features
    joined_test_data = torch.vstack((test_data_1, test_data_2))

    number_of_data_pts_1_test = test_data_1.shape[0]
    number_of_data_pts_2_test = test_data_2.shape[0]

    zeros_col = torch.zeros(number_of_data_pts_1_test)
    zeros_col = torch.reshape(zeros_col, (number_of_data_pts_1_test, 1))
    ones_col = torch.ones(number_of_data_pts_2_test)
    ones_col = torch.reshape(ones_col, (number_of_data_pts_2_test, 1))
    label_col = torch.vstack((zeros_col, ones_col))

    total_number_of_test_data_pts = number_of_data_pts_1_test + number_of_data_pts_2_test
    bias_col = torch.ones(total_number_of_test_data_pts)
    bias_col = torch.reshape(bias_col, (total_number_of_test_data_pts, 1))

    processed_test_data = torch.hstack((bias_col, joined_test_data))
    processed_test_data = torch.hstack((processed_test_data, label_col))

    return processed_training_data, processed_test_data

# Process final data
raw_data_1 = process_images(NORMAL_folder_1, target_size)
raw_data_2 = process_images(P_folder, target_size)
# if pneumonia present, label is 1
PvNormalDataTrain, PvNormalDataTest = process_data_in_batches(raw_data_1, raw_data_2)

training_data_to_save = torch.tensor(PvNormalDataTrain).numpy()
testing_data_to_save = torch.tensor(PvNormalDataTest).numpy()
np.save("./ProcessedData/TrainingSet/PvNormalDataNormalised", training_data_to_save)
np.save("./ProcessedData/TestingSet/PvNormalDataNormalised", testing_data_to_save)