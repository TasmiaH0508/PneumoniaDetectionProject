import os

import numpy as np
from PIL import Image, ImageOps

import torch

P_folder = "./Models/Data/RawData/PNEUMONIA"
NORMAL_folder_1 = "./Models/Data/RawData/NORMAL"
target_size = (256, 256)

def process_image(image_path):
    ''''
    Returns image as a torch tensor of shape (1, num_features) without any bias added.
    '''
    try:
        image = Image.open(image_path).convert("L")

        image_height, image_width = image.size
        if image_height < target_size[0] or image_width < target_size[1]:
            image.thumbnail(target_size, Image.LANCZOS)
            delta_w = 256 - image.size[0]
            delta_h = 256 - image.size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
            image = ImageOps.expand(image, padding, fill=0)
        else:
            image = image.resize((256, 256), Image.LANCZOS)

        image = np.array(image).flatten()
        image = image / 255
        image = torch.from_numpy(image).float()
        num_features = image.shape[0]
        image = torch.reshape(image, (1, num_features))
        return image
    except FileNotFoundError:
        print("File not found")

def process_images(image_folder, target_size):
    ''''
    Takes a folder of '.png' images and returns a 2D np array, where the rows are the data points and 
    the columns are the features(ie has shape (num data points, num columns)), each of which maps to a pixel
    '''
    image_features = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            # Load image
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).convert("L")  # Convert to grayscale
            img_height, img_width = img.size

            # Resize image
            if img_height < target_size[0] or img_width < target_size[1]:
                img.thumbnail((256, 256), Image.LANCZOS)
                delta_w = 256 - img.size[0]
                delta_h = 256 - img.size[1]
                padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
                img_resized= ImageOps.expand(img, padding, fill=0)
            else:
                img_resized = img.resize((256, 256), Image.LANCZOS)

            # Flatten to 1D feature vector
            img_array = np.array(img_resized).flatten()

            # Normalize pixel values to range [0, 1]
            img_array = img_array / 255.0

            # Append to list
            image_features.append(img_array)
    return np.array(image_features)

def find_min_max_dimensions_in_image_folder(image_folder):
    min_size = float('inf')
    max_size = float('-inf')
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).convert("L")
            img_shape = img.size
            img_pixels = img_shape[0] * img_shape[1]
            if img_pixels < min_size:
                min_size = img_pixels
            if img_pixels > max_size:
                max_size = img_pixels
    return min_size, max_size

def pca(data):
    ''''
    Takes a 2D torch array and returns a 2D torch array, with reduced number of features. The shape of the output 
    is (num data points, reduced number of features). Returns also the indices of the features kept.

    Data is expected to be normalised and of the type torch tensor.
    '''
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
    reconstructed_data = (U_reduced @ Z).T  # transpose so that the data format remains the same
    return reconstructed_data


def pca_with_batch_processing(data, batch_size=400):
    ''''
    Takes a 2D torch array and returns a 2D torch array, with reduced number of features. The shape of the output 
    is (num data points, reduced number of features). Returns also the indices of the features kept.

    Data is expected to be normalised and of the type torch tensor.
    '''
    data = data.T  # after transposing, the dimensions are (num features, num_samples)
    num_data_points = data.shape[1]
    num_features = data.shape[0]
    mean_matrix = torch.mean(data, dim=1) # computed over all features
    mean_matrix = torch.reshape(mean_matrix, (num_features, 1))
    mean_centred_data = data - mean_matrix
    # do batch processing as covariance matrix takes a long time to be computed
    covariance_matrix = torch.zeros((num_features, num_features), dtype=torch.float64)
    for i in range(0, num_data_points, batch_size):
        batch_end = min(i + batch_size, num_data_points)
        batch = mean_centred_data[:, i: batch_end]
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
    reconstructed_data = (U_reduced @ Z).T  # transpose so that the data format remains the same
    print("Reconstructed data")
    return reconstructed_data


def normalise_data_min_max(data):
    ''''
    Input is a data matrix of shape (number of observations, number of data points) and is a torch tensor

    Returns data(tensor) after min-max scaling, min matrix and range matrix for future computations.
    '''
    min_matrix = torch.min(data, dim=0)[0]
    max_matrix = torch.max(data, dim=0)[0]
    range_matrix = max_matrix - min_matrix
    range_matrix = torch.where(range_matrix == 0, 1, range_matrix)
    normalised_data = (data - min_matrix) / range_matrix
    return normalised_data, min_matrix, range_matrix

def min_max_normalise_with_predefined_params(data, min_matrix, range_matrix):
    data = (data - min_matrix) / range_matrix
    return data

def get_features_with_sufficient_var(data, threshold_var=0.01):
    ''''
    Takes in data that has been reconstructed after PCA and the threshold variance. Is a torch tensor has 
    shape (number of observations, number of features)

    Returns the indices of the features kept, based on whether variance of feature >= threshold_var. Is a torch tensor
    '''
    variance_feature_wise = torch.var(data, dim=0)
    num_features = data.shape[1]
    feature_indices = torch.arange(num_features)
    indices_of_features_to_keep = torch.where(variance_feature_wise >= threshold_var, feature_indices, -1)
    indices_of_features_to_keep = torch.where(indices_of_features_to_keep != -1)[0]
    return indices_of_features_to_keep  # torch tensor


def stack_and_label_data(data_1, data_2, bias=True):
    ''''
    Stacks data_1 on top data_2. data_1 will have the label 0 and data_2 will have the label 1.
    Adds the bias col and label col.
    data_1 and data_2 are torch tensors.
    '''
    num_data_pts_1 = data_1.shape[0]
    num_data_pts_2 = data_2.shape[0]
    zeros_col = torch.zeros(num_data_pts_1)
    zeros_col = torch.reshape(zeros_col, (num_data_pts_1, 1))
    ones_col = torch.ones(num_data_pts_2)
    ones_col = torch.reshape(ones_col, (num_data_pts_2, 1))
    label_col = torch.vstack((zeros_col, ones_col))

    joined_data = torch.vstack((data_1, data_2))

    processed_data = torch.hstack((joined_data, label_col))

    if bias:
        total_data_pts = num_data_pts_1 + num_data_pts_2
        bias_col = torch.ones(total_data_pts)
        bias_col = torch.reshape(bias_col, (total_data_pts, 1))
        processed_data = torch.hstack((bias_col, processed_data))

    return processed_data


def add_bias_and_label(data, num_zero_label):
    ''''
    Note: Used for data that has been stacked beforehand

    Data has the shape of (total num of observations, features). Data is a torch tensor
    Prepends bias col and appends the labels col, where the 0s are above the 1s
    '''
    total_samples = data.shape[0]
    bias_col = torch.ones(total_samples)
    bias_col = torch.reshape(bias_col, (total_samples, 1))

    zeros_col = torch.zeros(num_zero_label)
    zeros_col = torch.reshape(zeros_col, (num_zero_label, 1))
    num_one_label = total_samples - num_zero_label
    one_col = torch.ones(num_one_label)
    one_col = torch.reshape(one_col, (num_one_label, 1))
    label_col = torch.vstack((zeros_col, one_col))

    processed_data = torch.hstack((bias_col, data, label_col))

    return processed_data


def pick_observations_and_features(data, rows_to_remove, cols_to_keep):
    ''''
    Takes data(torch tensor), which is of shape (number of observations, number of features), rows_to_remove(np array)
    and cols_to_keep(torch tensor)
    
    Note: done before labels and bias col have been added

    Returns data, with relevant rows removed and features kept, as a torch tensor
    '''
    if rows_to_remove is not None:
        num_observations = data.shape[0]
        temp = -np.ones(num_observations)
        temp[rows_to_remove] = rows_to_remove
        rows_to_remove = temp
        rows_to_keep = np.arange(num_observations)
        rows_to_keep = np.where(rows_to_keep != rows_to_remove, rows_to_keep, -1)
        rows_to_keep = np.where(rows_to_keep != -1)[0]
        data = data[rows_to_keep]
    if cols_to_keep is not None:
        data = data[:, cols_to_keep]
    return data

def process_test_and_training_data_in_batches(raw_data_1, raw_data_2, var=0.04, sample_size=200, reduce_features=True):
    ''''
    Takes in 2 matrices, each of which is a numpy array of shape (num data points, num features). Note that num columns 
    must be the same for each matrix. If reduce_features is set to False, pca is not applied. If var=0, no features are removed.

    Note: raw_data_1 will have the label of 0 and raw_data_2 will have the label of 1. A bias column is also added.

    WARNING: Data shape may change with different runs

    Procedure:
    1. Pick random rows from raw_data_1 and raw_data_2
    2. Stack the processed(reduced) raw_data_1 on top of raw_data_2
    3. Normalise the data. Perform PCA (on the test set only) and reconstruct the data
    4. Remove the features where there is very little variance and keep track of the indices removed
    5. Remove features from the test set
    '''
    raw_data_1 = torch.from_numpy(raw_data_1)
    raw_data_2 = torch.from_numpy(raw_data_2)

    number_of_data_points_data_1 = raw_data_1.shape[0]
    number_of_data_points_data_2 = raw_data_2.shape[0]

    number_of_features = raw_data_1.shape[1]

    # Prepare data for training set
    np.random.seed(46)
    random_indices_1_train = np.random.choice(number_of_data_points_data_1, sample_size, replace=False)
    train_data_1 = raw_data_1[random_indices_1_train]  # has shape: (sample_size, num_features)
    random_indices_2_train = np.random.choice(number_of_data_points_data_2, sample_size, replace=False)
    train_data_2 = raw_data_2[random_indices_2_train]
    # stack data_1 on TOP of data_2
    joined_training_data = torch.vstack((train_data_1, train_data_2))  # has shape: (2*sample_size, num_features)
    print("Data joined with shape: ", joined_training_data.shape)

    # Normalisation should be done before PCA as PCA is sensitive to data with large ranges
    joined_data, min_matrix, range_matrix = normalise_data_min_max(joined_training_data)
    print("Normalised training data, min-max scaled.")

    if reduce_features:
        data_chunks = []
        i = 0
        while i < number_of_features:
            # pick 64 * 64 columns at one time
            num_columns_to_pick = 64 * 64
            if i + num_columns_to_pick > number_of_features:
                data_to_add = joined_data[:, i: number_of_features]
            else:
                data_to_add = joined_data[:, i: i + num_columns_to_pick]
            data_chunks.append(data_to_add)
            i += num_columns_to_pick

        processed_training_data = None
        for data_chunk in data_chunks:
            processed_data_chunk = pca_with_batch_processing(data_chunk)  # faster with batch processing
            if processed_training_data is None:
                processed_training_data = processed_data_chunk
            else:
                processed_training_data = torch.hstack((processed_training_data, processed_data_chunk))
        print("PCA complete and Data reconstructed")

        # Find the indices of the features that display little variance across samples, threshold_var is 0.01 if not declared otherwise
        indices_with_sufficiently_large_variance = get_features_with_sufficient_var(processed_training_data,
                                                                                    threshold_var=var)
    else:
        processed_training_data = joined_data
        indices_with_sufficiently_large_variance = get_features_with_sufficient_var(joined_data, threshold_var=var)

    # Process training data by removing irrelevant features, adding bias col and label col
    processed_training_data = processed_training_data[:, indices_with_sufficiently_large_variance]
    processed_training_data = add_bias_and_label(processed_training_data, sample_size)

    # Process data for test set by normalising the entire raw data set with the previously computed min and range matrices
    # Then remove the features and data points(that are already present in the training set
    raw_data_1 = min_max_normalise_with_predefined_params(raw_data_1, min_matrix, range_matrix)
    raw_data_2 = min_max_normalise_with_predefined_params(raw_data_2, min_matrix, range_matrix)
    test_data_1 = pick_observations_and_features(raw_data_1, random_indices_1_train,
                                                 indices_with_sufficiently_large_variance)
    test_data_2 = pick_observations_and_features(raw_data_2, random_indices_2_train,
                                                 indices_with_sufficiently_large_variance)
    print(test_data_1.shape)
    print(test_data_2.shape)

    # Stack data, normalise, add bias col and label col
    processed_test_data = stack_and_label_data(test_data_1, test_data_2)

    print("The shape of the training data is:", processed_training_data.shape)
    print("The shape of the test data is:", processed_test_data.shape)
    print("The number of features has been reduced by(in %):",
          ((number_of_features - (processed_training_data.shape[1] - 2)) / number_of_features) * 100)

    # the mean and range matrices are calculated based on the training set
    return processed_training_data, processed_test_data, indices_with_sufficiently_large_variance, min_matrix, range_matrix

def prepare_data():
    ''''
    Prepares data for training and testing for various machine learning techniques, that is not CNN.
    '''
    # Process final data
    raw_data_1 = process_images(NORMAL_folder_1, target_size)
    raw_data_2 = process_images(P_folder, target_size)
    # if pneumonia present, label is 1
    PvNormalDataTrain, PvNormalDataTest, indices_kept, min_matrix, range_matrix = process_test_and_training_data_in_batches(raw_data_1, raw_data_2, var=0.02, reduce_features=True)

    training_data_to_save = PvNormalDataTrain.numpy()
    testing_data_to_save = PvNormalDataTest.numpy()
    np.save("./Models/Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.02", training_data_to_save)
    np.save("./Models/Data/ProcessedRawData/TestSet/PvNormalDataNormalised_var0.02", testing_data_to_save)
    # to use the indices, the images must be turned into arrays first. Then, select the cols to keep using indices_kept.
    # Then add in the bias if needed. Add in the label if needed.
    np.save("./Models/Data/ProcessedRawData/Index/Indices_Kept_data_var0.02", indices_kept)
    np.save("./Models/Data/ProcessedRawData/MinData/min_across_all_features_var0.02", min_matrix)
    np.save("./Models/Data/ProcessedRawData/RangeData/range_across_all_features_var0.02", range_matrix)