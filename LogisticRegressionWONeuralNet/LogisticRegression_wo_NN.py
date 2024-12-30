import torch
from PrepareData import *
import numpy as np
import time

class LargeDataSizeError(Exception):
    pass

large_feature_num = 2400

def transform_features(data, poly_deg=2):
    ''''
    Takes data(torch tensor), which does not have the bias or label cols and transforms the features
    by combining features.

    Suitable for small datasets.

    For e.g., [u, v] -> [u, v, u^2, u.v, v^2], for poly_deg=2
    '''
    pow_to_range = dict() # stores the ranges at which linear, quadratic, cubic terms are present...
    pow_to_range[1] = range(0, data.shape[1])
    has_been_computed = set()
    if poly_deg >= 2:
        # by the end of every iteration of the outer loop, (transformed) features of deg_i are added.
        for deg_i in range(2, poly_deg + 1):
            low_deg = 1
            high_deg = deg_i - low_deg
            while low_deg <= high_deg:
                range_of_low_deg = pow_to_range[low_deg]
                range_of_high_deg = pow_to_range[high_deg]
                # double loop combines 2 features of low_deg and high_deg together
                for i in range_of_low_deg:
                    for j in range_of_high_deg:
                        if (j, i) not in has_been_computed and (i, j) not in has_been_computed:
                            col_to_add = data[:, i] * data[:, j]
                            col_to_add = torch.reshape(col_to_add, (col_to_add.shape[0], 1))
                            data = torch.hstack((data, col_to_add))
                            has_been_computed.add((i, j))
                low_deg += 1
                high_deg -= 1
            end_of_range_of_deg_preceding_deg_i = pow_to_range[deg_i - 1].stop
            num_features = data.shape[1]
            pow_to_range[deg_i] = range(end_of_range_of_deg_preceding_deg_i, num_features)
    return data

def transform_features_batch_processing(data, poly_deg=2, batch_size=30):
    pow_to_range = dict()  # stores the ranges at which linear, quadratic, cubic terms are present...
    pow_to_range[1] = range(0, data.shape[1])
    has_been_computed = set()
    if poly_deg >= 2:
        # by the end of every iteration of the outer loop, (transformed) features of deg_i are added.
        for deg_i in range(2, poly_deg + 1):
            low_deg = 1
            high_deg = deg_i - low_deg
            while low_deg <= high_deg:
                range_of_low_deg = pow_to_range[low_deg]
                range_of_high_deg = pow_to_range[high_deg]
                # double loop combines 2 single features of low_deg and high_deg together
                for i in range_of_low_deg:
                    for j in range_of_high_deg:
                        if (j, i) not in has_been_computed and (i, j) not in has_been_computed:
                            col_to_add = data[:, i] * data[:, j]
                            col_to_add = torch.reshape(col_to_add, (col_to_add.shape[0], 1))
                            data = torch.hstack((data, col_to_add))
                            has_been_computed.add((i, j))
                low_deg += 1
                high_deg -= 1
            end_of_range_of_deg_preceding_deg_i = pow_to_range[deg_i - 1].stop
            num_features = data.shape[1]
            pow_to_range[deg_i] = range(end_of_range_of_deg_preceding_deg_i, num_features)
    return data

data = np.load("../ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.06.npy")
data = torch.from_numpy(data)

def get_new_quadratic_features(data_1, data_2):
    ''''
    data_1 and data_2 are torch tensors, both of which have the exact same size.

    Returns the new quadratic features.

    Suitable for large data sets.
    '''
    num_data_pts = data_1.shape[0]
    num_features_1 = data_1.shape[1]
    num_features_2 = data_2.shape[1]
    num_new_features = num_features_1 * num_features_2
    collected_features = torch.zeros((num_data_pts, num_new_features))
    index = 0
    for i in range(num_features_1):
        for j in range(num_features_2):
            col_to_add = data_1[:, i] * data_2[:, j]
            collected_features[:, index] = col_to_add
            index += 1
    return collected_features

def train_model(epochs, train_data, has_bias=True, poly_deg=1, lr=0.001, gradient_descent_type='SGD'):
    """""""""
    Data must have the label col. Uses sgd
    """
    actual_label = get_label(train_data)
    num_features = train_data.shape[1]

    # check if data size is too big for transformation to be performed. If it is too big, throw an error.
    try:
        if num_features > large_feature_num and poly_deg > 1:
            raise LargeDataSizeError
    except LargeDataSizeError:
        print("If you want to continue using the same dataset, set poly_deg=1.")

    # transform the features
    data_wo_bias_and_label = get_data_without_bias_and_label(train_data, has_bias=has_bias, has_label=True)
    transformed_data = transform_features(data_wo_bias_and_label, poly_deg=poly_deg)

    # add back bias col
    transformed_data = add_bias(transformed_data)

    # initialise the weights
    num_features = transformed_data.shape[1]
    weights = torch.zeros(num_features)

    np.random.seed(32)
    # train the model
    if gradient_descent_type == 'standard':
        #todo
        weights = 0
    elif gradient_descent_type == 'BGD':
        #todo
        weights = 0
    else:
        weights = get_weights_stochastic_gradient_descent(epochs, transformed_data, actual_label, weights, lr)

    return weights

def get_weights_stochastic_gradient_descent(epochs, data_wo_label, label, weights, lr):
    num_data_pts = data_wo_label.shape[0]
    for i in range(epochs):
        random_index = np.random.choice(num_data_pts, size=1, replace=False)[0]
        data_pt = data_wo_label[random_index]
        h_w_of_data_pt = h_w(weights, data_pt)
        actual_label_of_data_pt = label[random_index]
        partial_derivative_of_loss_wrt_weights = data_pt * (h_w_of_data_pt - actual_label_of_data_pt)
        weights = weights - lr * partial_derivative_of_loss_wrt_weights
        if i == epochs - 1:
            print(weights)
    return weights

def get_weights_batch_gradient_descent(epochs, data_wo_label, label, weights, lr):
    #todo
    return 0

def compute_loss():
    #todo: can use to print loss at the end of training
    return 0

def h_w(weights, x):
    x = weights @ torch.t(x)
    x = 1 / (1 + np.exp(-x))
    return x

"""""""""
start = time.time()
train_quadratic_model(1, data)
end = time.time()
print("Process ended in ", end - start, "s") # 90s
"""

def add_bias(data):
    num_data_pts = data.shape[0]
    bias_col = torch.ones((num_data_pts, 1))
    data = torch.hstack((bias_col, data))
    return data

def predict(test_data, weights, has_label=True, has_bias=True, probability_threshold=0.5, poly_deg=1):
    # todo: throw an error if data size is too big and if poly_deg = 2

    # todo: also add in a way for poly_deg=2 to be able to call the relevant method with batch processing
    test_data = get_data_without_bias_and_label(test_data, has_bias=has_bias, has_label=has_label)
    test_data = transform_features(test_data, poly_deg=poly_deg)
    test_data = add_bias(test_data)
    pred = h_w(weights, test_data)
    pred = torch.where(pred > probability_threshold, 1, 0)
    return pred

def predict_with_saved_model(data_pt):
    #todo
    return 0