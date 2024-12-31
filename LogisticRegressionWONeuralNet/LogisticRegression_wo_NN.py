import torch
from PrepareData import *
import numpy as np
import time

class LargeDataSizeError(Exception):
    pass

large_feature_num = 2400 # this number can be adjusted to be a higher number

def transform_features(data, poly_deg=2):
    ''''
    Takes data(torch tensor), which does not have the bias or label cols and transforms the features
    by combining features.

    Suitable for small datasets.

    For e.g., [u, v] -> [u, v, u^2, u.v, v^2], for poly_deg=2
    '''
    # code kept for testing purposes
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

def transform_features_with_batch_processing(data, poly_deg=2, batch_size=600):
    pow_to_range = dict()  # stores the ranges at which linear, quadratic, cubic terms are present...
    pow_to_range[1] = range(0, data.shape[1])
    if poly_deg >= 2:
        # by the end of every iteration of the outer loop, (transformed) features of deg_i are added.
        for deg_i in range(2, poly_deg + 1):
            low_deg = 1
            high_deg = deg_i - low_deg
            while low_deg <= high_deg:
                range_of_low_deg = pow_to_range[low_deg]
                range_of_high_deg = pow_to_range[high_deg]
                # combines 2 single features of low_deg and high_deg together
                for i in range_of_low_deg:
                    col_to_multiply = data[:, i]
                    col_to_multiply = torch.reshape(col_to_multiply, (col_to_multiply.shape[0], 1))
                    cols_needed = torch.arange(range_of_high_deg.start, range_of_high_deg.stop)
                    mat = data[:, cols_needed] # mat may be very large
                    num_cols = mat.shape[1]
                    cols_to_add = None
                    # this loop applies batch processing to mat * col_to_multiply, as mat may be very large
                    for j in range(0, num_cols, batch_size):
                        batch_end = min(j + batch_size, num_cols)
                        mat_segment = mat[:, j : batch_end]
                        segment_to_add = mat_segment * col_to_multiply
                        if cols_to_add is None:
                            cols_to_add = segment_to_add
                        else:
                            cols_to_add = torch.hstack((cols_to_add, segment_to_add))
                    print("Taking feature: " + str(i))
                    print(cols_to_add)
                    if i == 1 and low_deg == high_deg == 1:
                        # todo: debug; this needs debugging since there is a double counting of features
                        cols_to_add = cols_to_add[:, 1]
                        cols_to_add = torch.reshape(cols_to_add, (cols_to_add.shape[0], 1))
                    data = torch.hstack((data, cols_to_add))
                low_deg += 1
                high_deg -= 1
                print("End of one iteration of while loop")
            end_of_range_of_deg_preceding_deg_i = pow_to_range[deg_i - 1].stop
            num_features = data.shape[1]
            pow_to_range[deg_i] = range(end_of_range_of_deg_preceding_deg_i, num_features)
    return data

def train_model(epochs, train_data, has_bias=True, poly_deg=1, lr=0.001, gradient_descent_type='SGD', batch_size=200):
    """""""""
    Data must have the label col. 
    """
    actual_label = get_label(train_data)
    num_features = train_data.shape[1]

    # check if data size is too big for transformation to be performed. If it is too big, throw an error.
    try:
        if num_features > large_feature_num and poly_deg > 1:
            raise LargeDataSizeError
    except LargeDataSizeError:
        print("If you want to continue using the same dataset, set poly_deg=1 or some other smaller number.")

    # transform the features
    data_wo_bias_and_label = get_data_without_bias_and_label(train_data, has_bias=has_bias, has_label=True)
    transformed_data = transform_features_with_batch_processing(data_wo_bias_and_label, poly_deg=poly_deg, batch_size=batch_size)

    # add back bias col
    transformed_data = add_bias(transformed_data)

    # initialise the weights
    num_features = transformed_data.shape[1]
    weights = torch.zeros(num_features)

    np.random.seed(32)
    # train the model
    if gradient_descent_type == 'BGD':
        #todo
        weights = 0
    elif gradient_descent_type == 'mBGD':
        #todo
        weights = 0
    else:
        weights = get_weights_stochastic_gradient_descent(epochs, transformed_data, actual_label, weights, lr)

    return weights

def get_weights_stochastic_gradient_descent(epochs, data_wo_label, label, weights, lr):
    ''''

    '''
    num_data_pts = data_wo_label.shape[0]
    for i in range(epochs):
        random_index = np.random.choice(num_data_pts, size=1, replace=False)[0]
        data_pt = data_wo_label[random_index]
        h_w_of_data_pt = h_w(weights, data_pt)
        actual_label_of_data_pt = label[random_index]
        partial_derivative_of_loss_wrt_weights = data_pt * (h_w_of_data_pt - actual_label_of_data_pt)
        weights = weights - lr * partial_derivative_of_loss_wrt_weights
    return weights

def get_weights_batch_gradient_descent(epochs, data_wo_label, label, weights, lr):

    return 0

def compute_loss():
    #todo: can use to print loss at the end of training
    return 0

def h_w(weights, x):
    x = weights @ torch.t(x)
    x = 1 / (1 + np.exp(-x))
    return x

def add_bias(data):
    num_data_pts = data.shape[0]
    bias_col = torch.ones((num_data_pts, 1))
    data = torch.hstack((bias_col, data))
    return data

def predict(test_data, weights, has_label=True, has_bias=True, probability_threshold=0.5, poly_deg=1):
    test_data = get_data_without_bias_and_label(test_data, has_bias=has_bias, has_label=has_label)
    test_data = transform_features(test_data, poly_deg=poly_deg)
    test_data = add_bias(test_data)
    pred = h_w(weights, test_data)
    pred = torch.where(pred > probability_threshold, 1, 0)
    return pred

def predict_with_saved_model(data_pt):
    #todo
    return 0