import torch
from PrepareData import *

def transform_features(data, poly_deg=2):
    pow_to_range = dict() # stores the ranges at which linear, quadratic, cubic terms are present...
    pow_to_range[1] = range(0, data.shape[1])
    has_been_computed = set()
    if poly_deg >= 2:
        # by the end of every iteration of the outer loop, features of deg_i are added.
        for deg_i in range(2, poly_deg + 1):
            low_deg = 1
            high_deg = deg_i - low_deg
            while low_deg <= high_deg:
                range_of_low_deg = pow_to_range[low_deg]
                range_of_high_deg = pow_to_range[high_deg]
                # inner loop combines 2 features of low_deg and high_deg together
                for i in range_of_low_deg:
                    for j in range_of_high_deg:
                        if (j, i) not in has_been_computed and (i, j) not in has_been_computed:
                            # print("Features " + str(i) + " and " + str(j) + " are being transformed.")
                            col_to_add = data[:, i] * data[:, j]
                            col_to_add = torch.reshape(col_to_add, (col_to_add.shape[0], 1))
                            data = torch.hstack((data, col_to_add))
                            has_been_computed.add((i, j))
                low_deg += 1
                high_deg -= 1
            end_of_range_of_deg_preceding_deg_i = pow_to_range[deg_i - 1].stop
            num_features = data.shape[1]
            pow_to_range[deg_i] = range(end_of_range_of_deg_preceding_deg_i, num_features)
            # print("The range of terms of degree " + str(deg_i) + " is :", (pow_to_range[deg_i].start, pow_to_range[deg_i].stop - 1))
    return data

def train_model(epochs, train_data, poly_deg=2, has_bias=True, has_label=True, gradient_descent_type='sgd', regularization=True):
    #todo
    # remove label and bias col
    data_wo_bias_and_label = get_data_without_bias_and_label(train_data, has_bias=has_bias, has_label=has_label)
    return 0

def predict(test_data):
    #todo
    return 0

def predict_with_saved_model(data_pt):
    #todo
    return 0