import numpy as np
import torch
import cvxopt

# data shape is expected to be (number of observations, number of features) = (400, 1936)

#todo: SVM with linear, hingeloss and Gaussian
def SVM(data, kernel, gamma=0.1):
    # Replace the 0 labels with -1 labels
    data = replace_0_labels_with_negative_1(data)

    # Apply kernel trick to get K
    # if the kernel is linear, K has the shape (number of observations, 1)
    K = kernel(data)

    # Solve for alpha

    return 0

def replace_0_labels_with_negative_1(raw_data):
    '''''''''''
    Takes data(torch tensor) of shape (number of observations, number 0f features + 1), where the last column is the 
    label column.
    
    Returns data where the 0 labels are changed to -1.
    '''''
    num_features = raw_data.shape[1] - 1
    label = raw_data[:, num_features]
    label = torch.where(label == 0, -1, label)
    raw_data[:, num_features - 1] = label
    return raw_data

def Gaussian_kernel(data, gamma=0.1):
    #todo
    return 0

def linear_kernel(data, has_bias=True):
    '''''''''''
    Takes data(torch tensor) of shape (number of observations, number 0f features + 1), where the last column is the 
    label column.
    
    For each data point, linear_kernel(data_pt) = dot(data_pt, 1) 
    '''
    num_features = data.shape[1] - 1
    if has_bias:
        data = data[:, 1 : num_features] # bias has to be removed
    else:
        data = data[:, 0 : num_features]
    res = torch.sum(data, dim=1, keepdim=True) # has shape (number of observations, 1)
    return res

def compute_weights():
    #todo
    return 0

def solve_for_alpha():
    #todo
    return 0

def predict(data):
    #todo
    return 0