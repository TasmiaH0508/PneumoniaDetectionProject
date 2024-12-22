from sklearn.svm import SVC
import torch
from ComputeMetrics import *

def get_predictions_svm_method(train_data, test_data, kernel_type='linear'):
    '''''''''
    Takes 2 torch tensors: train_data and test_data, each of which have 
    '''
    clf = SVC(kernel=kernel_type, C=1.0)

    # Prepare data for training
    train_data_wo_bias_and_label = get_data_without_bias_and_label(train_data)
    training_labels = get_label(train_data)
    training_labels = replace_0s_with_neg_1(training_labels)

    # Train
    clf.fit(train_data_wo_bias_and_label, training_labels)

    # Prepare data for testing
    test_data_wo_bias_and_label = get_data_without_bias_and_label(test_data)

    # Get predictions
    y_pred = clf.predict(test_data_wo_bias_and_label)
    y_pred = torch.from_numpy(y_pred)
    y_pred = replace_neg_1s_with_0(y_pred)

    return y_pred

def replace_neg_1s_with_0(pred):
    pred = torch.where(pred == -1, 0, pred)
    return pred

def get_label(data):
    '''''''''
    Takes out the last column, which is the label col, from the data(torch tensor)
    '''
    num_features = data.shape[1] - 1
    label = data[:, num_features]
    return label

def get_data_without_bias_and_label(data):
    '''''''''
    Removes the labels and bias cols from data(torch tensor)
    '''
    num_features = data.shape[1] - 1
    data = data[:, 1 : num_features]
    return data

def replace_0s_with_neg_1(label):
    '''''''''
    Takes the label col and replaces 0s with -1
    '''
    label = torch.where(label == 0, -1, label)
    return label

def main():
    return 0