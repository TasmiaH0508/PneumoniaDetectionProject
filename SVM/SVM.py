from sklearn.svm import SVC
import torch
from ComputeMetrics import *
import time
import numpy as np

def get_predictions_svm_method(train_data, test_data, kernel_type='linear'):
    '''''''''
    Takes 2 torch tensors: train_data and test_data and kernel choice
    '''
    clf = SVC(kernel=kernel_type, C=1.0)

    # prepare data for training
    train_data_wo_bias_and_label = get_data_without_bias_and_label(train_data)
    training_labels = get_label(train_data)
    training_labels = replace_0s_with_neg_1(training_labels)
    print("Prepared training data.")

    # train
    clf.fit(train_data_wo_bias_and_label, training_labels)
    print("Trained model.")

    # prepare data for testing
    test_data_wo_bias_and_label = get_data_without_bias_and_label(test_data)
    print("Prepared test data.")

    # get predictions
    y_pred = clf.predict(test_data_wo_bias_and_label)
    y_pred = torch.from_numpy(y_pred)
    y_pred = replace_neg_1s_with_0(y_pred)
    print("Getting predictions.")

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
    # work with data with 0.035 variance and linear kernel
    start = time.time()

    # the labels are 1 if pneumonia is present and 0 otherwise
    test_data = np.load("../ProcessedData/TestSet/PvNormalDataNormalised_var0.035.npy")
    train_data = np.load("../ProcessedData/TrainingSet/PvNormalDataNormalised_var0.035.npy")

    test_data = torch.from_numpy(test_data)
    train_data = torch.from_numpy(train_data)

    y_pred_linear_kernel = get_predictions_svm_method(train_data, test_data)
    actual_labels = get_label(test_data)
    acc = get_accuracy(actual_labels, y_pred_linear_kernel)

    end = time.time()

    print("The accuracy(in %) is:", acc)

    print("Time taken:", end - start)

main()

"""""""""
0.04 var, 17246 features -> 92.75% accuracy
0.035 var, 26112 features -> 93.0% accuracy
0.03 var, 37397 features -> 92.75% accuracy
0.025 var, 47862 features -> 93.75% accuracy
0.02 var, 54983 features -> 92% accuracy
"""""