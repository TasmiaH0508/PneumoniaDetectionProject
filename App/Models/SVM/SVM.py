import os

import numpy as np
import torch
from sklearn import svm

from App.ComputeMetrics import get_accuracy
from App.PrepareData import get_data_without_bias_and_label, get_label
from App.ProcessRawData import pad_image, min_max_normalise_with_predefined_params


def train_model(train_data, kernel='rbf', degree=1, gamma=1):
    '''
    :param kernel: gaussian is the default kernel
    :param degree: for linear kernel, to choose degree of features
    :param gamma: for gaussian kernel
    :param train_data: torch tensor. Expected to have label
    :return: model
    '''
    train_data_wo_label = get_data_without_bias_and_label(train_data, has_bias=False)
    train_labels = get_label(train_data)
    if kernel == 'linear':
        clf = svm.SVC(degree=degree, kernel=kernel)
    else:
        clf = svm.SVC(kernel=kernel, gamma=gamma, class_weight='balanced')
    clf.fit(train_data_wo_label, train_labels)
    return clf

def predict_with_input_model(clf, test_data, has_label=True):
    '''
    :param clf: model
    :param test_data: torch tensor. Expected to have bias col.
    :return: predicted labels, as a torch tensor
    '''
    test_data_wo_bias_and_label = get_data_without_bias_and_label(test_data, has_label=has_label, has_bias=False)
    predicted = clf.predict(test_data_wo_bias_and_label)
    predicted = torch.from_numpy(predicted)
    return predicted

def process_image(image_path, is_within_same_file=False):
    padded_image = pad_image(image_path)
    image_arr = np.array(padded_image).flatten()
    image_arr = np.reshape(image_arr, (1, image_arr.shape[0]))
    if is_within_same_file:
        range_matrix = np.load("../Data/ProcessedRawData/RangeData/range_across_all_features.npy")
        min_matrix = np.load("../Data/ProcessedRawData/MinData/min_across_all_features.npy")
    else:
        range_matrix = np.load("App/Models/Data/ProcessedRawData/RangeData/range_across_all_features.npy")
        min_matrix = np.load("App/Models/Data/ProcessedRawData/MinData/min_across_all_features.npy")
    range_matrix = np.reshape(range_matrix, (1, range_matrix.shape[0]))
    min_matrix = np.reshape(min_matrix, (1, min_matrix.shape[0]))
    image_arr = min_max_normalise_with_predefined_params(image_arr, min_matrix, range_matrix)
    # no features were removed for the chosen model so feature mapping not needed
    bias = np.ones((1, 1))
    image_arr = np.hstack((bias, image_arr))
    image_arr = torch.from_numpy(image_arr)
    return image_arr

# why is the model returning the same predictions for everything???
train_data = np.load("../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy")
train_data = torch.from_numpy(train_data)
model = train_model(train_data, kernel='rbf')

test_data = train_data = np.load("../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy")
test_data = torch.from_numpy(test_data)
pred = predict_with_input_model(model, test_data, has_label=True)
actual_labels = get_label(test_data)
print(get_accuracy(actual_labels, pred))

test_folder = "../Data/TestImages"
test_arr = None
for file in os.listdir(test_folder):
    path = os.path.join(test_folder, file)
    if test_arr is None:
        test_arr = process_image(path, is_within_same_file=True)
    else:
        test_arr = np.vstack((test_arr, process_image(path, is_within_same_file=True)))
pred = predict_with_input_model(model, test_arr, has_label=False)
print(pred)