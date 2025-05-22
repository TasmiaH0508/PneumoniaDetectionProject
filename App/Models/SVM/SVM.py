import numpy as np
import torch
from sklearn import svm

from App.ComputeMetrics import get_accuracy, get_recall, get_precision
from App.PrepareData import get_data_without_bias_and_label, get_label


def train_model(train_data, kernel='rbf', degree=2, gamma=0.5):
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
        clf = svm.SVC(kernel=kernel, gamma=gamma)
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

train_set = np.load("../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.02.npy")
train_set = torch.from_numpy(train_set)
model = train_model(train_set, kernel='rbf', gamma=1/1000)

test_set = np.load("../Data/ProcessedRawData/TestSet/PvNormalDataNormalised_var0.02.npy")
test_set = torch.from_numpy(test_set)
actual_labels = get_label(test_set)
pred = predict_with_input_model(model, test_set)

accuracy = get_accuracy(actual_labels, pred)
print("The accuracy in % is ", accuracy)
recall = get_recall(actual_labels, pred)
print("The recall is ", recall)
precision = get_precision(actual_labels, pred)
print("The precision is ", precision)

"""
folder = "../Data/TestImages"
processed_images = process_images(folder)
processed_images = torch.from_numpy(processed_images)
bias = torch.ones((processed_images.shape[0], 1))
processed_images = torch.hstack((bias, processed_images))

pred = predict_with_input_model(model, processed_images, has_label=False)
print(pred)
"""