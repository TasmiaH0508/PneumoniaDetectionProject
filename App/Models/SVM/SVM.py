import os

import numpy as np
import torch
from PIL import Image
from sklearn import svm

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

def process_image(image_path):
    image = Image.open(image_path).convert('L')
    cropped_image = crop_and_centralise_image_into_square(image)
    padded_and_cropped_image = pad_image(cropped_image)
    print_image(padded_and_cropped_image)
    image_array = np.array(padded_and_cropped_image).flatten()
    # need the scaling
    image_array = image_array / 255
    image_array = np.reshape(image_array, (1, image_array.shape[0]))
    min_matrix = np.load("../Data/ProcessedRawData/MinData/min_across_all_features.npy")
    min_matrix = np.reshape(min_matrix, (image_array.shape[0], image_array.shape[1]))
    range_matrix = np.load("../Data/ProcessedRawData/RangeData/range_across_all_features.npy")
    range_matrix = np.reshape(range_matrix, (image_array.shape[0], image_array.shape[1]))
    normalised_arr = min_max_normalise_with_predefined_params(image_array, min_matrix, range_matrix)
    bias = np.ones((1, 1))
    normalised_arr = np.hstack((bias, normalised_arr))
    normalised_arr = torch.from_numpy(normalised_arr)
    return normalised_arr

def crop_and_centralise_image_into_square(image):
    width, height = image.size
    crop_size = min(width, height)
    if height > width:
        box = (0, 0, width, width)
    else:
        box = (0, 0, crop_size, crop_size)
    cropped_image = image.crop(box)
    return cropped_image

image_path = "../../../../../Users/tasmiahaque/Desktop/PneumoniaDetectionProject/App/Models/Data/TestImages/Pneumonia.png"
test_image = process_image(image_path)

train_data = np.load("../../../../../Users/tasmiahaque/Desktop/PneumoniaDetectionProject/App/Models/Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy")
train_data = torch.from_numpy(train_data)
trained_model = train_model(train_data, kernel='rbf')

print(predict_with_input_model(trained_model, test_image, has_label=False))