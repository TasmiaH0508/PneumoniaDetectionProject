import joblib
import torch
from sklearn import svm

from PrepareData import get_data_without_bias_and_label, get_label

def train_model(train_data, kernel='rbf', degree=1, gamma=0.0004, save_model=False):
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

    if save_model:
        joblib.dump(clf, 'svm_model.joblib')
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

def predict_with_saved_model(processed_image_array, has_label=True):
    model = joblib.load('Models/SVM/svm_model.joblib')
    input = processed_image_array
    if has_label:
        input = get_data_without_bias_and_label(processed_image_array, has_bias=False)
    pred = model.predict(input)
    pred = torch.from_numpy(pred)
    return pred