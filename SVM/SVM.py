import joblib
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import torch
from PrepareData import *
from ComputeMetrics import *
from joblib import dump
import numpy as np
import time

class SVMClassifier:
    def __init__(self, kernel_type='linear', regularisation_param=1.0):
        self.clf = SVC(kernel=kernel_type, C=regularisation_param)

def train_model(SVM_classifier, train_data, has_bias=True, save_model=False, file_to_save_to="svm_model.joblib"):
    ''''
    Takes a classifier instance and trains it on the training data.
    The training data must have the label col(which is taken to be the last col).
    '''
    clf = SVM_classifier.clf

    # prepare data for training
    train_data_wo_bias_and_label = get_data_without_bias_and_label(train_data, has_bias=has_bias)
    training_labels = get_label(train_data)
    training_labels = replace_0s_with_neg_1(training_labels)
    print("Prepared training data.")

    # train
    clf.fit(train_data_wo_bias_and_label, training_labels)
    print("Trained model.")

    if save_model:
        dump(clf, file_to_save_to)

def get_predictions(SVM_classifier, test_data, has_bias=True):
    ''''
    Takes a classifier instance and tests it on the test data.
    Data must have a label col(which is taken to be the last col).
    '''
    clf = SVM_classifier.clf
    test_data_wo_bias_and_label = get_data_without_bias_and_label(test_data, has_bias=has_bias)
    y_pred = clf.predict(test_data_wo_bias_and_label)
    y_pred = torch.from_numpy(y_pred)
    y_pred = replace_neg_1s_with_0(y_pred)
    return y_pred

def get_predictions_with_previously_loaded_model(test_data, has_bias=True, has_label=False, file_to_read_from="svm_model.joblib"):
    ''''
    '''
    loaded_model = joblib.load(file_to_read_from)
    test_data = get_data_without_bias_and_label(test_data, has_bias=has_bias, has_label=has_label)
    y_pred = loaded_model.predict(test_data)
    y_pred = torch.from_numpy(y_pred)
    y_pred = replace_neg_1s_with_0(y_pred)
    return y_pred

def replace_neg_1s_with_0(pred):
    ''''
    Takes the predictions(torch tensor) from the SVM classifier and replaces -1 with 0 to match
    conventional labels: 0 and 1.
    '''
    pred = torch.where(pred == -1, 0, pred)
    return pred

def replace_0s_with_neg_1(label):
    ''''
    Takes the label col(torch tensor) and replaces 0s with -1
    '''
    label = torch.where(label == 0, -1, label)
    return label

def main():
    start = time.time()
    model = SVMClassifier(kernel_type='rbf')

    train_data = np.load("../ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.04.npy")
    print("The shape of train_data is", train_data.shape)
    train_data = torch.from_numpy(train_data)
    train_model(model, train_data, save_model=False)

    test_data = np.load("../ProcessedRawData/TestSet/PvNormalDataNormalised_var0.04.npy")
    print("The shape of test_data is", test_data.shape)
    test_data = torch.from_numpy(test_data)

    pred = get_predictions(model, test_data)
    actual_labels = get_label(test_data)
    print("The accuracy in % is:", get_accuracy(actual_labels, pred))

    recall = get_recall(actual_labels, pred)
    print("The recall is:", recall)

    conf_matrix = confusion_matrix(actual_labels, pred)
    print("The confusion matrix is:\n", conf_matrix)

    end = time.time()
    print("Process took", end - start, "seconds.")