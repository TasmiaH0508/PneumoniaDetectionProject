import torch
from torch import nn
import numpy as np
import time
from PrepareData import *
from ComputeMetrics import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, num_input_features=1000):
        super().__init__()
        self.num_input_features = num_input_features
        self.L1 = nn.Linear(num_input_features, 968)
        self.L2 = nn.LeakyReLU(negative_slope=0.2)
        self.L3 = nn.Linear(968, 242)
        self.L4 = nn.LeakyReLU(negative_slope=0.2)
        self.L5 = nn.Linear(242, 24)
        self.L6 = nn.ReLU()
        self.L7 = nn.Linear(24, 1)
        self.L8 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.L7(x)
        x = self.L8(x)
        return x

def train_model(model, epochs, train_data, optimiser, bias_present=True, use_old_weights=False, save_weights=True,
                file_name_to_read_from="./torch_weights.pth", file_name_to_write_to="./torch_weights.pth"):
    ''''
    Trains model.

    The train_data param is a tensor that must also include the label.
    '''
    # prepare data
    train_labels = get_label(train_data)
    train_labels = torch.reshape(train_labels, (train_labels.shape[0],1))
    train_data = get_data_without_bias_and_label(train_data, has_bias=bias_present)

    if use_old_weights:
        try:
            state_dict = torch.load(file_name_to_read_from, weights_only=True)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found, old weights cannot be used.")

    # train
    loss = nn.BCELoss()
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model.forward(train_data)
        loss_val = loss(y_pred, train_labels)
        loss_val.backward()
        optimiser.step()
        if i == epochs - 1:
            print("The loss is now:", loss_val.item())

    if save_weights:
        weights = model.state_dict()
        torch.save(weights, file_name_to_write_to)  # overwrites old weights with new weights

def predict(model, threshold_prob, test_data, bias_present=True, has_label=True):
    ''''
    Returns the prediction.

    The test_data param is a tensor that may or may not include the label or bias.
    If bias is present, set bias_present=True
    If label is present, set has_label=True
    '''
    test_data = get_data_without_bias_and_label(test_data, has_bias=bias_present, has_label=has_label)
    pred = model.forward(test_data)
    pred = torch.where(pred >= threshold_prob, 1, 0)
    return pred

def predict_with_saved_weights(test_data, threshold=0.65, has_bias=False, has_label=False, file_to_read_from="./torch_weights_var_0.02.pth"):
    ''''
    Used to make predictions in ../Analysis/PredictPneumonia.py
    '''
    if has_bias:
        num_input_features = test_data.shape[1] - 1
    else:
        num_input_features = test_data.shape[1]
    if has_label:
        num_input_features = num_input_features - 1
    try:
        model = NeuralNetwork(num_input_features=num_input_features)
        weights = torch.load(file_to_read_from, weights_only=True)
        model.load_state_dict(weights)
        test_data = get_data_without_bias_and_label(test_data, has_bias=has_bias, has_label=has_label)
        pred = model.forward(test_data)
        pred = torch.where(pred >= threshold, 1, 0)
        return pred
    except FileNotFoundError:
        print("File not found, weights cannot be used.")

def main():
    start = time.time()
    train_data = np.load("../ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.02.npy")
    train_data = torch.from_numpy(train_data)
    test_data = np.load("../ProcessedRawData/TestSet/PvNormalDataNormalised_var0.02.npy")
    test_data = torch.from_numpy(test_data)

    num_features_wo_bias = train_data.shape[1] - 2

    model = NeuralNetwork(num_input_features=num_features_wo_bias)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, 350, train_data, optimiser, save_weights=True)
    print("Model has been trained")

    predictions = predict(model, 0.65, test_data)
    actual_test_labels = get_label(test_data)
    print("The accuracy of the model is:", get_accuracy(actual_test_labels, predictions))

    print("The recall of this model is:", get_recall(actual_test_labels, predictions))

    end = time.time()
    print("Time taken in seconds:", end - start)