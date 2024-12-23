import torch
from torch import nn
import numpy as np
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
        self.L6 = nn.LeakyReLU(negative_slope=0.2)
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

def train_model(model, epochs, train_data, optimiser, bias_present=True, use_old_weights=False, save_weights=True):
    '''''''''
    
    '''
    # prepare data
    train_labels = get_label(train_data)
    train_data = get_data_without_bias_and_label(train_data, has_bias=bias_present)
    num_features = train_data.shape[1]

    model.num_input_features = num_features

    if use_old_weights:
        try:
            state_dict = torch.load("torch_weights.pth", weights_only=True)
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

    if save_weights:
        weights = model.state_dict()
        torch.save(weights, "./torch_weights.pth")  # overwrites old weights with new weights

def predict(model, threshold_prob, test_data, bias_present=True):
    '''''''''
    Weights need to be saved before predicting
    '''''
    try:
        state_dict = torch.load("torch_weights.pth", weights_only=True)
        model.load_state_dict(state_dict)
        test_data = get_data_without_bias_and_label(test_data, has_bias=bias_present)
        pred = model.forward(test_data)
        pred = torch.where(pred >= threshold_prob, 1, 0)
        return pred
    except FileNotFoundError:
        print("File not found, cannot predict.")

def main():
    #todo

#main()