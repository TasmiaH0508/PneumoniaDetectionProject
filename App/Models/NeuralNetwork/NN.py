import numpy as np
import torch
from torch import nn

from App.ComputeMetrics import get_accuracy, get_recall
from App.PrepareData import get_label, get_data_without_bias_and_label

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, num_input_features_wo_bias):
        super().__init__()
        self.num_input_features = num_input_features_wo_bias
        self.L1 = nn.Linear(num_input_features_wo_bias, 1)
        self.L2 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        return x

def train_model(model, epochs, train_data, lr=0.001, bias_present_for_training_set=True, save_model=False,
                use_old_weights=False, path_to_use="weights.pth"):
    if use_old_weights:
        weights = torch.load(path_to_use, weights_only=True)
        model.load_state_dict(weights)

    train_labels = get_label(train_data)
    train_labels = torch.reshape(train_labels, (train_labels.shape[0],1))
    train_set = get_data_without_bias_and_label(train_data, has_bias=bias_present_for_training_set)

    loss = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model.forward(train_set)
        loss_val = loss(y_pred, train_labels)
        if i % 100 == 0:
            print(loss_val.item())
        if i == epochs - 1:
            print("The loss is now:", loss_val.item())
            print("Epochs used:", i + 1)
        loss_val.backward()
        optimiser.step()

    if save_model:
        torch.save(model.state_dict(), "weights.pth")

    return model

def predict_with_saved_model(processed_image_arr):
    num_input_features = 65536
    model = NeuralNetwork(num_input_features)
    weights = torch.load("App/Models/NeuralNetwork/weights.pth", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    threshold_prob = 0.4
    input_arr = processed_image_arr.float()
    pred = predict(model, threshold_prob, input_arr, has_label=False)
    pred = pred[0].item()
    return pred

def predict(model, threshold_prob, data, bias_present=True, has_label=True):
    """

    :param model:
    :param threshold_prob:
    :param data:
    :param bias_present:
    :param has_label:
    :return:
    """
    data = get_data_without_bias_and_label(data, has_bias=bias_present, has_label=has_label)
    pred = model(data)
    pred = torch.where(pred >= threshold_prob, 1, 0)
    return pred