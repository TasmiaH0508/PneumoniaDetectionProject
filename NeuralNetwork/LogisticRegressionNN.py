import os
import torch
from torch import nn
import numpy as np

data = np.load("../ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(1935, 968, bias=False)
        self.L2 = nn.LeakyReLU(negative_slope=0.2)
        self.L3 = nn.Linear(968, 242)
        self.L4 = nn.LeakyReLU(negative_slope=0.2)
        self.L5 = nn.Linear(242, 96)
        self.L6 = nn.LeakyReLU(negative_slope=0.2)
        self.L7 = nn.Linear(96, 24)
        self.L8 = nn.LeakyReLU(negative_slope=0.2)
        self.L9 = nn.Linear(24, 1)
        self.L10 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.L7(x)
        x = self.L8(x)
        x = self.L9(x)
        x = self.L10(x)
        return x

#model = NeuralNetwork().to(device)
#print(model)

data = np.load("../ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data) # has shape (400, 1936), where the last col is the label
data_without_label = data[:, 0: 1935]
label = data[:, 1935]

def train_model(epochs, data, label):
    '''''''''
    Input:
    - data excludes the labels 
    - labels contains the 
    Returns weights of the trained model
    '''''''''''
    loss = float('inf')
    optimiser =
    loss_function = nn.BCELoss()
    for i in range(epochs):
        #todo tmrw