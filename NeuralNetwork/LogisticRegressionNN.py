import torch
from torch import nn
import numpy as np
from ComputeMetrics import get_accuracy

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

# Process data first

data = np.load("../ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data) # has shape (400, 1936), where the last col is the label
data_without_label = data[:, 0: 1935]
label = data[:, 1935]
number_of_data_points = label.shape[0]
label = torch.reshape(label, (label.shape[0], 1))

model = NeuralNetwork()

def train_model(epochs, data_without_label, optimiser, save_weights=True, use_old_weights=False):
    '''''''''
    Trains neural network with given parameters.
    
    If save_weights=False, the weights will not be saved to an external file and the weights generated cannot be 
    used in future to train model further.
    
    If use_old_weights=False, the model will not use the weights previously generated for learning
    '''
    if use_old_weights:
        state_dict = torch.load("./torch_weights.pth", weights_only=True)
        model.load_state_dict(state_dict)

    loss_fn = nn.BCELoss()
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model(data_without_label)
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimiser.step()

    if save_weights:
        weights = model.state_dict()
        torch.save(weights, "./torch_weights.pth")

def predict(data_without_label, threshold_probability):
    model.load_state_dict(torch.load("./torch_weights.pth", weights_only=True))
    model.eval()
    predictions = model(data_without_label)
    print(predictions)
    return 0

optimiser = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0)
#train_model(50, data_without_label, optimiser, save_weights=True, use_old_weights=False)
predict(data_without_label, 0.5)