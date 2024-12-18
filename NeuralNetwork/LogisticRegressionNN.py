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
    def __init__(self, input_size):
        # WARNING: input size may change with different runs of PCA
        super().__init__()
        self.L1 = nn.Linear(input_size, 968, bias=False)
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
data = torch.from_numpy(data)
data_without_label = data[:, 0: 1935]
label = data[:, 1935]
number_of_data_points = label.shape[0]
label = torch.reshape(label, (label.shape[0], 1))

def train_model(epochs, data_without_label, optimiser, save_weights=True, use_old_weights=False):
    '''''''''
    Trains neural network with given parameters.
    
    If save_weights=False, the weights will not be saved to an external file and the weights generated cannot be 
    used in future to train model further.
    
    If use_old_weights=False, the model will not use the weights previously generated for learning
    '''
    if use_old_weights:
        try:
            state_dict = torch.load("./torch_weights.pth", weights_only=True)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found, old weights cannot be used. Training new model...")

    loss_fn = nn.BCELoss()
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model(data_without_label) # compute the value of y_hat (not the same as the expected label)
        loss = loss_fn(y_pred, label) # compute loss
        # perform back propagation (computes partial derivatives of loss with respect to weights for optimisation):
        loss.backward()
        optimiser.step()

    if save_weights:
        weights = model.state_dict()
        torch.save(weights, "./torch_weights.pth") # overwrites old weights with new weights

def predict(data_without_label, threshold_probability):
    # need to split data into training and test set
    # todo: think about whether predict should be using test or training data
    y_pred = model(data_without_label)
    print(y_pred.shape)
    return 0

model = NeuralNetwork(data_without_label.shape[1])
optimiser = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0)
train_model(50, data_without_label, optimiser, use_old_weights=True)