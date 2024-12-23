import torch
import torch.nn as nn

train_arr = torch.asarray([[1, 10, 20, 30, 0],
                           [1, 15, 15, 90, 1],
                           [1, 20, 15, 120, 1]])

test_arr = torch.asarray([[1, 10, 20, 130, 1]])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(4, 2)
        self.L2 = nn.LeakyReLU(0.3)
        self.L3 = nn.Linear(2, 1)
        self.L4 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        return x

model = NeuralNetwork()


def train_model(epochs, data_without_label, label, optimiser, save_weights=True, use_old_weights=False):
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
        y_pred = model(data_without_label)  # compute the value of y_hat (not the same as the expected label)
        loss = loss_fn(y_pred, label)  # compute loss
        # perform back propagation (computes partial derivatives of loss with respect to weights for optimisation):
        loss.backward()
        optimiser.step()

    if save_weights:
        weights = model.state_dict()
        torch.save(weights, "./torch_weights.pth")  # overwrites old weights with new weights

