import torch
import torch.nn as nn

# This neural net can be represented as y = mx + c

train_arr = torch.asarray([[1, 3, 0],
                           [1, 7, 1]], dtype=torch.float32) # rows are observations.
# This is the type of data torch.nn works with

test_arr = torch.asarray([[1, 10, 1]], dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(1, 1, bias=False)
        self.l2 = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x