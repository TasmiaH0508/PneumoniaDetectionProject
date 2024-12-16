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