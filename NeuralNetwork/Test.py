import torch
import torch.nn as nn

# simulates test scores and whether it is a pass or a fail
train_data = torch.asarray([[1, 20, 0],
                               [1, 40, 0],
                               [1, 60, 1],
                               [1, 70, 1],
                               [1, 80, 1]], dtype=torch.float)