import numpy as np
import torch

# do SVM with kernel trick, to implement from scratch

data = np.load("../ProcessedData/PvNormalData.npy")
data = torch.from_numpy(data) # Use data as tensor
print(data.shape)

# data shape is expected to be (number of observations, number of features) = (400, 1936)
def SVM(data):
    return 0