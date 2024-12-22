import numpy as np
import torch

from SVM import *

train_arr = np.array([[1, -2, -2, 0],
                [1, -2, 0, 0],
                [1, 0, 2, 1],
                [1, 1, 1, 1],
                [1, 3, 0, 1]]) # the rows are the observations

train_arr = torch.from_numpy(train_arr)

test_arr = np.array([[1, 3, 8, 1],
                     [1, -3, -8, 0],
                     [1, 4, 1000, 1]])

test_arr = torch.from_numpy(test_arr)

arr = np.load("../ProcessedData/TestSet/PvNormalDataNormalised_var0.035.npy")
print("Found")