import numpy as np
import torch
from SVM import *

arr = np.array([[-2, -2, -1],
                [-2, 0, -1],
                [0, 2, 1],
                [1, 1, 1],
                [3, 0, 1]]) # the rows are the observations

lin_kernel_expected_res_alpha = np.array([[0],
                                          [0.25],
                                          [0.25],
                                          [0],
                                          [0]])

expected_weights_lin_kernel = np.array([0.5, 0.5])

arr = np.array([[-2, -2, 0],
                [-2, 0, 0],
                [0, 2, 1],
                [1, 1, 1],
                [3, 0, 1]])

arr = torch.from_numpy(arr).float()

print(linear_kernel(arr, has_bias=False))