import numpy as np
import torch

# do SVM with kernel trick, to implement from scratch

data = np.load("./ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data) # Use data as tensor
print(data.shape)

# data shape is expected to be (number of observations, number of features) = (400, 1936),
# provided that sample size was not chosen
def SVM(data, gamma):
    '''''''''
    Input:
    - Data is normalised and has shape (number of observations, number of features)
    - gamma = 1 / (2 * standard_deviation ^ 2), is a hyperparameter
    '''
    num_data_points = data.shape[0]
    alpha = torch.zeros(num_data_points) # row vector
    # Kernel trick (Gaussian)
    # Represent as a matrix where M[i, j] = K(x_i, x_j). M will have the shape: (num_data_points, num_data_points)
    K = torch.zeros((num_data_points, num_data_points))
    computed = torch.zeros((num_data_points, num_data_points))
    # If computed[i, j] = 0, result has not been computed yet. Else, has been computed.
    memo = torch.zeros((num_data_points, num_data_points))
    for i in range(num_data_points):
        for j in range(num_data_points):
            pt_1 = data[i, :]
            pt_2 = data[j, :]
            if computed[i, j] == 1:
                K[i, j] = memo[i, j]
            elif computed[j, i] == 1:
                K[i, j] = memo[j, i]
            else:
                res = kernel_trick(pt_1, pt_2, gamma)
                K[i, j] = res
                memo[i, j] = res
                computed[i, j] = 1
    # solve the problem using the dual objective function
    #todo tmrw

    return 0

def kernel_trick(pt_1, pt_2, gamma):
    a = torch.linalg.norm(pt_1 - pt_2)
    p = - (a * a) + gamma
    return torch.exp(p)