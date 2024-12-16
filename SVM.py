import numpy as np
import torch
import cvxopt

# do SVM with kernel trick, to implement from scratch

data = np.load("./ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data) # Use data as tensor
print(data.shape)

# data shape is expected to be (number of observations, number of features) = (400, 1936),
# provided that sample size was not chosen
def SVM(data, gamma=0.1):
    '''''''''
    Input:
    - Data is normalised and has shape (number of observations, number of features)
    - gamma = 1 / (2 * standard_deviation ^ 2), is a hyperparameter
    '''
    num_data_points = data.shape[0]
    # Kernel trick (Gaussian)
    # Represent as a matrix where K[i, j] = K(x_i, x_j)
    K = torch.zeros((num_data_points, num_data_points))
    computed = torch.zeros((num_data_points, num_data_points))
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
    # Since SVM is being performed wo regularisation, set C to be a very large value
    labels = data[:, -1]
    # todo, with package CVXOPT
    return 0

def kernel_trick(pt_1, pt_2, gamma):
    a = torch.linalg.norm(pt_1 - pt_2)
    p = - (a * a) + gamma
    return torch.exp(p)