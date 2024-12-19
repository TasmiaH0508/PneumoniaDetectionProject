import numpy as np
import torch
import cvxopt

data = np.load("./ProcessedData/PvNormalDataNormalised.npy")
data = torch.from_numpy(data) # Use data as tensor
print(data.shape)

# data shape is expected to be (number of observations, number of features) = (400, 1936),
# provided that sample size was not chosen

#todo: SVM with linear, hingeloss and Gaussian
def SVMNonLinear(data, gamma=0.1):
    '''''''''
    Input:
    - Data is normalised and has shape (number of observations, number of features)
    - gamma = 1 / (2 * standard_deviation ^ 2), is a hyperparameter
    
    Returns 
    '''
    num_data_points = data.shape[0]
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
    # Solve quadratic problem
    labels = data[:, -1]
    C = 400
    P = cvxopt.matrix(torch.outer(labels, labels) * K) # equivalent to K.y^Ty, is the quadratic term
    q = cvxopt.matrix(-torch.ones(num_data_points)) # linear term
    G = cvxopt.matrix(torch.vstack((-torch.eye(num_data_points), torch.eye(num_data_points))))
    h = cvxopt.matrix(torch.hstack((torch.zeros(num_data_points), torch.ones(num_data_points) * C)))
    A = cvxopt.matrix(labels.reshape(1, -1).astype('double'))
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = torch.ravel(solution['x'])
    print("Alpha is:", alpha)
    print("Alpha has the shape:", alpha.shape)

    # find support vectors
    threshold = 1e-5
    indices_for_support_vectors = torch.where(alpha > threshold)[0]
    print("The indices for the support vectors are:", indices_for_support_vectors)
    support_vectors = data[indices_for_support_vectors]
    support_alpha = alpha[indices_for_support_vectors]
    support_labels = labels[indices_for_support_vectors]

    # compute b
    b = support_labels[0] - torch.sum(support_alpha * support_labels * K[indices_for_support_vectors[0], indices_for_support_vectors])
    #todo: debug and testing, with scikit learning

    return support_vectors, support_alpha, support_labels, b

def kernel_trick(pt_1, pt_2, gamma):
    a = torch.linalg.norm(pt_1 - pt_2)
    p = - (a * a) + gamma
    return torch.exp(p)

def predict(data):
    #todo
    return 0