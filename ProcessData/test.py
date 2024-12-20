import numpy as np
import torch
from ProcessData import pca, normalise_data_min_max

arr_1 = np.array([[1, 1, 10, 25],
                     [1, 11, 11, 25],
                     [1, 2, 12, 25]], dtype=float)  # the rows are the observations

arr_2 = np.array([[1, 1, 10, 25],
                        [1, 11, 11, 25],
                        [1, 2, 12, 24]], dtype=float) # the rows are the observations

arr_3 = np.array([[1, 25, 10, 1],
                        [1, 25, 11, 11],
                        [1, 25, 12, 2]], dtype=float) # the rows are the observations

data_arrays = [arr_1, arr_2, arr_3]

def test_normalise_data_min_max():
    test_res_1 =
    for i in range(len(data_arrays)):
        #todo

def test_PCA():
    for i in range(len(data_arrays)):
        data = data_arrays[i]
        #todo

num_features = data.shape[0]
num_data_pts = data.shape[1]

data = torch.from_numpy(data)
data = normalise_data_min_max(data)

data = data.T

mean_matrix = torch.mean(data, dim=1)
mean_matrix = torch.reshape(mean_matrix, (mean_matrix.shape[0], 1))

mean_centred_data = data - mean_matrix

cov_matrix = (mean_centred_data @ mean_centred_data.T) / num_features

U, S, V = torch.linalg.svd(cov_matrix)

x = torch.cumsum(S, dim=0)
explained_variance_ratio = torch.cumsum(S, dim=0) / torch.sum(S)

r = torch.searchsorted(explained_variance_ratio, 0.99) + 1

U_reduced = U[:, :r]

Z = U_reduced.T @ data

reconstructed_data = (U_reduced @ Z).T
print("This is the expected PCA result:\n", reconstructed_data)

variance_of_reconstructed_data = torch.var(U_reduced, dim=1)

data = np.array([[1, 1, 10, 25],
                        [1, 11, 11, 25],
                        [1, 2, 12, 25]], dtype=float) # the rows are the observations

data = torch.from_numpy(data)

data = normalise_data_min_max(data)
#pca_result = pca_with_batch_processing(data, 1)
pca_result = pca(data)
#print("This is the result of the function:\n", pca_result)

# test ideology/process_data
data = np.array([[1, 1, 10, 25],
                        [1, 11, 11, 25],
                        [1, 2, 12, 25]], dtype=float) # the rows are the observations