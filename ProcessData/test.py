import numpy as np
import torch
from ProcessData import *

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

data = data_arrays[2]
print("This is the starting data:\n", data)

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

data = data_arrays[2]

data = torch.from_numpy(data)

data = normalise_data_min_max(data)

pca_result = pca(data)

features_with_sufficient_var = get_features_with_sufficient_var(pca_result)
print('Features with sufficient var:', features_with_sufficient_var)

print(pick_observations_and_features(pca_result, None, features_with_sufficient_var))

arr = np.array([1, 2])
arr = torch.from_numpy(arr)
arr = arr.numpy()
np.save("../ProcessedData/TestSet/test", arr)

#arr = np.load("../ProcessedData/TestSet/test.npy")
print(arr)