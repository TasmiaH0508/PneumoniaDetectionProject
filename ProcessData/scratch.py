import numpy as np
import torch
from ProcessData import pca, normalise_data_min_max, pca_with_batch_processing

'''''''''''
#Passed:
data = np.array([[1, 1, 10, 25],
                        [1, 11, 11, 25],
                        [1, 2, 12, 25]], dtype=float) # the rows are the observations
'''''
'''''''''''
#Passed:
data = np.array([[1, 1, 10, 25],
                        [1, 11, 11, 25],
                        [1, 2, 12, 24]], dtype=float) # the rows are the observations
'''''
#Passed
data = np.array([[1, 25, 10, 1],
                        [1, 25, 11, 11],
                        [1, 25, 12, 2]], dtype=float)

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
print("This is the expected result:\n", reconstructed_data)

variance_of_reconstructed_data = torch.var(U_reduced, dim=1)

data = np.array([[1, 25, 10, 1],
                        [1, 25, 11, 11],
                        [1, 25, 12, 2]], dtype=float)
data = torch.from_numpy(data)

data = normalise_data_min_max(data)
#pca_result = pca(data)
pca_result = pca_with_batch_processing(data, batch_size=2)
print(pca_result)

# Since there are many samples and a large number of features, perform PCA in the following manner:
# 1. Normalise the data
# 2. Randomly select some data points which will be a part of the test set
# 3. Perform PCA and reconstruct data (Note that the data shape will be the same as the original data set, just that
# some dimensions are more squashed than before)
# 4. Compute the variances feature-wise and remove features with very little variance, at the same time keeping track
# of the indices of the features being removed
# 5. Store the indices for the test set.