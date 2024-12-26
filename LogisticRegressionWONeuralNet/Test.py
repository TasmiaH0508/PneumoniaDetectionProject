import torch

from LogisticRegressionWONeuralNet.LogisticRegression_wo_NN import transform_features

test = torch.asarray([[0, 1, 1],
                      [1, 0, 1],
                      [0, 0, 1],
                      [2, 0, 0],
                      [0, 2, 0]]) # the first 2 cols are features
test = test[:, [0, 1]]

print(transform_features(test, poly_deg=3))
# is returning [u, v, u^2, u.v, v.u, v^2]

# if poly_deg == 1,
# [u, v] -> [1, u, v]

# if poly_deg == 2,
# [u, v] -> [u, v, u^2, uv, v^2]

# [x, y, z] -> [x, y, z, x.x, x.y, x.z, y.y, y.z, z.z]

# if poly_deg == 3,
# [u, v]
# -> [u, v, uv, u^2, v^2] (Get poly deg of 2 first)
# -> [u, v, uv, u^2, v^2, u^2v, ...] (j should loop from: 0, 1, k should loop from: 2, 3, ...)

# [u] -> [u, u.u]