import torch
import numpy as np

test = torch.asarray([[0, 1, 1],
                      [1, 0, 1],
                      [0, 0, 1],
                      [2, 0, 0],
                      [0, 2, 0]]) # the first 2 cols are features

num_samples = 5
np.random.seed(0)
for i in range(num_samples):
    rand_index = np.random.choice(num_samples, size=1, replace=False)[0]
    print(rand_index)


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