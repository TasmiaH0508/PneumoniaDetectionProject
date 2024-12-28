import torch
import numpy as np
from LogisticRegression_wo_NN import *

# decision boundary = x^2 + 2xy
# if x^2 + 2xy > 0, 1 else 0

train = torch.asarray([[0, 1, 0],
                      [1, 0, 1],
                      [0, 0, 0],
                      [2, 0, 1],
                      [0, 2, 0]]) # the first 2 cols are features

test = torch.asarray([[0, 3, 0],
                      [3, 0, 1]])

weights = train_model(100, train, has_bias=False, poly_deg=2, lr=0.01)

pred = predict(weights, test, has_bias=False, poly_deg=2)

print(pred)