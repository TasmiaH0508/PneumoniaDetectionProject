import torch
import numpy as np
from LogisticRegression_wo_NN import *
import time

''''
Consider a decision boundary: (x - y)^2 <= 1 -> x^2 - 2xy - y^2 <= 1
Points outside the boundary are classified as 0 and those within are classified as 1
'''

train = torch.asarray([[0.5, 0.5, 1],
                       [2, 0, 0],
                       [4, 2, 0],
                       [1, 0, 1]])

test = torch.asarray([[0, 0, 1],
                      [15, 1, 0],
                      [2, 4, 0],
                      [0, 1, 1]])

weights = train_model(300, train, has_bias=False, poly_deg=3, batch_size=1)

print(weights)

pred = predict(test, weights, has_bias=False, poly_deg=3)

print(pred)

print("Now setting poly deg as 1...")

weights = train_model(300, train, has_bias=False, poly_deg=1, batch_size=1)
print(weights)
pred = predict(test, weights, has_bias=False, poly_deg=1)
print(pred) # produces a less accurate result, which is expected