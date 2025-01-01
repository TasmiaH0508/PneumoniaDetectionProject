import torch
import numpy as np
from LogisticRegression_wo_NN import *
import time

# decision boundary = x^2 + 2xy
# if x^2 + 2xy > 0, 1 else 0

train = torch.asarray([[0, 1, 0],
                      [1, 0, 1],
                      [0, 0, 0],
                      [2, 0, 1],
                      [0, 2, 0]]) # the first 2 cols are features

test = torch.asarray([[0, 3, 1, 2],
                      [3, 0, 3, 4]])

#weights = train_model(10, train, has_bias=False, poly_deg=2, lr=0.01)
#print(predict(test, weights, has_label=True, has_bias=False, poly_deg=2))

weights = torch.asarray([[1, 1, 0, 1]])
label = torch.asarray([[0, 1]])

get_weights_batch_gradient_descent(1, test, label, weights, 0.1)