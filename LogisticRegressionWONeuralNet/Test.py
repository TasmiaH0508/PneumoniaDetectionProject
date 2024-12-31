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

test = torch.asarray([[0, 3],
                      [3, 0]])

#weights = train_model(10, train, has_bias=False, poly_deg=2, lr=0.01)
#print(predict(test, weights, has_label=True, has_bias=False, poly_deg=2))

testing = torch.asarray([[0, 1, 0],
                         [1, 0, 1]])

#print(transform_features(testing))

print(transform_features_with_batch_processing(testing, batch_size=1))