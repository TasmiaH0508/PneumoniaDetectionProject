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
                      [3, 0, 1],
                      [2, 4, 1]])

#weights = train_model(10, train, has_bias=False, poly_deg=2, lr=0.01)
#print(predict(test, weights, has_label=True, has_bias=False, poly_deg=2))

test_1 = torch.asarray([[1, 2],
                      [2, 4]])
#print(test_1)

res = transform_features(test_1, poly_deg=3)
#print(res)

test_1_up_to_quadratic_term = transform_features(test_1, poly_deg=2)
print(test_1_up_to_quadratic_term)

quadratic_features = get_new_quadratic_features(test_1, test_1)
print("The quadratic terms are: \n", quadratic_features)