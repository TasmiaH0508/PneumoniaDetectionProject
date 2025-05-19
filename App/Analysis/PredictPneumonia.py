from App.ProcessRawData import min_max_normalise_with_predefined_params, process_image
from App.NeuralNetwork.LogisticRegressionNN import predict_with_saved_weights
import numpy as np
import torch

# feed in a 256 by 256 xray picture and get predictions
def simplify_image_data(model, data):
    if model == 'svm' or model == 'nn':
        min_matrix = np.load("../ProcessedRawData/MinData/min_across_all_features_var0.02.npy")
        min_matrix = torch.from_numpy(min_matrix).float()
        range_matrix = np.load("../ProcessedRawData/RangeData/range_across_all_features_var0.02.npy")
        range_matrix = torch.from_numpy(range_matrix).float()
        data = min_max_normalise_with_predefined_params(data, min_matrix, range_matrix)
        indices_kept = np.load("../ProcessedRawData/Index/Indices_Kept_data_var0.02.npy")
        data = data[:, indices_kept]
    return data

def predict_pneumonia(image_path, model='svm'):
    data = process_image(image_path)
    data = simplify_image_data(model, data)
    if model == 'nn':
        pred = predict_with_saved_weights(data, has_bias=False, file_to_read_from="../NeuralNetwork/torch_weights_var_0.02.pth")
    elif model == 'cnn':
        #todo
        pred = 0
    elif model == 'standard':
        #todo
        pred = 0
    else:
        #todo
        pred = 0
    return pred

print(predict_pneumonia("./Normal.png"))