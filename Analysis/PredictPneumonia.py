from ProcessRawData import process_image
from SVM.SVM import get_predictions_with_previously_loaded_model
from NeuralNetwork.LogisticRegressionNN import predict_with_saved_weights
import numpy as np
import torch

# feed in a 256 by 256 xray picture and get predictions

def simplify_image_data(model, data):
    if model == 'svm' or model == 'nn':
        indices_kept = np.load("../ProcessedRawData/Index/Indices_Kept_data_var0.02.npy")
        data = data[:, indices_kept]
    return data

def predict_pneumonia(image_path, model='svm'):
    data = process_image(image_path)
    data = simplify_image_data(model, data)
    pred = None
    if model == 'svm':
        pred = get_predictions_with_previously_loaded_model(data, has_bias=False, file_to_read_from="../SVM/svm_model.joblib")
    elif model == 'nn':
        pred = predict_with_saved_weights(data, has_bias=False, file_to_read_from="../NeuralNetwork/torch_weights_var_0.02.pth")
    elif model == 'cnn':
        #todo
        pred = 0
    elif model == 'standard':
        #todo
        pred = 0
    if pred is not None:
        if pred == 0:
            print("No pneumonia detected.")
        else:
            print("Pneumonia detected.")
    else:
        print("You did not pick a valid method.")

#image_path = "../RawData/PNEUMONIA/Pneumonia.png"
#predict_pneumonia(image_path, model='nn')