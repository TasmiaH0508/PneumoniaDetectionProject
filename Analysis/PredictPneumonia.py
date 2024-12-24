from SVM.SVM import SVMClassifier, get_predictions_with_previously_loaded_model


# feed in a 256 by 256 xray picture and get predictions
# the indices need to be removed...
def process_image(image):
    #todo: return array with bias only and no label
    return 0

# need to consider how the best model can be loaded
def predict_pneumonia_SVM(image):
    model = SVMClassifier(kernel_type='rbf')
    data = process_image(image)
    pred = get_predictions_with_previously_loaded_model(model, data)
    if pred = 0:
        print("No pneumonia detected.")
    else:
        print("Pneumonia detected.")

def predict_pneumonia_CNN(image):
    #todo
    return 0

def predict_pneumonia_NN(image):
    #todo
    return 0

def predict_pneumonia_wo_NN(image):
    #todo
    return 0