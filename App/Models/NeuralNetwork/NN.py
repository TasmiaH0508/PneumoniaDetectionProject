from torch import nn

from App.PrepareData import *
from App.ComputeMetrics import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, num_input_features):
        super().__init__()
        self.num_input_features = num_input_features
        self.L1 = nn.Linear(num_input_features, 1)
        self.L2 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        return x

def train_model(model, epochs, train_data, test_data, lr, bias_present_for_training_set=True, bias_present_for_testing_set=True, min_accuracy_for_highest_recall_model=92, save_model=False):
    # prepare data
    train_labels = get_label(train_data)
    train_labels = torch.reshape(train_labels, (train_labels.shape[0],1))
    train_data = get_data_without_bias_and_label(train_data, has_bias=bias_present_for_training_set)
    test_labels = get_label(test_data)
    test_data = get_data_without_bias_and_label(test_data, has_bias=bias_present_for_testing_set)

    # train
    loss = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    max_accuracy_model = None
    max_accuracy_score = 0
    max_recall_model = None
    max_recall_score = 0
    thresholds = [0.4, 0.5, 0.6]
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model.forward(train_data)
        loss_val = loss(y_pred, train_labels)
        if i % 100 == 0:
            print(loss_val.item())
        if i == epochs - 1:
            print("The loss is now:", loss_val.item())
            print("Epochs used:", i + 1)
        loss_val.backward()
        optimiser.step()
        pred = model(test_data)
        for threshold in thresholds:
            pred_given_threshold = torch.where(pred >= threshold, 1, 0)
            accuracy_score = get_accuracy(test_labels, pred_given_threshold)
            recall_score = get_recall(test_labels, pred_given_threshold)
            if accuracy_score > max_accuracy_score:
                max_accuracy_score = accuracy_score
                iterations = i + 1
                max_accuracy_model = (lr, iterations, threshold, accuracy_score, recall_score)
                if save_model:
                    torch.save(model.state_dict(), "model_max_accuracy.pth")
            if recall_score > max_recall_score and accuracy_score > min_accuracy_for_highest_recall_model:
                max_recall_score = recall_score
                iterations = i + 1
                max_recall_model = (lr, iterations, threshold, accuracy_score, recall_score)
                if save_model:
                    torch.save(model.state_dict(), "model_max_recall.pth")
    return (max_accuracy_model, max_recall_model)

def predict_with_saved_model(processed_image_arr):
    num_input_features = 65536
    model = NeuralNetwork(num_input_features)
    weights = torch.load("App/Models/NeuralNetwork/model_max_recall.pth", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    threshold_prob = 0.4
    input_arr = processed_image_arr.float()
    pred = predict(model, threshold_prob, input_arr, has_label=False)
    pred = pred[0].item()
    return pred

def predict(model, threshold_prob, test_data, bias_present=True, has_label=True):
    ''''
    Returns the prediction.

    The test_data param is a tensor that may or may not include the label or bias.
    If bias is present, set bias_present=True
    If label is present, set has_label=True
    '''
    test_data = get_data_without_bias_and_label(test_data, has_bias=bias_present, has_label=has_label)
    pred = model(test_data)
    pred = torch.where(pred >= threshold_prob, 1, 0)
    return pred

"""
../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy
single layer
(0.001, 518, 0.5, 95.47395388556788, 0.9512820512820512) (0.001, 430, 0.4, 95.2604611443211, 0.9632478632478633)

../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.02.npy
(0.001, 587, 0.5, 95.21776259607174, 0.9538461538461539) (0.001, 341, 0.4, 94.49188727583262, 0.9658119658119658)
"""