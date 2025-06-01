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
    # seed needed for reproducibility
    def __init__(self, num_input_features, output_size_per_layer, seed=23, negative_slope=0.2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        super().__init__()
        self.num_input_features = num_input_features
        self.L1 = nn.Linear(num_input_features, output_size_per_layer[0])
        self.L2 = nn.LeakyReLU(negative_slope=negative_slope)
        self.L3 = nn.Dropout()
        self.L4 = nn.Linear(output_size_per_layer[0], output_size_per_layer[1])
        self.L5 = nn.LeakyReLU(negative_slope=negative_slope)
        self.L6 = nn.Dropout()
        self.L7 = nn.Linear(output_size_per_layer[1], output_size_per_layer[2])
        self.L8 = nn.ReLU()
        self.L9 = nn.Linear(output_size_per_layer[2], 1)
        self.L10 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.L7(x)
        x = self.L8(x)
        x = self.L9(x)
        x = self.L10(x)
        return x

def train_model(model, epochs, train_data, optimiser, bias_present=True):
    ''''
    Trains model.

    The train_data param is a tensor that must also include the label.
    '''
    # prepare data
    train_labels = get_label(train_data)
    train_labels = torch.reshape(train_labels, (train_labels.shape[0],1))
    train_data = get_data_without_bias_and_label(train_data, has_bias=bias_present)

    # train
    loss = nn.BCELoss()
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model.forward(train_data)
        loss_val = loss(y_pred, train_labels)
        loss_val.backward()
        optimiser.step()
        if i == epochs - 1:
            print("The loss is now:", loss_val.item())
            return loss_val.item()

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

def estimate_best_hyperparameters(file_path_to_training_data, file_path_to_test_data):
    train_data = np.load(file_path_to_training_data)
    train_data = torch.from_numpy(train_data).float()
    test_data = np.load(file_path_to_test_data)
    test_data = torch.from_numpy(test_data).float()
    test_labels = get_label(test_data)

    num_features_wo_bias = train_data.shape[1] - 2

    layer_configs = ((1024, 256, 64),
                     (3024, 256, 32))
    negative_slopes = [0.1, 0.01]
    lr = 0.001
    thresholds = [0.4, 0.5, 0.6]

    max_accuracy = 0.0
    most_acc_model = None
    max_recall = 0.0
    greatest_recall_model = None
    min_change_in_loss = 0.0001
    for layer_config in layer_configs:
        print("Trying config:", layer_config)
        for negative_slope in negative_slopes:
            loss, prev_loss = 0, float('inf')
            model = NeuralNetwork(num_features_wo_bias, layer_config, negative_slope=negative_slope)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            for i in range(6):
                print("Training model ", i, "th iteration")
                if i == 0:
                    loss = train_model(model, 150, train_data, optimiser)
                else:
                    loss = train_model(model, 50, train_data, optimiser)
                if loss >= prev_loss or prev_loss - loss <= min_change_in_loss:
                    break
                prev_loss = loss
                for threshold in thresholds:
                    pred = predict(model, threshold, test_data)
                    acc_score = get_accuracy(test_labels, pred)
                    epochs_used = 150 + 50 * i
                    recall_score = get_recall(test_labels, pred)
                    if acc_score > max_accuracy:
                        max_accuracy = acc_score
                        most_acc_model = (layer_config, negative_slope, threshold, epochs_used, acc_score, recall_score)
                    if recall_score > max_recall:
                        max_recall = recall_score
                        greatest_recall_model = (layer_config, negative_slope, threshold, epochs_used, acc_score, recall_score)
    return most_acc_model, greatest_recall_model

(most_acc_model,
 greatest_recall_model) = (
    estimate_best_hyperparameters("../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy",
                                  "../Data/ProcessedRawData/TestSet/PvNormalDataNormalised.npy"))

print(most_acc_model)
print(greatest_recall_model)

"""""""""""
"../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy"
seed = 423
((3024, 256, 32), 0.01, 0.4, 250, 92.22361024359775, 0.93875)
((3024, 256, 32), 0.01, 0.4, 150, 87.88257339163023, 0.979375)

seed = 23

"../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised_var0.02.npy"
seed = 423
((3024, 256, 32), 0.01, 0.4, 150, 91.88007495315428, 0.9325)
((3024, 256, 32), 0.1, 0.4, 150, 49.968769519050596, 1.0)
"""""