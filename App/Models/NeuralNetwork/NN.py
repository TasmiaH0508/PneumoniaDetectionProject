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

def create_validation_set_from_training_set(train_data, num_samples_in_training_set):
    num_samples = train_data.shape[0]
    num_negative_samples = num_samples // 2

    indices_for_negative_samples = np.arange(num_negative_samples)
    negative_examples = train_data[indices_for_negative_samples]

    indices_for_positive_samples = np.arange(num_negative_samples, num_samples)
    positive_examples = train_data[indices_for_positive_samples]

    np.random.seed(42)
    random_indices_training_indices = np.random.choice(num_negative_samples, num_samples_in_training_set, replace=False)
    indices_in_validation_set = np.ones(num_negative_samples)
    indices_in_validation_set = -indices_in_validation_set
    indices_in_validation_set[random_indices_training_indices] = random_indices_training_indices
    indices_in_validation_set = np.where(indices_in_validation_set == -1)[0]
    negative_examples_training = negative_examples[random_indices_training_indices]
    negative_examples_validation = negative_examples[indices_in_validation_set]

    positive_examples_training = positive_examples[random_indices_training_indices]
    positive_examples_validation = positive_examples[indices_in_validation_set]

    training_set = torch.vstack((negative_examples_training, positive_examples_training))
    validation_set = torch.vstack((negative_examples_validation, positive_examples_validation))
    return training_set, validation_set

def train_model(model, epochs, train_data, lr, bias_present_for_training_set=True, save_most_accurate_model=False, save_highest_recall_model=False, num_samples_in_training_set=504):
    train_set, validation_set = create_validation_set_from_training_set(train_data, num_samples_in_training_set) # 80% of training data

    train_labels = get_label(train_set)
    train_labels = torch.reshape(train_labels, (train_labels.shape[0],1))
    train_set = get_data_without_bias_and_label(train_set, has_bias=bias_present_for_training_set)

    validation_set = get_data_without_bias_and_label(validation_set, has_bias=bias_present_for_training_set)
    validation_labels = get_label(validation_set)

    loss = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    highest_accuracy_score = 0
    highest_recall_score = 0
    accuracy_score_for_highest_recall_model = 0
    highest_recall_model = None
    highest_accuracy_model = None
    threshold_used_for_high_accuracy = 0
    threshold_used_for_high_recall = 0
    thresholds = [0.4, 0.5, 0.6]
    for i in range(epochs):
        optimiser.zero_grad()
        y_pred = model.forward(train_set)
        loss_val = loss(y_pred, train_labels)
        if i % 100 == 0:
            print(loss_val.item())
        if i == epochs - 1:
            print("The loss is now:", loss_val.item())
            print("Epochs used:", i + 1)
        loss_val.backward()
        optimiser.step()
        pred_for_validation_set = model(validation_set)
        for threshold in thresholds:
            pred_for_validation_set = torch.where(pred_for_validation_set >= threshold, 1, 0)
            accuracy_score = get_accuracy(validation_labels, pred_for_validation_set)
            recall_score = get_recall(validation_labels, pred_for_validation_set)
            if accuracy_score > highest_accuracy_score:
                highest_accuracy_score = accuracy_score
                highest_accuracy_model = model
                threshold_used_for_high_accuracy = threshold
            if recall_score > highest_recall_score and accuracy_score > 0.9:
                highest_recall_score = recall_score
                highest_recall_model = model
                accuracy_score_for_highest_recall_model = accuracy_score
                threshold_used_for_high_recall = threshold

    print("The highest accuracy score is:", highest_accuracy_score)
    print("The highest recall score is:", highest_recall_score)
    print("The accuracy score for the highest recall model is", accuracy_score_for_highest_recall_model)
    print("Thresholds for highest accuracy, highest recall:", threshold_used_for_high_accuracy, threshold_used_for_high_recall)

    if save_most_accurate_model and highest_accuracy_model is not None:
        torch.save(highest_accuracy_model.state_dict(), "model_weights_most_accurate.pth")

    if save_highest_recall_model and highest_recall_score is not None:
        torch.save(highest_recall_model.state_dict(), "model_weights_highest_recall.pth")

train_set = np.load("../Data/ProcessedRawData/TrainingSet/PvNormalDataNormalised.npy")
train_set = torch.from_numpy(train_set).float()
num_features = train_set.shape[1] - 2
model = NeuralNetwork(num_features)
train_model(model, 500, train_set, lr=0.001, save_most_accurate_model=True, save_highest_recall_model=True)

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

def predict(model, threshold_prob, data, bias_present=True, has_label=True):
    ''''
    Returns the prediction.

    The test_data param is a tensor that may or may not include the label or bias.
    If bias is present, set bias_present=True
    If label is present, set has_label=True
    '''
    data = get_data_without_bias_and_label(data, has_bias=bias_present, has_label=has_label)
    pred = model(data)
    pred = torch.where(pred >= threshold_prob, 1, 0)
    return pred