import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

target_size = (256, 256)

transform_to_use = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
                                torchvision.transforms.Resize(target_size),
                                torchvision.transforms.ToTensor()])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.L1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.L2 = nn.MaxPool2d(2, 2)

        self.L3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.L4 = nn.MaxPool2d(2, 2)

        self.L5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.L6 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.L7 = nn.Linear(128 * 32 * 32, 1)
        self.L8 = nn.Sigmoid()

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.flatten(x)
        x = self.L7(x)
        x = self.L8(x)
        return x.view(-1)

def prepare_data_loader(transform=transform_to_use, get_training_set=True, get_validation_set=True, get_test_set=False,
                        batch_size=32):
    path_to_training_data = '../Data/CNNData/Train'
    path_to_validation_set = '../Data/CNNData/Validation'
    path_to_test_set = '../Data/CNNData/Test'
    # NORMAL samples have label 0 and PNEUMONIA samples have label 1
    training_loader = None
    if get_training_set:
        training_set = torchvision.datasets.ImageFolder(path_to_training_data, transform=transform)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    validation_loader = None
    if get_validation_set:
        validation_set = torchvision.datasets.ImageFolder(path_to_validation_set, transform=transform)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    test_loader = None
    if get_test_set:
        test_set = torchvision.datasets.ImageFolder(path_to_test_set, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return training_loader, validation_loader, test_loader

def get_accuracy(actual_labels, predicted_labels):
    total_labels = 0
    correct_labels = 0
    for i in range(len(actual_labels)):
        for j in range(len(actual_labels[i])):
            total_labels += 1
            if actual_labels[i][j] == predicted_labels[i][j]:
                correct_labels += 1
    accuracy = correct_labels / total_labels
    return accuracy

def get_recall(actual_labels, predicted_labels):
    true_positives = 0
    false_negatives = 0
    for i in range(len(actual_labels)):
        for j in range(len(actual_labels[i])):
            if actual_labels[i][j] == predicted_labels[i][j] and actual_labels[i][j] == 1:
                true_positives += 1
            if predicted_labels[i][j] == 0 and predicted_labels[i][j] != actual_labels[i][j]:
                false_negatives += 1
    recall_score = true_positives / (true_positives + false_negatives)
    return recall_score

def train(epochs, save_highest_accuracy_model=True, save_highest_recall_model=True, save_final_model=True, use_old_model=False,
          path_to_old_model='highest_recall_model.pth', path_to_use_for_max_accuracy_model='highest_accuracy_model.pth',
          path_to_use_for_max_recall_model='highest_recall_model.pth', threshold=0.4):
    net = CNN()

    if use_old_model:
        weights = torch.load(path_to_old_model, weights_only=True)
        net.load_state_dict(weights)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loader, val_loader, _ = prepare_data_loader()

    use_validation_set = save_highest_accuracy_model or save_highest_recall_model

    max_accuracy_score = 0
    max_recall_score = 0
    max_accuracy_model = None
    max_recall_model = None
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            train_inputs, train_labels = data
            train_labels = train_labels.float()

            optimizer.zero_grad()

            outputs = net(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Training loss:", running_loss)

        if use_validation_set:
            net.eval()
            validation_loss = 0
            actual_labels = []
            predicted_labels = []
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    val_inputs, val_labels = data
                    val_labels = val_labels.float()
                    actual_labels.append(val_labels)
                    pred = net(val_inputs)
                    pred = (pred >= threshold).float()
                    predicted_labels.append(pred)
                    loss = criterion(pred, val_labels)
                    validation_loss += loss.item()

            print("Validation loss:", validation_loss)

            if save_highest_accuracy_model:
                accuracy_score = get_accuracy(actual_labels, predicted_labels)
                if accuracy_score > max_accuracy_score:
                    max_accuracy_score = accuracy_score
                    max_accuracy_model = net

            if save_highest_recall_model:
                recall_score = get_recall(actual_labels, predicted_labels)
                if recall_score > max_recall_score:
                    max_recall_score = recall_score
                    max_recall_model = net

    print('Finished Training')

    if save_highest_accuracy_model and max_accuracy_model is not None:
        torch.save(max_accuracy_model.state_dict(), path_to_use_for_max_accuracy_model)
        print("Max accuracy reached:", max_accuracy_score)

    if save_highest_recall_model and max_recall_model is not None:
        torch.save(max_recall_model.state_dict(), path_to_use_for_max_recall_model)
        print("Max recall reached:", max_recall_score)

    if save_final_model:
        torch.save(net.state_dict(), 'final_model.pth')

    return net

def print_metrics_of_input_set(path_to_model, set_to_use='validation', threshold=0.4):
    if set_to_use == 'test':
        set_to_compute = prepare_data_loader(get_test_set=True, get_validation_set=False, get_training_set=False)[2]
    else:
        set_to_compute = prepare_data_loader(get_training_set=False, get_validation_set=True, get_test_set=False)[1]

    model = CNN()
    weights = torch.load(path_to_model, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for i, data in enumerate(set_to_compute, 0):
            inputs, labels = data
            labels = labels.float()
            actual_labels.append(labels)
            pred = model(inputs)
            pred = (pred >= threshold).float()
            predicted_labels.append(pred)

    accuracy = get_accuracy(actual_labels, predicted_labels)
    recall = get_recall(actual_labels, predicted_labels)
    print('Accuracy:', accuracy)
    print('Recall:', recall)

def predict_with_saved_model():
    net = CNN()
    # todo
    return 0

"""
10:
Training loss: 44.959220826625824
Validation loss: 226.7857141494751
Training loss: 23.764911457896233
Validation loss: 399.1071434020996
Training loss: 20.690328285098076
Validation loss: 115.625
Training loss: 15.628364123404026
Validation loss: 183.0357141494751
Training loss: 14.99319913238287
Validation loss: 104.9107141494751
Training loss: 13.012308485805988
Validation loss: 117.85714340209961
Training loss: 12.811642792075872
Validation loss: 100.44642853736877
Training loss: 11.897422458976507
Validation loss: 88.39285707473755
Training loss: 11.719549261033535
Validation loss: 91.51785707473755
Training loss: 12.035624828189611
Validation loss: 87.94642853736877
Finished Training
Max accuracy reached: 0.9481481481481482
Max recall reached: 0.9925925925925926

"Weights/highest_accuracy_model_0.948_t0.4.pth" 
Val
Accuracy: 0.9481481481481482
Recall: 0.9333333333333333
Test
Accuracy: 0.9574074074074074
Recall: 0.9555555555555556

"Weights/highest_recall_model_0.993_t0.4.pth"
Val
Accuracy: 0.8925925925925926
Recall: 0.9777777777777777

Test
Accuracy: 0.8814814814814815
Recall: 0.9851851851851852
"""