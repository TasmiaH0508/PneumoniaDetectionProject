import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

target_size = (256, 256)

transform_to_use = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
                                #torchvision.transforms.RandomHorizontalFlip(p=0.5),
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

def prepare_data_loader(transform=transform_to_use, path_to_training_data='../Data/CNNData/Train',
                        path_to_validation_set='../Data/CNNData/Validation',
                        path_to_test_set='../Data/CNNData/Test', batch_size=32):
    # NORMAL samples have label 0 and PNEUMONIA samples have label 1
    training_set = torchvision.datasets.ImageFolder(path_to_training_data, transform=transform)
    validation_set = torchvision.datasets.ImageFolder(path_to_validation_set, transform=transform)
    test_set = torchvision.datasets.ImageFolder(path_to_test_set, transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return training_loader, validation_loader, test_loader

def train(epochs, save_highest_accuracy_model=True, save_highest_recall_model=True, save_final_model=True, use_old_model=False,
          path_to_use_for_old_model='highest_recall_model.pth', path_to_use_for_max_accuracy_model='highest_accuracy_model.pth',
          path_to_use_for_max_recall_model='highest_recall_model.pth', threshold=0.4):
    net = CNN()

    if use_old_model:
        weights = torch.load(path_to_use_for_old_model, weights_only=True)
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
                accuracy_score = get_accuracy(actual_labels, predicted_labels, threshold)
                if accuracy_score > max_accuracy_score:
                    max_accuracy_score = accuracy_score
                    max_accuracy_model = net
            else:
                recall_score = get_recall(actual_labels, predicted_labels, threshold)
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
        torch.save(net.state_dict(), 'Weights/final_model.pth')

    return net

def get_accuracy(actual_labels, predicted_labels, threshold):
    total_labels = 0
    correct_labels = 0
    for i in range(len(actual_labels)):
        for j in range(len(actual_labels[i])):
            total_labels += 1
            if predicted_labels[i][j] >= threshold:
                predicted_labels[i][j] = 1
            else:
                predicted_labels[i][j] = 0
            if actual_labels[i][j] == predicted_labels[i][j]:
                correct_labels += 1
    accuracy = correct_labels / total_labels
    return accuracy

def get_recall(actual_labels, predicted_labels, threshold):
    true_positives = 0
    false_negatives = 0
    for i in range(len(actual_labels)):
        for j in range(len(actual_labels[i])):
            if predicted_labels[i][j] >= threshold:
                predicted_labels[i][j] = 1
            else:
                predicted_labels[i][j] = 0
            if actual_labels[i][j] == predicted_labels[i][j] and actual_labels[i][j] == 1:
                true_positives += 1
            if predicted_labels[i][j] == 0 and predicted_labels[i][j] != actual_labels[i][j]:
                false_negatives += 1
    recall_score = true_positives / (true_positives + false_negatives)
    return recall_score

train(2, use_old_model=True, path_to_use_for_old_model='Weights/final_model.pth')

"""
1:
Training loss: 44.959220826625824
Validation loss: 226.7857141494751
Finished Training
Max accuracy reached: 0.8666666666666667

3:
Training loss: 18.04400011152029
Validation loss: 122.32142853736877
Finished Training
Max accuracy reached: 0.9277777777777778
"""