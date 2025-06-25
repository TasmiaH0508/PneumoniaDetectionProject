import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim

from App.ComputeMetrics import get_accuracy, get_recall

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
    training_set = torchvision.datasets.ImageFolder(path_to_training_data, transform=transform)
    validation_set = torchvision.datasets.ImageFolder(path_to_validation_set, transform=transform)
    test_set = torchvision.datasets.ImageFolder(path_to_test_set, transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return training_loader, validation_loader, test_loader

def train(epochs, save_highest_accuracy_model=True, save_highest_recall_model=True, save_resulting_model=True, use_old_model=False,
          path_to_use_for_old_model='highest_recall_model.pth'):
    net = CNN()

    if use_old_model:
        weights = torch.load(path_to_use_for_old_model, weights_only=True)
        net.load_state_dict(weights)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loader, val_loader, _ = prepare_data_loader()

    val_inputs, val_labels = None, None
    for j, val_data in enumerate(val_loader):
        val_inputs, val_labels = val_data

    max_accuracy_score = 0
    max_recall_score = 0
    max_accuracy_model = None
    max_recall_model = None
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.float()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if save_highest_accuracy_model:
            output = net(val_inputs)
            accuracy_score = get_accuracy(val_labels, output)
            if accuracy_score > max_accuracy_score:
                max_accuracy_score = accuracy_score
                max_accuracy_model = net

        if save_highest_recall_model:
            output = net(val_inputs)
            recall_score = get_recall(val_labels, output)
            if recall_score > max_recall_score:
                max_recall_score = recall_score
                max_accuracy_model = net

    print('Finished Training')

    if max_accuracy_model is not None and save_highest_accuracy_model:
        torch.save(max_accuracy_model.state_dict(), 'highest_accuracy_model.pth')
        print('Saved highest accuracy model. Model has an accuracy of', max_accuracy_score)

    if max_recall_model is not None and save_highest_recall_model:
        torch.save(max_recall_model.state_dict(), 'highest_recall_model.pth')
        print('Saved highest recall model. Model has an recall of', max_recall_score)

    if save_resulting_model:
        torch.save(net.state_dict(), 'model.pth')

    return net
