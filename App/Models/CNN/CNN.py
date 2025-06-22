import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

target_size = (256, 256)

transform_to_use = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                torchvision.transforms.Resize(target_size),
                                torchvision.transforms.ToTensor()])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

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

def train():
    #t odo
    net = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, valloader, testloader = prepare_data_loader()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')