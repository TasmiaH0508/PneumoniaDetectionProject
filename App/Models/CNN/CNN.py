import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

target_size = (256, 256)

transform_to_use = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3),
                                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
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

        self.L7 = nn.Linear(128 * 32 * 32, 128)
        self.L8 = nn.Dropout(0.5)
        self.L9 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        print(x.shape) # batch_size=32, input_channels=128, height=64, width=64
        x = self.L6(x)
        print(x.shape) # batch_size=32, input_channels=128, height=32, width=32
        x = self.L7(x)
        print(x.shape)
        x = self.L8(x)
        print(x.shape)
        x = self.L9(x)
        print(x.shape)
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

def train(epochs):
    net = CNN()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loader, val_loader, test_loader = prepare_data_loader()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')