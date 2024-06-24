#Name :- Mohit Hemnani
#Roll No. :- 22EE10043
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Set random seed for reproducibility
torch.manual_seed(42)

# Define data transformations and load CIFAR-10 dataset
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the CNN-Vanilla model
class CNNVanilla(nn.Module):
    def __init__(self):
        super(CNNVanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the CNN-Resnet model
class CNNResnet(nn.Module):
    def __init__(self):
        super(CNNResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the CNN-Resnet model with four-level Resnet block
class CNNResnetFourLevel(nn.Module):
    def __init__(self):
        super(CNNResnetFourLevel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# List of models to compare
models = [
    ('CNN-Vanilla', CNNVanilla()),
    ('CNN-Resnet', CNNResnet()),
    ('CNN-Resnet (Four-Level Resnet)', CNNResnetFourLevel()),
]


# Define the list of optimizers and their configurations
optimizers = [
    ('SGD', optim.SGD, {'lr': 0.001}),
    ('Mini-Batch GD (No Momentum)', optim.SGD, {'lr': 0.001, 'momentum': 0}),
    ('Mini-Batch GD (Momentum 0.9)', optim.SGD, {'lr': 0.001, 'momentum': 0.9}),
    ('Adam', optim.Adam, {'lr': 0.001}),
]

# Function to train and evaluate a model
def train_and_evaluate(model, trainloader, testloader, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_instance = optimizer[1](model.parameters(), **optimizer[2])

    training_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer_name = optimizer[0]  # Get the optimizer name
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer_instance.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_instance.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        training_accuracy.append(accuracy)

        print(f'Optimizer: {optimizer_name}, Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {accuracy:.2f}%')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, training_accuracy

# Main loop to run experiments
results = []
best_choice = {}  # To store the best choice among models and optimizers

# Experiment 1: Training accuracy vs epochs for CNN-Vanilla and CNN-Resnet
# Define the number of epochs
num_epochs = 50

# Experiment 1: Training accuracy vs epochs for CNN-Vanilla and CNN-Resnet
for model_name, model in models:
    for optimizer in optimizers:
        accuracy, training_accuracy = train_and_evaluate(model, trainloader, testloader, optimizer, num_epochs)

        results.append((model_name, optimizer[0], accuracy))

        # Find the best choice based on the highest test accuracy
        if model_name not in best_choice or accuracy > best_choice[model_name]['accuracy']:
            best_choice[model_name] = {'optimizer': optimizer[0], 'accuracy': accuracy}

        # Plot training accuracy vs epochs
        plt.plot(range(1, num_epochs + 1), training_accuracy, label=f'{model_name} - {optimizer[0]}')

