import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
NUM_EPOCHS = 5
NUM_CLASSES = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.001


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):

        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def train(self, train_loader):
        self.to(DEVICE)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        # Train the model
        for epoch in range(NUM_EPOCHS):
            for _, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, loss.item()))
            torch.save(self.state_dict(), f'model_{epoch+1}.ckpt')

    def test(self, test_loader):
        # Test the model
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(
                100 * correct / total))
