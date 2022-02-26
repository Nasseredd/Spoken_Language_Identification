from email import header
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
NUM_EPOCHS = 10
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

        self.fc = nn.Linear(32*32*107, num_classes)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.act(out)
        return out

    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader = None):
        self = self.to(DEVICE)
        losses = np.zeros((NUM_EPOCHS, 2))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        # Training iteration (for map())
        def train_iter(mel_spectro, lang): return self.__train_iteration(
            criterion, optimizer, mel_spectro, lang)
        
        # Validation function: execute validation() or return None
        validate = lambda: self.validation(criterion, validation_loader) if validation_loader else lambda: None

        # Train the model
        for epoch in range(NUM_EPOCHS):
            loss = list(itertools.starmap(train_iter, train_loader))[-1].item()

            val_loss = validate()

            print('Epoch [{}/{}], Loss: {:.4f}, Val. Loss: {:.4f}'.format(epoch +
                  1, NUM_EPOCHS, loss, val_loss))

            losses[epoch] = [loss, val_loss]
            torch.save(self.state_dict(), f'model_{epoch+1}.ckpt')

        losses_df = pd.DataFrame(losses, columns=["Training", "Validation"])
        losses_df.to_csv("training_losses.csv")

    def validation(self, criterion, validation_loader):
        self.eval()
        loss = 0
        with torch.no_grad():
            for mel_spectro, lang in validation_loader:
                mel_spectro = mel_spectro.to(DEVICE)
                lang = lang.to(DEVICE)
                lang = lang.squeeze()

                out = self(mel_spectro)
                loss += criterion(out, lang)
        self.train()
        return loss / len(validation_loader)

    def __train_iteration(self, criterion, optimizer, mel_spectro, lang):
        mel_spectro = mel_spectro.to(DEVICE)
        lang = lang.to(DEVICE)
        lang = lang.squeeze()
        # Forward pass
        outputs = self(mel_spectro)
        loss = criterion(outputs, lang)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def test(self, test_loader: DataLoader):
        # Test the model
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for mel_spectro, lang in test_loader:
                mel_spectro = mel_spectro.to(DEVICE)
                lang = lang.to(DEVICE)
                lang = lang.squeeze()
                outputs = self(mel_spectro)
                _, predicted = torch.max(outputs.data, 1)
                total += lang.size(0)
                correct += (predicted == lang).sum().item()

            str_output = 'Test Accuracy of the model on the {} test mel spectrograms: {} %'.format(total,
                                                                                            100 * correct / total)
            print(str_output)

            with open("test_result.txt", 'w') as file:
                file.write(str_output)
