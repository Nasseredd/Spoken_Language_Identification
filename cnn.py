import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Number of classes: en, de, es
NUM_CLASSES = 3

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_epochs, learning_rate):

        super(ConvNet, self).__init__()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

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

        self.fc = nn.Linear(32*32*107, NUM_CLASSES)
        self.act = nn.Softmax(dim=1)

        # Loss and optimizer (initialized during training)
        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.act(out)
        return out

    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader = None):
        # Send model (self) to DEVICE
        self = self.to(DEVICE)

        # Store losses into Numpy Array
        losses = np.zeros((self.num_epochs, 2))

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

        # Shortcut Function for validation
        def validate(): return self.validation(validation_loader)

        # Train the model
        for epoch in range(1, self.num_epochs + 1):
            # Perform training and validation.
            train_loss = np.mean(
                list(itertools.starmap(self.train_iter, train_loader)))
            val_loss = validate()

            # Print & store losses
            print('Epoch [{}/{}], Loss: {:.4f}, Val. Loss: {:.4f}'.format(epoch,
                  self.num_epochs, train_loss, val_loss))
            losses[epoch] = [train_loss, val_loss]

            # Create model checkpoint
            self.create_checkpoint(epoch)

        # Store losses into CSV file.
        losses_df = pd.DataFrame(losses, columns=["Training", "Validation"])
        losses_df.to_csv("training_losses.csv")

    def forward_pass(self, mel_spectro, expected):
        mel_spectro = mel_spectro.to(DEVICE)
        expected = expected.to(DEVICE)
        expected = expected.squeeze()

        # Forward pass
        predicted = self(mel_spectro)
        return self.criterion(predicted, expected)

    def train_iter(self, mel_spectro, lang):
        loss = self.forward_pass(mel_spectro, lang)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation(self, validation_loader):
        self.eval()  # Activate evaluation mode
        with torch.no_grad():
            def validation_iter(x, y): return self.forward_pass(x, y).item()
            loss = np.mean(list(itertools.starmap(
                validation_iter, validation_loader)))
        self.train()  # Activate training mode
        return loss

    def create_checkpoint(self, epoch):
        path = os.path.join('checkpoints', f'model_{epoch}.ckpt')
        try:
            torch.save(self.state_dict(), path)
        except FileNotFoundError:
            os.mkdir('checkpoints')
            torch.save(self.state_dict(), path)

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
