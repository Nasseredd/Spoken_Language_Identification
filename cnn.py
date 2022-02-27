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


class ConvNet(nn.Module):
    """ Convolutional neural network for spoken language recognition.
    Subclass of pyTorch's nn.Module class.
    Use 2 Sequential layers: Convolution (k=5, s=1, p=2), Batch normalization, ReLU activation, Max Pooling (k=2, s=2)
    One linear layer, and a softmax one.
    """

    def __init__(self, num_epochs: int, learning_rate: float):
        """Initializer/constructor

        Initialize the layers, store hyper-parameters that directly affect the network.
        Initialize the optimizer and loss criterion to None, to declare them as instance attributes
        They are initialized in the training function, where they are needed.

        Args:
            num_epochs (int): the number of epochs to perform in training.
            learning_rate (float): the network's learning rate.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation
        Pass the input tensor x and its derivatives through the different layers.

        Args:
            x (Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output of the neural network.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.act(out)
        return out

    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader):
        """The model's training function.
        Perform training and validation tasks on the model, thanks to dataloaders.
        Print the losses obtained during training, and store them into Numpy Array, then CSV format through Pandas.
        Relies on itertool.starmap() and np.sum().

        Args:
            train_loader (DataLoader): DataLoader containing the training data.
            validation_loader (DataLoader): DataLoader containing the validation data.
        """
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
        for epoch in range(self.num_epochs):
            epoch_ind = epoch + 1
            # Perform training and validation.
            train_loss = np.mean(
                list(itertools.starmap(self.train_iter, train_loader)))
            val_loss = validate()

            # Print & store losses
            print('Epoch [{}/{}], Loss: {:.4f}, Val. Loss: {:.4f}'.format(epoch_ind,
                  self.num_epochs, train_loss, val_loss))
            losses[epoch] = [train_loss, val_loss]

            # Create model checkpoint
            self.create_checkpoint(epoch_ind)

        # Store losses into CSV file.
        losses_df = pd.DataFrame(losses, columns=["Training", "Validation"])
        losses_df.to_csv("training_losses.csv")

    def forward_pass(self, mel_spectro: torch.Tensor, expected: torch.Tensor):
        """Perform a forward pass with loss computation
        Transfer the arguments to DEVICE, then pass the first one through the network.
        The network output is evaluated with the loss function and the expected output.

        Args:
            mel_spectro (torch.Tensor): Mel Spectrogram — Input to the network.
            expected (torch.Tensor): The fragment's language — the expected result.

        Returns:
            Loss?: The loss of the output compared to the expectation.
        """
        mel_spectro = mel_spectro.to(DEVICE)
        expected = expected.to(DEVICE)

        # Forward pass
        predicted = self(mel_spectro)
        return self.criterion(predicted, expected)

    def train_iter(self, mel_spectro: torch.Tensor, lang: torch.Tensor):
        """Iteration of the training session (excluding validation)
        Perform a forward pass and a backward propagation given an input and a reference.
        Used for and by itertools.starmap(). 

        Args:
            mel_spectro (torch.Tensor): Mel Spectrogram — Input to the network.
            lang (torch.Tensor): The fragment's language — the expected result.

        Returns:
            float?: The numerical value of the loss.
        """
        loss = self.forward_pass(mel_spectro, lang)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation(self, validation_loader: DataLoader) -> float:
        """Compute the validation loss of the model at a given epoch.
        Put the model on evaluation and no_grad mode, before computing individual validation losses, 
        then the mean of those to get the batch's. Reactivate the model's training mode after computations.

        Use the output of forward_pass and extract the numerical value through a emdbed function, 
        to use itertools.starmap().

        Args:
            validation_loader (DataLoader): A DataLoader instance containing the validation data.

        Returns:
            float: The mean of the validation losses of the batch — the batch's validation loss.
        """
        self.eval()  # Activate evaluation mode
        with torch.no_grad():
            def validation_iter(x, y): return self.forward_pass(x, y).item()
            loss = np.mean(list(itertools.starmap(
                validation_iter, validation_loader)))
        self.train()  # Activate training mode
        return loss

    def create_checkpoint(self, epoch_id: int):
        """Create 'checkpoints' of the model
        Write .ckpt files containing the model's parameters at the end of each epoch.

        Use the 'Ask for forgiveness, not permission' principle to improve performances and
        create a folder if needed (which should happen once at most per execution).

        Args:
            epoch_id (int): Number of the completed epoch (numbering starts at 1).
        """
        path = os.path.join('checkpoints', f'model_{epoch_id}.ckpt')
        try:
            torch.save(self.state_dict(), path)
        except FileNotFoundError:
            os.mkdir('checkpoints')
            torch.save(self.state_dict(), path)

    def test(self, test_loader: DataLoader):
        """Test the model through a testing set.

        Switch on the evaluation mode, before computing the accuracy of the model.
        Print the results, and write them to a text file.

        Args:
            test_loader (DataLoader): DataLoader containing the testing data.
        """
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for mel_spectro, lang in test_loader:
                mel_spectro = mel_spectro.to(DEVICE)
                lang = lang.to(DEVICE)
                outputs = self(mel_spectro)
                _, predicted = torch.max(outputs.data, 1)
                total += lang.size(0)
                correct += (predicted == lang).sum().item()

            str_output = 'Test Accuracy of the model on the {} test mel spectrograms: {} %'.format(total,
                                                                                                   100 * correct / total)
            print(str_output)

            with open("test_result.txt", 'w') as file:
                file.write(str_output)
