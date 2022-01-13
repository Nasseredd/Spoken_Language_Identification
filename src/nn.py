import torch.nn as nn
import torch


# Define a CNN classifier module.
class LICNN(nn.Module):

    def __init__(self, nb_channels_1: int = 1, nb_channels_2: int = 1, nb_classes: int = 3):
        """[summary]

        Args:
            nb_channels_1 (int, optional): [description]. Defaults to 1.
            nb_channels_2 (int, optional): [description]. Defaults to 1.
            num_classes (int, optional): [description]. Defaults to 3.
        """
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(nb_channels_1, nb_channels_2,
                      kernel_size=5, padding=4, biais=False),
            nn.ReLU(),
            nn.BatchNorm2d(nb_channels_2),
            nn.MaxPool2d(kernel_size=2))
        # (num_channels, num_frames, num_features)
        self.linear_layer = nn.Linear(nb_channels_2, nb_classes)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        """[summary]

        Args:
            x (torch.Tensor): Input to the CNN

        Returns:
            torch.Tensor: Output of the CNN
        """
        output = self.cnn_layer1(x)
        output = output.reshape(output.shape(0), -1)
        output = output.linear_layer(output)
        output = output.softmax(output)
        return output
