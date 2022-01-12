import torch
import torch.nn as nn
import numpy as np

# Define a CNN classifier module.

class CNNClassif(nn.Module):
    def __init__(self, num_channels1=16, num_channels2=32, num_classes=10):
        super(CNNClassif, self).__init__()

        self.cnn_layer1 = nn.Sequential(nn.Conv2d(1, num_channels1, kernel_size=5, padding=2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=5, padding=2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))
        self.cnn_layer3 = nn.Linear(num_channels2*7*7, num_classes)

    def forward(self, x):

        # TO DO: write the forward pass, which:
        # - applies the two cnn layers to produce feature maps
        a1 = self.cnn_layer1(x)
        a2 = self.cnn_layer2(a1)
        # - vectorize the feature maps
        out_vec = a2.reshape(a2.shape[0], -1)
        # - applies the linear layer
        out = self.cnn_layer3(out_vec)

        return out
