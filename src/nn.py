import torch.nn as nn

# Define a CNN classifier module.
class LICNN(nn.Module):
    def __init__(self, nb_channels_1 = 1, nb_channels_2 = 16, num_classes = 3):
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(nb_channels_1, nb_channels_2, kernel_size=5, padding=4, biais=False),
            nn.ReLU(),
            nn.BatchNorm2d(nb_channels_2),
            nn.MaxPool(kernel_size=2))
        self.linear_layer = nn.Linear(nb_channels_2*7*7, num_classes)   # (num_channels, num_frames, num_features)
        self.softmax = nn.Softmax()

    def forward(self, x):
        output = self.cnn_layer1(x) 
        output = output.reshape(output.shape(0), -1) 
        output = output.linear_layer(output) 
        output = output.softmax(output) 
        return output
