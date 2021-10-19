import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()

        """ Convolutional Layers """
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=6, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16, kernel_size=(3, 3), padding=1)

        """ Fully Connected Layers """
        self.fc_1 = nn.Linear(56 * 56 * 16, 128)
        self.fc_2 = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32, output_dim)

    def forward(self, x):
        """ Convolution """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.reshape(x, (-1, 56 * 56 * 16))
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        return self.fc_3(x)
