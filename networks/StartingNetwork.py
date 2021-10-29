import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()

        """ Convolutional Layers """
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels = 96, 
                                kernel_size=(11, 11), stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels = 256, 
                                kernel_size=(5, 5), padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels = 384, 
                                kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels = 384, 
                                kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels = 256, 
                                kernel_size=(3, 3), padding=1)

        """ Fully-Connected Layers """
        self.fc_1 = nn.Linear(4 * 4 * 256, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, 1000)
        self.fc_4 = nn.Linear(1000, output_dim)

        """ Other utility layers """
        self.pool = nn.maxPool2d((3, 3), stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """ Convolution """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))

        """ Fully-Connected """
        x = torch.reshape(x, (-1, 4 * 4 * 256))
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_2(x))
        x = self.dropout(x)
        x = F.relu(self.fc_3(x))

        return self.fc_4(x)
