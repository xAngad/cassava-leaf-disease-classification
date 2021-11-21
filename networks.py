import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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
        self.fc_1 = nn.Linear(5 * 5 * 256, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, 1000)
        self.fc_4 = nn.Linear(1000, output_dim)

        """ Batch normalization layers """
        self.bn_1 = nn.BatchNorm2d(96)   # after conv1
        self.bn_2 = nn.BatchNorm2d(256)  # after conv2, conv5
        self.bn_3 = nn.BatchNorm2d(384)  # after conv3, conv4
        self.bn_4 = nn.BatchNorm1d(4096) # after fc_1, fc_2
        self.bn_5 = nn.BatchNorm1d(1000) # after fc_3

        """ Other utility layers """
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """ Convolution """
        x = self.pool(F.relu(self.bn_1(self.conv1(x))))
        x = self.pool(F.relu(self.bn_2(self.conv2(x))))
        x = F.relu(self.bn_3(self.conv3(x)))
        x = F.relu(self.bn_3(self.conv4(x)))
        x = self.pool(F.relu(self.bn_2(self.conv5(x))))

        """ Fully-Connected """
        x = torch.reshape(x, (-1, 5 * 5 * 256))
        x = F.relu(self.bn_4(self.fc_1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_4(self.fc_2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_5(self.fc_3(x)))

        return self.fc_4(x)

class TransferNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()

        """ Transfer from ResNet """
        # self.model_a = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained = True)
        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.model_a = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        """ Fully-Connected Layer """
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        """ ResNet """
        with torch.no_grad():
            features = self.model_a(x)

        """ Fully-Connected """
        features = torch.reshape(features, (-1, 512))
        prediction = self.fc(features)

        return prediction

class BuffedTransferNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels = 3, 
                                kernel_size=(3, 3), stride = 2)

        """ Transfer from ResNet """
        # self.model_a = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained = True)
        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.model_a = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        """ Fully-Connected Layer """
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        """ Single Convolutional Layer """
        x = self.conv(x)

        """ ResNet """
        features = self.model_a(x)

        """ Fully-Connected """
        features = torch.reshape(features, (-1, 512))
        prediction = self.fc(features)

        return prediction


# if __name__ == "__main__":
#     model = StartingNetwork(3, 5)
#     print(model)
