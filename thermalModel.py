import torch
from torch import nn
from torchvision.transforms import v2

# 640 X 512
class ThermalModel(nn.Module):
    def __init__(self):
        super(ThermalModel, self).__init__()



        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding='same')
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding='same')

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding='same')

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding='same')

        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding='same')

        self.fc1 = nn.Linear(64*64*80, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()


    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool(x)

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.pool(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x