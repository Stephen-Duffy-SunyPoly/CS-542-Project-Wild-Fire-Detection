import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights



# Load pre-trained ResNet-50 with default ImageNet weights
res50_weights = ResNet50_Weights.DEFAULT
res50_model = resnet50(weights=res50_weights)

for param in res50_model.parameters():
    param.requires_grad = False

# 640 X 512
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # grab the first few pre-trained layers from resnet
        self.pretrained = nn.Sequential(*list(res50_model.children())[0:5])
        # the result of this first pre-trained layer is 256 x 128 x 160
        # 256 is the number of channels
        self.convolution = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # 32 X 16 X 20

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 20, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.convolution(x)
        x = self.fully_connected(x)
        return x