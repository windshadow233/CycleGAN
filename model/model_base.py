import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        convs = [
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.Sequential(*convs)

    def forward(self, x):
        return x + self.layers(x)

