import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pr1 = nn.PReLU()
        self.blk1 = GeneratorBlock(16, 16)
        self.blk2 = GeneratorBlock(16, 16)
        self.blk3 = GeneratorBlock(16, 16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pixelshuffle = nn.PixelShuffle(2)  # (16,H,W)->(4,2H,2W)
        self.conv3 = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=(3, 3), padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=(3, 3), padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=4, out_channels=3, kernel_size=(3, 3), padding=1
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.pr1(x2)
        x4 = x3 + self.blk1(x3)
        x5 = x4 + self.blk2(x4)
        x6 = x5 + self.blk3(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + x3
        x10 = self.pixelshuffle(x9)
        x11 = self.conv3(x10)
        x12 = self.conv4(x11)
        out = self.conv5(x12)
        return out


class GeneratorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.output_dim)
        self.pr1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.output_dim,
            out_channels=self.output_dim,
            padding=1,
            kernel_size=(3, 3),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.pr1(x2)
        out = self.conv2(x3)
        return out
