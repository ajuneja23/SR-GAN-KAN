import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from KANLayer import KANLayer


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.lrelu1 = nn.LeakyReLU()
        self.blk1 = DiscriminatorBlock(16, 16)
        self.blk2 = DiscriminatorBlock(16, 16)
        self.blk3 = DiscriminatorBlock(16, 16)
        self.flatten = nn.Flatten()  # 16*H*w assume 16*28*28
        self.kan1 = KANLayer(
            in_features=16 * input_dim[0] * input_dim[1], out_features=1024
        )
        self.kan2 = KANLayer(in_features=1024, out_features=1024)
        self.kan3 = KANLayer(in_features=1024, out_features=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.lrelu1(x1)
        x3 = self.blk1(x2)
        x4 = self.blk2(x3)
        x5 = self.blk3(x4)
        x6 = self.flatten(x5)
        x7 = self.kan1(x6)
        x8 = self.kan2(x7)
        x9 = self.kan3(x8)
        out = x9.item()
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiscriminatorBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.lrelu1 = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        out = self.lrelu1(x2)
        return out
