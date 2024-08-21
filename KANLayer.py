import torch
import torch.nn as nn

"""
Implementation Credit to Original Paper PyKan Repo and Association of Data Scientists
https://adasci.org/revolutionizing-language-models-with-kan-a-deep-dive/
https://github.com/KindXiaoming/pykan 

"""


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_points=10):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_points = grid_points
        self.control_points = nn.Parameter(torch.randn(out_features, grid_points))

    def bspline(self, x):
        basis = torch.zeros(x.size(0), self.grid_points)
        for i in range(self.grid_points):
            basis[:, i] = torch.exp(-1 * ((x - i) ** 2))
        return basis

    def forward(self, x):
        b_spline_basis = self.bspline(x)
        return torch.matmul(b_spline_basis, self.control_points)
