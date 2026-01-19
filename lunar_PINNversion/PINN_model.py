import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.optim.lr_scheduler import ExponentialLR

if torch.cuda.is_available():
    device = torch.device("cuda") # Select GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu") # Fallback to CPU
    print("Using CPU")

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_frequencies=4,
        base_freq=0.98,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = base_freq ** torch.arange(num_frequencies)
        print(f"freq bands are: {freq_bands}")
        self.register_buffer("freq_bands", freq_bands)

    def forward(
        self,
        x,
    ):
        """
        x: (N, 3)
        """
        enc = [x]
        for f in self.freq_bands:
            enc.append(torch.sin(2 * np.pi * f * x))
            enc.append(torch.cos(2 * np.pi * f * x))
        return torch.cat(enc, dim=-1)

    def out_dim(self):
        return self.in_dim * (1 + 2 * self.num_frequencies)

    def out_dim(self):
        return self.in_dim * (1 + 2 * self.num_frequencies)


class PINN(nn.Module):
    def __init__(
        self,
        pe_num_freqs=6,
        base_freq=1.3,
        layers=[64, 128, 128, 128],
    ):
        super().__init__()
        # in_dim = 3

        self.pe = PositionalEncoding(
            in_dim=3,
            num_frequencies=pe_num_freqs,
            base_freq=base_freq,
        )

        in_dim = self.pe.out_dim()

        layer_dims = [in_dim] + layers + [1]

        net = []
        for i in range(len(layer_dims) - 1):
            net.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(
        self,
        xyz,
    ):
        xyz_pe = self.pe.forward(xyz)
        return self.net(xyz_pe)
        # return self.net(xyz)

    def laplacian_phi(model, xyz):
        xyz.requires_grad_(True)
        phi = model(xyz)

        grad = torch.autograd.grad(
            phi, xyz, torch.ones_like(phi), create_graph=True,
        )[0]

        phix, phiy, phiz = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]

        phixx = torch.autograd.grad(
            phix, xyz, torch.ones_like(phix), create_graph=True,
        )[0][:, 0:1]

        phiyy = torch.autograd.grad(
            phiy, xyz, torch.ones_like(phiy), create_graph=True,
        )[0][:, 1:2]

        phizz = torch.autograd.grad(
            phiz, xyz, torch.ones_like(phiz), create_graph=True,
        )[0][:, 2:3]

        return phixx + phiyy + phizz
