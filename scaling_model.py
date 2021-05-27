import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "linear": lambda x: x}


class FinalResNet(nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
        delta_type,  # 'none', 'shared', 'multi',
        initial_sd,
        activation,  # 'relu', 'tanh'
    ):
        """Generic fully configurable version of 1D ResNet.
        Update rule:
            x[l+1] = x[l] + delta[l] * sigma(A[l] * x[l] + b[l])
        """

        super(FinalResNet, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.delta_type = delta_type  # 'none', 'shared', 'multi'
        self.sigma = ACTIVATIONS[activation]  # 'relu', 'tanh'
        self.initial_sd = initial_sd

        self.A = [nn.Parameter(torch.Tensor(dim, dim)) for _ in range(num_layers)]
        self.b = [nn.Parameter(torch.Tensor(dim)) for _ in range(num_layers)]

        if delta_type == "shared":
            self.delta = nn.Parameter(torch.Tensor(1))
        elif delta_type == "multi":
            self.delta = [nn.Parameter(torch.Tensor(1)) for _ in range(num_layers)]
        else:
            self.delta = 1.0

    def forward(self, x):
        for l in range(self.num_layers):
            x = x + self.update_rule(x, l)
        return x

    def update_rule(self, x, l):
        h = self.sigma(torch.mm(x, self.A[l]) + self.b[l])
        if self.delta_type == "shared":
            return torch.abs(self.delta) * h
        elif self.delta_type == "multi":
            return self.delta[l] * h
        else:
            return h

    def init_values(self, method="xavier"):
        L = self.num_layers
        dim = self.dim
        if self.delta_type == "shared":
            self.delta.data = torch.FloatTensor([1 / L])
        elif self.delta_type == "multi":
            for l in range(L):
                self.delta[l].data.normal_(0, 1.0 / L)

        if method == "xavier":
            for l in range(L):
                self.A[l].data.normal_(0, self.initial_sd / dim)
                self.b[l].data.normal_(0, self.initial_sd / np.sqrt(dim))

        elif method == "xavier-depth":
            for l in range(L):
                self.A[l].data.normal_(0, self.initial_sd / (L * dim))
                self.b[l].data.normal_(0, self.initial_sd / (L * np.sqrt(dim)))

        else:
            raise ValueError(
                f"Unknown method {method}. Allowed: 'xavier' and 'xavier-depth'."
            )

    def init_from_data(self, A, b, delta):
        """A: [(d, d)], b: [(d,)], delta: [()]."""
        if self.delta_type == "shared":
            self.delta.data = torch.FloatTensor(delta)
        for l in range(self.num_layers):
            self.A[l].data = torch.FloatTensor(A[l])
            self.b[l].data = torch.FloatTensor(b[l])
            if self.delta_type == "multi":
                self.delta[l].data = torch.FloatTensor(delta[l])
