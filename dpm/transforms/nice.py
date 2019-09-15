import torch
import torch.nn as nn
from .transform import Transform


class NICE(Transform):

    def __init__(self, in_shape=1, num_layers=4,
                 hidden_size=100, num_hidden_layers=3):
        super().__init__()
        assert num_layers % 2 == 0, "Only Handles Even Num Layers"
        self.in_shape = in_shape
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if num_hidden_layers == 0:
                self.layers.append(nn.Sequential(
                    nn.Linear(self.in_shape // 2, self.in_shape // 2)
                ))
            else:
                hidden_layers = [nn.Linear(self.in_shape // 2, hidden_size), nn.ReLU()]
                for _ in range(num_hidden_layers - 1):
                    hidden_layers.append(nn.Linear(hidden_size, hidden_size))
                    hidden_layers.append(nn.ReLU())
                hidden_layers.append(nn.Linear(hidden_size, self.in_shape // 2))

                self.layers.append(nn.Sequential(*hidden_layers))

        self.scaling_factor = nn.Parameter(torch.zeros(in_shape))


    def forward(self, x):
        x_1, x_2 = torch.chunk(x, 2, dim=-1)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x_1 = x_1 + layer(x_2)
            else:
                x_2 = x_2 + layer(x_1)

        out = torch.cat((x_1, x_2), dim=-1)
        out = self.scaling_factor.exp() * out
        return out

    def inverse(self, y):
        y = y / self.scaling_factor.exp()
        y_1, y_2 = torch.chunk(y, 2, dim=-1)

        for i, layer in enumerate(reversed(self.layers)):
            if i % 2 == 0:
                y_1 = y_1 - layer(y_2)
            else:
                y_2 = y_2 - layer(y_1)

        out = torch.cat((y_1, y_2), dim=-1)
        return out

    def log_abs_det_jacobian(self, x, y):
        return self.scaling_factor.sum(-1)
