from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 fcs1_units: int = 400,
                 fc2_units: int = 300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.linear1 = nn.Linear(state_size, fcs1_units)
        self.batch1 = nn.BatchNorm1d(fcs1_units)

        self.linear2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.batch2 = nn.BatchNorm1d(fc2_units)

        self.linear3 = nn.Linear(fc2_units, 1)
        self.batch3 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear1.weight.data.uniform_(*self.hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*self.hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-1e-3, 1e-3)
        self.linear1.bias.data.fill_(0.5)
        self.linear2.bias.data.fill_(0.5)
        self.linear3.bias.data.fill_(0.5)

    def hidden_init(self, layer: nn.Module) -> Tuple[float, float]:
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = state
        x = F.relu(self.batch1(self.linear1(x)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.batch2(self.linear2(x)))
        x = self.linear3(x)
        return x
