'''

TOOLS FOR NONLINEAR MODELS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import numpy as np

import os
import numpy as np
import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt


# %% FEEDFORWARD NETWORK

class feedforward_network(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(feedforward_network, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

# %% CONVOLUTIONAL NETWORK

class conv_network(nn.Module):
    def __init__(self):
        super(conv_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)
        self.linear1 = nn.Linear(1728, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)