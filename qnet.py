import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.bc1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.bc1(self.fc1(state)))
        x = F.relu(self.bc2(self.fc2(x)))
        return self.fc3(x)