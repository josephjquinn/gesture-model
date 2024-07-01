import torch.nn as nn
import torch.nn.functional as F


class LandmarkNN(nn.Module):
    def __init__(self):
        super(LandmarkNN, self).__init__()
        self.fc1 = nn.Linear(63, 256)  # 21 * 3 = 63 21 rows of 3 dims
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)  # Output layer 3 classes

    def forward(self, x):
        x = x.view(-1, 21 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
