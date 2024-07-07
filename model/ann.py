import torch.nn as nn
import torch.nn.functional as F


class ImageANN(nn.Module):
    def __init__(self):
        super(ImageANN, self).__init__()
        self.fc1 = nn.Linear(250 * 250, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class LandmarkANN(nn.Module):
    def __init__(self):
        super(LandmarkANN, self).__init__()
        self.fc1 = nn.Linear(63, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 30)

    def forward(self, x):
        x = x.view(-1, 21 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
