import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calculate the output size after conv2 and pooling
        self.fc1_input_size = (
            16 * 59 * 59
        )  # Adjust based on actual output size after flattening

        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LandnarkCNN(nn.Module):
    def __init__(self):
        super(LandnarkCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(
            21 * 16, 120
        )  # Adjust input size based on output channels and number of rows (21)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 30)  # Output layer with 30 classes

    def forward(self, x):
        # Input x has shape (batch_size, 21, 3)
        x = x.permute(0, 2, 1)  # Shape becomes (batch_size, 3, 21)

        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 6, 21)
        x = F.relu(self.conv2(x))  # Output shape: (batch_size, 16, 21)

        # Flatten
        x = x.view(-1, 21 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
