import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Input assumed to be 256x256 with 3 channels.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces HxW by half.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # After two poolings: 256 -> 128 -> 64.
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 256, 256)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch_size, 16, 128, 128)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch_size, 32, 64, 64)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Quick test to ensure the model runs.
    model = SimpleCNN(num_classes=10)
    print(model)