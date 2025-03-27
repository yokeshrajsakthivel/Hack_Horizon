import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 🔹 Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 🔹 Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 🔹 Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 128 channels from conv3, image size reduced to 4x4
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 output classes (CIFAR-10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pooling
        
        x = x.view(-1, 128 * 4 * 4)  # Flatten for FC layers
        x = F.relu(self.fc1(x))  # Fully Connected 1 + ReLU
        x = F.relu(self.fc2(x))  # Fully Connected 2 + ReLU
        x = self.fc3(x)  # Fully Connected 3 (Output)
        
        return x

# 🔹 Test Model Initialization
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)  # Print model architecture
