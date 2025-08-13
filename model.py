import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, input_size=224):  # Default matches test_transform
        super(AnimalClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened size dynamically
        self.flattened_size = self._compute_flattened_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _compute_flattened_size(self, input_size):
        """Pass a dummy tensor through conv layers to get flattened size."""
        with torch.no_grad():
            x = torch.randn(1, 3, input_size, input_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Debug shapes
        # print("Shape before flattening:", x.shape)

        x = x.view(x.size(0), -1)  # Flatten
        # print("Shape after flattening:", x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Test
if __name__ == "__main__":
    model = AnimalClassifier(num_classes=5, input_size=224)
    print(model)
