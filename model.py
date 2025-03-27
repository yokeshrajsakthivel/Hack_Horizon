import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, input_size=64):
        super(AnimalClassifier, self).__init__()

        # 🔹 Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 🔹 Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 🔹 Compute Flattened Size Automatically
        self.flattened_size = self._compute_flattened_size(input_size)

        # 🔹 Fully Connected Layers
        self.fc1 = nn.Linear(802816, 256)  # Change 100352 → 802816 # Replace 100352 with your actual shape
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _compute_flattened_size(self, input_size):
        """Pass a dummy tensor through conv layers to get flattened size."""
        with torch.no_grad():
            x = torch.randn(1, 3, input_size, input_size)  # Dummy input
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).size(1)  # Get the flattened size dynamically

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        print("Shape before flattening:", x.shape)  # Debugging line

        x = x.view(x.size(0), -1)  # Flatten
        print("Shape after flattening:", x.shape)  # Debugging line

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 🔹 Test Model Initialization
if __name__ == "__main__":
    num_classes = 5  # Set this based on your dataset
    model = AnimalClassifier(num_classes=num_classes, input_size=64)  # Adjust input_size if needed
    print(model)
