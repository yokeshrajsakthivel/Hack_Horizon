import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import AnimalClassifier

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Hyperparameters
batch_size = 32
epochs = 10
learning_rate = 0.001
input_size = 224  # match the transform size

# 🔹 Transforms
train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 🔹 Dataset
train_dataset = datasets.ImageFolder(root="data/images", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 🔹 Number of classes (from dataset)
num_classes = len(train_dataset.classes)
print(f"Classes found: {train_dataset.classes}")

# 🔹 Model
model = AnimalClassifier(num_classes=num_classes, input_size=input_size).to(device)

# 🔹 Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🔹 Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 🔹 Save Model
torch.save(model.state_dict(), "model.pth")
print("✅ Model training complete and saved as model.pth")
