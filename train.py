import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
<<<<<<< HEAD
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import AnimalClassifier  # Ensure this is correctly imported

# 🔹 Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Define Hyperparameters
num_classes = 5  # Update this based on your dataset
batch_size = 32
epochs = 10
learning_rate = 0.001

# 🔹 Data Preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Increase resolution
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Ensure 3-channel normalization
])


# 🔹 Load Custom Dataset (✅ **Fix: Use `train_transform` instead of `transform`**)
train_dataset = datasets.ImageFolder(root="data/images", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 🔹 Initialize Model and Move to GPU
model = AnimalClassifier(num_classes=num_classes).to(device)

# 🔹 Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🔹 Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 🔹 Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model training complete! 🚀")
=======
from model import SimpleCNN
from secure_layer import ModelSecurity  # Import security layer

# ... [previous imports and setup code remains the same until model initialization]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🔹 Initialize Model with Security
model = SimpleCNN().to(device)
secure_trainer = ModelSecurity(model)  # Security wrapper

# 🔹 Enhanced training loop with data validation
def train_model(epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        valid_samples = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # Validate training data
            if not secure_trainer.data_validator.validate(images):
                continue
                
            # Standard training procedure
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            valid_samples += 1
        
        # Verify model integrity after each epoch
        if not secure_trainer.verify_model_integrity():
            raise SecurityException("Model compromised during training!")
    
    # Save model with integrity check
    torch.save(model.state_dict(), "model.pth")

# ... [rest of the file remains the same]
>>>>>>> a670ca125c792f78ebc6d73c1150ea222d664cf5
