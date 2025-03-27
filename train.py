import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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