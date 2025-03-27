import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import AnimalClassifier  # Import Correct Model
from torchvision import datasets

# Load dataset to check class-to-index mapping
dataset = datasets.ImageFolder(root="./data/images")  # Change this to your actual dataset path
print("Class-to-index mapping:", dataset.class_to_idx)

# Define Class Labels
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog']  # Update this list

# Set Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = AnimalClassifier(num_classes=len(classes)).to(device)  # Match number of classes
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define Image Transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Higher resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to Show Image with Prediction
def show_image(img, label):
    img = img * 0.5 + 0.5  # Correct unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert tensor to image
    plt.title(f"Predicted: {label}")
    plt.axis("off")
    plt.show()

# Function to Detect Objects
def detect_objects():
    # Define DataLoader
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=2
    )
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # Add detection logic here

# Run Detection
if __name__ == "__main__":
    detect_objects()
