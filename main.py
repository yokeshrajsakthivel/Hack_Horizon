import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN

# 🔹 CIFAR-10 Classes
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 🔹 Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load Model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# 🔹 Load Test Dataset (Batch size = 1 for single image processing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# 🔹 Function to Show Image with Prediction
def show_image_with_prediction(img, prediction):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Show Predictions on Image
    plt.text(2, 2, f"{prediction}", fontsize=14, color="red", 
             bbox=dict(facecolor='white', alpha=0))

    plt.axis('off')  # Hide axes
    plt.show()  # Show image

# 🔹 Function to Detect Objects
def detect_objects():
    dataiter = iter(testloader)
    images, labels = next(dataiter)  # Load one image

    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Convert prediction to class name
    prediction = classes[predicted.item()]

    # Print Prediction in Terminal
    print(f"Predicted Object: {prediction}")

    # Show image with prediction
    show_image_with_prediction(images.cpu().squeeze(), prediction)

# 🔹 Run Detection
if __name__ == "__main__":
    detect_objects()

