import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN
from secure_layer import ModelSecurity  # Import the security layer

# 🔹 CIFAR-10 Classes
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 🔹 Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load Model with Security Wrapper
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Initialize security layer
secure_model = ModelSecurity(model)

# 🔹 Load Test Dataset with additional security checks
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.clamp(x, -1, 1))  # Added security
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# 🔹 Enhanced detection function with security
def detect_objects():
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    try:
        # Use secured prediction
        images, labels = images.to(device), labels.to(device)
        outputs = secure_model.secure_predict(images)
        _, predicted = torch.max(outputs, 1)
        
        prediction = classes[predicted.item()]
        print(f"✅ Secure Prediction: {prediction}")
        
        # Show image with security indicator
        show_image_with_prediction(images.cpu().squeeze(), prediction, secure=True)
        
    except SecurityException as e:
        print(f"🚨 Security Alert: {str(e)}")
        show_image_with_prediction(images.cpu().squeeze(), "BLOCKED", secure=False)

# 🔹 Updated visualization with security status
def show_image_with_prediction(img, prediction, secure=True):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    color = "green" if secure else "red"
    bbox_color = "lime" if secure else "pink"
    
    plt.text(2, 2, f"{prediction}", fontsize=14, color=color,
             bbox=dict(facecolor=bbox_color, alpha=0.8))
    
    status = "SECURE" if secure else "BLOCKED"
    plt.title(f"Status: {status}", color=color)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    detect_objects()