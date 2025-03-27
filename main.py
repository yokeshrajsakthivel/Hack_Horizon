import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
from model import AnimalClassifier  # Import Correct Model
from torchvision import datasets
=======
from model import SimpleCNN
from secure_layer import ModelSecurity  # Import the security layer
>>>>>>> a670ca125c792f78ebc6d73c1150ea222d664cf5

# Load dataset to check class-to-index mapping
dataset = datasets.ImageFolder(root="./data/images")  # Change this to your actual dataset path
print("Class-to-index mapping:", dataset.class_to_idx)

# Define Class Labels
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog']  # Update this list

# Set Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

<<<<<<< HEAD
# Load Model
model = AnimalClassifier(num_classes=len(classes)).to(device)  # Match number of classes
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define Image Transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Higher resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
=======
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
>>>>>>> a670ca125c792f78ebc6d73c1150ea222d664cf5
])


# Load Custom Dataset
testset = torchvision.datasets.ImageFolder(root="./data/images", transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

<<<<<<< HEAD
# Function to Show Image with Prediction
def show_image(img, label):
    img = img * 0.5 + 0.5  # Correct unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert tensor to image
    plt.title(f"Predicted: {label}")
    plt.axis("off")
    plt.show()


# Function to Detect Objects
=======
# 🔹 Enhanced detection function with security
>>>>>>> a670ca125c792f78ebc6d73c1150ea222d664cf5
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

<<<<<<< HEAD
    # Get Predicted Class Name
    prediction = classes[predicted.item()]

    # Print Prediction in Terminal
    print(f"Predicted Object: {prediction}")

    # Show Image with Prediction
    show_image(images.cpu().squeeze(), prediction)

# Run Detection
if __name__ == "__main__":
    detect_objects()
=======
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
>>>>>>> a670ca125c792f78ebc6d73c1150ea222d664cf5
