import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from model import AnimalClassifier
from torchvision import datasets
from secure_layer import ModelSecurity  # import secure layer

# Define Image Transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset with transform applied
dataset = datasets.ImageFolder(root="./data/images", transform=test_transform)
print("Class-to-index mapping:", dataset.class_to_idx)

# Define Class Labels
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog']

# Set Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = AnimalClassifier(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Wrap model in security layer
security = ModelSecurity(model)

# Function to Show Image with Prediction
def show_image(img, label, secure_status, reason):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f"Predicted: {label}\nSecurity: {secure_status} ({reason})")
    plt.axis("off")

# Function to Detect Objects (random image, quit with q)
def detect_objects():
    indices = list(range(len(dataset)))
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(6, 6))  # keep panel size stable

    while True:
        idx = random.choice(indices)
        image, _ = dataset[idx]
        image = image.to(device)

        # Secure prediction
        outputs, secure_status, reason = security.secure_predict(image)
        if outputs is not None:
            _, predicted = torch.max(outputs, 1)
            predicted_label = classes[predicted[0]]
        else:
            predicted_label = "N/A"

        plt.clf()
        show_image(image.cpu(), predicted_label, secure_status, reason)
        plt.draw()

        print(f"Prediction: {predicted_label}")
        print(f"Security: {secure_status} — {reason}")
        print("Press 'q' in the figure window to quit, or any other key to continue...")

        key = plt.waitforbuttonpress()

        # If user closes the window, break
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.close(fig)

# Run Detection
if __name__ == "__main__":
    detect_objects()
