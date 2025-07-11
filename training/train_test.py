import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import random
import numpy as np

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 5  # No DR, Mild, Moderate, Severe, Proliferative DR

# Transforms
transform = transforms.Compose([
    transforms.Resize((728, 728)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3199, 0.2240, 0.1609], std=[0.3020, 0.2183, 0.1741])
])

# Dataset paths
train_dir = "data/train"
test_dir = "data/test"

# Debug: Check directory access
print("Current Working Directory:", os.getcwd())
print("Train Dir Exists?", os.path.exists(train_dir))
print("Test Dir Exists?", os.path.exists(test_dir))

# Load data
dataset_train = datasets.ImageFolder(train_dir, transform=transform)

# 🔧 SANITY TEST OPTION — Use train data as test data to check for 100% accuracy
# dataset_test = datasets.ImageFolder(train_dir, transform=transform)  # Uncomment for sanity test
dataset_test = datasets.ImageFolder(test_dir, transform=transform)     # Real test

train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Map class indices to human-readable labels
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

# Load pre-trained CNN (ResNet18) and customize last layer
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
print("\nTraining started...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model, "new_model.pth")
print("\nModel saved as new_model.pth")

# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print predicted vs actual for debugging
        for i in range(len(labels)):
            print(f"Predicted: {class_names[predicted[i]]}, Actual: {class_names[labels[i]]}")

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
