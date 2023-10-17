import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Define dataset paths

data_dir = os.path.join(os.getcwd() , "../dataset")

# Define transformations for data preprocessing and augmentation
train_transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets and data loaders
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"), transform=train_transform
)
valid_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"), transform=val_test_transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "test"), transform=val_test_transform
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the pre-trained ViT model
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=len(train_dataset.classes)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # Adjust learning rate

# Learning rate scheduler with warm-up
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, total_steps=len(train_loader) * 20
)

# Training loop
num_epochs = 2
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model.to(device)

best_validation_accuracy = 0.0

model.load_state_dict(torch.load("best_model.pth"))

# Testing the model
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.logits, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()


test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")
