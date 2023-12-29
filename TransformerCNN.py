import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
from PIL import Image

class GaitSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_id=100): #change back to 100 or whatever later
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.max_id = max_id
        self.subfolders = ['A1_silhouette', 'A2_silhouette', 'A3_silhouette']
        
        # Load data
        self.load_dataset()

    def load_dataset(self):
        # Assuming the folder names are like '00001', '00002', ..., '00100'
        user_id_folders = sorted(next(os.walk(self.root_dir))[1])[:self.max_id]  # Limit to the first 100 IDs

        for user_id in user_id_folders:
            for subfolder in self.subfolders:
                silhouette_dir = os.path.join(self.root_dir, user_id, subfolder)
                print(f'{silhouette_dir}:')
                if os.path.isdir(silhouette_dir):
                    image_files = sorted(os.listdir(silhouette_dir))
                    
                    # Create sequences of 3 images
                    for i in range(0, len(image_files) - 2, 3):  # Step is 3 to get non-overlapping sequences
                        self.data.append([
                            os.path.join(silhouette_dir, image_files[i]),
                            os.path.join(silhouette_dir, image_files[i + 1]),
                            os.path.join(silhouette_dir, image_files[i + 2])
                        ])
                        # Convert user ID to integer label (remove leading zeros)
                        self.labels.append(int(user_id.lstrip('0')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths = self.data[idx]
        images = [Image.open(image_path) for image_path in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        # Stack images along the channel dimension to create a 3-channel image
        stacked_images = torch.cat(images, dim=0)  # Shape: [3, 128, 128]
        label = self.labels[idx] - 1  # Adjust labels to be 0-indexed
        return stacked_images, label

class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_transformer_layers=3):
        super(CNNTransformerClassifier, self).__init__()
        
        self.d_model = d_model

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flattening layer
        self.flatten = nn.Flatten()

        # Linear layer to transform the feature dimension
        self.feature_transform = nn.Linear(64 * 32 * 32, d_model)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))

        # Flatten and transform the feature dimension
        x = self.flatten(x)
        x = self.feature_transform(x)

        # Reshape for transformer encoder: [batch size, sequence length, features]
        x = x.view(x.size(0), -1, self.d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Classification head
        x = self.fc(x[:, 0, :])  # Use the output corresponding to the first token
        return x


# Assume that 'num_classes' is the total number of classes you have
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNTransformerClassifier(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

transform = Compose([ToTensor()])
root_dir = r'C:\Users\Win\Desktop\GaitCO\processed_images\00001-04000'
dataset = GaitSequenceDataset(root_dir=root_dir, transform=transform)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# Checking the training data:
# Assuming 'dataset' is your GaitSequenceDataset instance
total_data_points = len(dataset)
print(f"Total data points: {total_data_points}")
number_classes = len(set(dataset.labels))
print(f"Number of classes (user id): {number_classes}")

# # Now, to find out the number of data points per class, we can use a Counter
# from collections import Counter
# label_counts = Counter(dataset.labels)
# print(f"Number of data points per class: {label_counts}")

# # If you've split your dataset into training and testing datasets:
# train_label_counts = Counter([label for _, label in train_dataset])
# print(f"Number of training data points per class: {train_label_counts}")


# Training loop
print('Start Training Simple CNN')
# Training loop with device assignment for data and metrics calculation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        # Move data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total_predictions += labels.size(0)

        # Total correct predictions
        correct_predictions += (predicted == labels).sum().item()

    # Calculate accuracy by dividing correct predictions by total predictions
    accuracy = (correct_predictions / total_predictions) * 100

    # Print statistics
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%')

# Testing loop
model.eval()  # Set the model to evaluation mode
y_true = []
y_pred = []
with torch.no_grad():  # No need to track the gradients
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(f'{i}: y: {labels} --> y_pred: {predicted}')
        y_true += labels.tolist()
        y_pred += predicted.tolist()

# Calculate metrics using true labels and predictions
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Testing Metrics: Accuracy: {accuracy*100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')