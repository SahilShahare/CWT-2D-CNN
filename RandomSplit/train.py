import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import sys
from datetime import datetime

os.makedirs('./logs', exist_ok=True)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H:%M:%S")

log_file = f"./logs/{timestamp}.txt"

def log_print(msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

seed = 42
generator = torch.Generator().manual_seed(seed)


class AFDetectionCNN(nn.Module):
    def __init__(self):
        super(AFDetectionCNN, self).__init__()
        # Four convolutional layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=10)    # Input: (batch, 5, 128, 128)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=10)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=8)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4)
        
        # Two max pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(15488, 256)
        self.fc2 = nn.Linear(256, 1)  # Binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = self.dropout2(x)
        
        x = torch.flatten(x,1)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc2(x))
        return x
    
class AFBeatDataset(Dataset):
    def __init__(self, X_path, Y_path, transform=None):
        self.X = np.load(X_path)            # shape: (N, 5, 128, 128)
        self.Y = np.load(Y_path)            # shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)  # shape (5, 128, 128)
        y = self.Y[idx].astype(np.float32)  # for BCELoss or BCEWithLogitsLoss

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x)  # shape (5, 128, 128)
        y = torch.tensor(y)  # shape ()

        return x, y

X_path = 'X.npy'
Y_path = 'y.npy'

# Instantiate dataset
dataset = AFBeatDataset(X_path, Y_path)

labels = np.load('y.npy')

# Get indices
indices = np.arange(len(labels))

# Stratified split
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"Using device: {device}")

# Instantiate and move model to device
model = AFDetectionCNN().to(device)


# ---------------------------
# Model, Loss, Optimizer
# ---------------------------
model = AFDetectionCNN().to(device)
criterion = nn.BCELoss()

# Hyperparameters from the paper
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=1e-6)

# ---------------------------
# Training Loop
# ---------------------------

log_print("Training:")
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        TP += ((preds == 1) & (labels == 1)).sum().item()
        TN += ((preds == 0) & (labels == 0)).sum().item()
        FP += ((preds == 1) & (labels == 0)).sum().item()
        FN += ((preds == 0) & (labels == 1)).sum().item()

    train_acc = 100 * correct / total
    train_sens = TP / (TP + FN + 1e-6)
    train_spec = TN / (TN + FP + 1e-6)

    log_print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Sens: {train_sens:.4f} | "
          f"Spec: {train_spec:.4f}")
    train_cm = np.array([[TN, FP],
                     [FN, TP]])
    log_print("Train Confusion Matrix:")
    log_print(str(train_cm))

# ---------------------------
# Evaluation
# ---------------------------

log_print("Testing:")
 
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    acc = 100 * correct / total
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    cm = np.array([[TN, FP],
                     [FN, TP]])
    

    return acc, sensitivity, specificity, cm

# Final evaluation
val_acc, val_sens, val_spec, cm = evaluate(model, val_loader)

log_print("\n Final Evaluation on Validation Set:")
log_print(f"Accuracy     : {val_acc:.2f}%")
log_print(f"Sensitivity  : {val_sens:.4f}")
log_print(f"Specificity  : {val_spec:.4f}")
log_print("Test Confusion Matrix:")
log_print(str(cm))