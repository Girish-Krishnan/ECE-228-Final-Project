import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset class
class SeismicDataset(Dataset):
    def __init__(self, data_pairs):
        self.data = []
        self.targets = []
        for seismic_path, velocity_path in data_pairs:
            seismic = np.load(seismic_path)
            velocity = np.load(velocity_path)
            self.data.append(torch.from_numpy(seismic).float())
            self.targets.append(torch.from_numpy(velocity).float())

        self.data = torch.cat(self.data, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

# Collect data file pairs
data_pairs = [
    ('small_dataset/FlatVel_A/data/data1.npy', 'small_dataset/FlatVel_A/model/model1.npy'),
    ('small_dataset/CurveFault_A/seis2_1_0.npy', 'small_dataset/CurveFault_A/vel2_1_0.npy'),
    ('small_dataset/FlatFault_A/seis2_1_0.npy', 'small_dataset/FlatFault_A/vel2_1_0.npy'),
    ('small_dataset/Style_A/data/data1.npy', 'small_dataset/Style_A/model/model1.npy')
]

dataset = SeismicDataset(data_pairs)

# Split into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * 250 * 17, 512),
            nn.ReLU(),
            nn.Linear(512, 70 * 70),
            nn.Unflatten(1, (1, 70, 70))
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = SimpleCNN().to(device)

# Define loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train MAE: {train_loss:.4f} Val MAE: {val_loss:.4f}")

# Visualization
model.eval()
inputs, targets = next(iter(val_loader))
inputs, targets = inputs.to(device), targets.to(device)
with torch.no_grad():
    predictions = model(inputs)

for i in range(min(3, inputs.size(0))):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(targets[i][0].cpu(), cmap='jet')
    axs[0].set_title('Ground Truth Velocity')
    axs[1].imshow(predictions[i][0].cpu(), cmap='jet')
    axs[1].set_title('Predicted Velocity')
    plt.show()
