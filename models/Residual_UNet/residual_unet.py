"""
Train the Residual UNet on the small OpenFWI subset

Requirements
------------
pip install torch torchvision tqdm
"""

import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------
class SeismicDataset(Dataset):
    """Pairs each seismic cube with its velocity map"""

    def __init__(self, pairs):
        cubes, maps_ = [], []
        for seis_path, vel_path in pairs:
            seis = np.load(seis_path).astype(np.float32)           # (N, 5, 1000, 70)
            vel  = np.load(vel_path ).astype(np.float32)           # (N, 70, 70) or (N, 1, 70, 70)
            if vel.ndim == 3:
                vel = vel[:, None, :, :]                           # add channel if absent
            cubes.append(torch.from_numpy(seis))
            maps_.append(torch.from_numpy(vel))
        self.seis   = torch.cat(cubes, dim=0)
        self.velmap = torch.cat(maps_, dim=0)
        # Normalize velocity maps to roughly [-1, 1]
        self.velmap = torch.clamp((self.velmap - 1500.0) / 1500.0, -1.0, 1.0)
        self.velmap = self.velmap * 1500.0 + 1500.0  # Rescale back to original range

    def __len__(self):
        return self.seis.shape[0]

    def __getitem__(self, idx):
        return self.seis[idx], self.velmap[idx]

def gather_pairs(data_root):
    pairs = []
    # Vel & Style groups
    for fam in ["FlatVel_A", "Style_A"]:
        pattern = os.path.join(data_root, fam, "data", "*.npy")
        for sp in sorted(glob.glob(pattern)):
            mp = sp.replace("data", "model")
            if os.path.exists(mp):
                pairs.append((sp, mp))
    # Fault groups
    for fam in ["CurveFault_A", "FlatFault_A"]:
        pattern = os.path.join(data_root, fam, "seis*_*.npy")
        for sp in sorted(glob.glob(pattern)):
            mp = os.path.join(data_root, fam,
                              os.path.basename(sp).replace("seis", "vel"))
            if os.path.exists(mp):
                pairs.append((sp, mp))
    return pairs

# ----------------------------------------------------------------------
# Model definition (copied from the prompt)
# ----------------------------------------------------------------------
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualDoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=1,
                 init_features=32, depth=5, bilinear=True):
        super().__init__()
        self.depth = depth
        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))
        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        curr = init_features
        for _ in range(depth):
            self.encoder_convs.append(ResidualDoubleConv(curr, curr * 2))
            self.encoder_pools.append(nn.MaxPool2d(2))
            curr *= 2
        self.bottleneck = ResidualDoubleConv(curr, curr)
        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            self.decoder_blocks.append(Up(curr, curr // 2, bilinear))
            curr //= 2
        self.outc = OutConv(curr, n_classes)

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        _, _, h, w = x.shape
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))
            h = target_h
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))
            w = target_w
        if h > target_h:
            crop_top = (h - target_h) // 2
            x = x[:, :, crop_top:crop_top + target_h, :]
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left:crop_left + target_w]
        return x

    def forward(self, x):
        x = self.initial_pool(x)
        x = self._pad_or_crop(x)
        skips = []
        x = self.inc(x)
        skips.append(x)
        for i in range(self.depth):
            x = self.encoder_convs[i](x)
            skips.append(x)
            x = self.encoder_pools[i](x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.decoder_blocks):
            skip_idx = self.depth - 1 - i
            x = up(x, skips[skip_idx])
        logits = self.outc(x)
        output = logits * 1500.0 + 1500.0
        return output

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    pairs = gather_pairs(".")
    dataset = SeismicDataset(pairs)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet().to(device)
    
    model.load_state_dict(torch.load("unet_epoch_100.pth", map_location=device))  # Load pre-trained weights if available
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for seismic, vel in loop:
            seismic, vel = seismic.to(device), vel.to(device)
            optimizer.zero_grad()
            preds = model(seismic)
            loss = criterion(preds, vel)
            loss.backward()
            optimizer.step()
            running += loss.item() * seismic.size(0)
            loop.set_postfix(train_mae=loss.item())
        train_mae = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for seismic, vel in val_loader:
                seismic, vel = seismic.to(device), vel.to(device)
                preds = model(seismic)
                val_running += criterion(preds, vel).item() * seismic.size(0)
        val_mae = val_running / len(val_loader.dataset)
        print(f"Epoch {epoch:2d} | Train MAE {train_mae:.4f} | Val MAE {val_mae:.4f}")

        # Save model weights every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"unet_epoch_{epoch}.pth")

    # # quick visual check
    model.eval()
    #
    idxs = [0, 500, 1000, 1500, 999]
    seis_sample = dataset.seis[idxs].to(device)
    vel_sample = dataset.velmap[idxs].to(device)
    preds = model(seis_sample)
    preds = preds.cpu().detach().numpy()
    vel_sample = vel_sample.cpu().detach().numpy()

    for i in range(len(idxs)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(vel_sample[i, 0], cmap="jet")
        ax[0].set_title("Ground truth (m/s)")
        ax[0].axis("off")

        preds = torch.clamp((torch.tensor(preds) - 1500.0) / 1500.0, -1.0, 1.0)
        preds = preds * 1500.0 + 1500.0
        preds = preds.cpu().detach().numpy()

        im1 = ax[1].imshow(preds[i, 0], cmap="jet")
        ax[1].set_title("Prediction (m/s)")
        ax[1].axis("off")

        plt.tight_layout()
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        plt.show()


if __name__ == "__main__":
    train()