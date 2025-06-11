# train_resunet.py

import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ---------------------- Dataset ---------------------- #
class SeismicDataset(Dataset):
    def __init__(self, pairs):
        cubes, maps_ = [], []
        for seis_path, vel_path in pairs:
            seis = np.load(seis_path).astype(np.float32)
            vel  = np.load(vel_path ).astype(np.float32)
            if vel.ndim == 3:
                vel = vel[:, None, :, :]
            cubes.append(torch.from_numpy(seis))
            maps_.append(torch.from_numpy(vel))
        self.seis   = torch.cat(cubes, dim=0)
        self.velmap = torch.clamp((torch.cat(maps_, dim=0) - 1500.0) / 1500.0, -1.0, 1.0)

    def __len__(self): return len(self.seis)
    def __getitem__(self, idx): return self.seis[idx], self.velmap[idx]

def gather_pairs(root):
    pairs = []
    for fam in ["FlatVel_A", "Style_A"]:
        for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
            mp = sp.replace("data", "model")
            if os.path.exists(mp):
                pairs.append((sp, mp))
    for fam in ["CurveFault_A", "FlatFault_A"]:
        for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
            mp = os.path.join(root, fam, os.path.basename(sp).replace("seis", "vel"))
            if os.path.exists(mp):
                pairs.append((sp, mp))
    return pairs

# ---------------------- Model ------------------------ #
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = ResidualDoubleConv(in_ch + out_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = ResidualDoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32, depth=5):
        super().__init__()
        self.initial_pool = nn.AvgPool2d((14, 1), stride=(14, 1))
        self.inc = ResidualDoubleConv(in_ch, base)
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = base
        for _ in range(depth):
            self.enc.append(ResidualDoubleConv(c, c * 2))
            self.pool.append(nn.MaxPool2d(2))
            c *= 2
        self.bottleneck = ResidualDoubleConv(c, c)
        self.dec = nn.ModuleList()
        for _ in range(depth):
            self.dec.append(Up(c, c // 2))
            c //= 2
        self.outc = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        x = self.initial_pool(x)
        x = self._pad_or_crop(x)
        skips = [self.inc(x)]
        x = skips[-1]
        for conv, pool in zip(self.enc, self.pool):
            x = conv(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.dec):
            x = up(x, skips[-(i + 2)])
        return self.outc(x) * 1500.0 + 1500.0

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        _, _, h, w = x.shape
        if h < target_h:
            pad = [(target_h - h) // 2, target_h - h - (target_h - h) // 2]
            x = F.pad(x, (0, 0, *pad))
        if w < target_w:
            pad = [(target_w - w) // 2, target_w - w - (target_w - w) // 2]
            x = F.pad(x, (*pad, 0, 0))
        if h > target_h:
            crop = (h - target_h) // 2
            x = x[:, :, crop:crop + target_h, :]
        if w > target_w:
            crop = (w - target_w) // 2
            x = x[:, :, :, crop:crop + target_w]
        return x

# -------------------- Training ----------------------- #
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = gather_pairs("./data")
    dataset = SeismicDataset(pairs)
    N = len(dataset)
    train_len = int(0.7 * N)
    val_len = int(0.2 * N)
    test_len = N - train_len - val_len
    train_set, val_set, _ = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        train_loss = 0
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            loop.set_postfix(train_mae=loss.item())
        train_mae = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_mae = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch:3d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"resunet_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
