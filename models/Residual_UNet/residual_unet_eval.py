# eval_resunet.py

import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

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
        self.velmap = self.velmap * 1500.0 + 1500.0

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

# ---------------------- Evaluation ---------------------- #
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = gather_pairs("./data")
    dataset = SeismicDataset(pairs)
    N = len(dataset)
    test_len = int(0.1 * N)
    val_len = int(0.2 * N)
    train_len = N - test_len - val_len
    _, _, test_set = random_split(dataset, [train_len, val_len, test_len])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load("resunet_epoch_100.pth", map_location=device))
    model.eval()

    mae_list, rmse_list, ssim_list, rel_l2_list = [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mae_list.append(torch.mean(torch.abs(pred - y)).item())
            rmse_list.append(torch.sqrt(torch.mean((pred - y) ** 2)).item())
            ssim_val = ssim(pred, y, data_range=3000.0, size_average=True)
            ssim_list.append(ssim_val.item())
            rel = torch.norm(pred - y) / torch.norm(y)
            rel_l2_list.append(rel.item())

    print(f"Test MAE:  {np.mean(mae_list):.4f}")
    print(f"Test RMSE: {np.mean(rmse_list):.4f}")
    print(f"Test SSIM: {np.mean(ssim_list):.4f}")
    print(f"Test RelL2:{np.mean(rel_l2_list):.4f}")

    # Visualize first 5 predictions
    for i in range(5):
        x, y = test_set[i]
        with torch.no_grad():
            pred = model(x[None].to(device)).cpu().squeeze(0).numpy()
        x_img = y.squeeze().numpy()
        y_img = pred.squeeze()

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(x_img, cmap="jet")
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(y_img, cmap="jet")
        ax[1].set_title("Prediction")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluate()
