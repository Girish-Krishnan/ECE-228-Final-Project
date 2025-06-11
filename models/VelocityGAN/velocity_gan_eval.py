import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from math import ceil

class SmallOpenFWI(Dataset):
    def __init__(self, pairs):
        seis, vel = [], []
        for sp, vp in pairs:
            s = np.load(sp).astype(np.float32)
            v = np.load(vp).astype(np.float32)
            if v.ndim == 3:
                v = v[:, None, :, :]
            seis.append(torch.from_numpy(s))
            vel.append(torch.from_numpy(v))
        self.seis = torch.cat(seis)
        self.vel = torch.clamp((torch.cat(vel) - 1500.0) / 1500.0, -1, 1)

    def __len__(self): return len(self.seis)
    def __getitem__(self, i): return self.seis[i], self.vel[i]

def gather_pairs(root):
    pairs = []
    for fam in ["FlatVel_A", "Style_A"]:
        for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
            mp = sp.replace("data", "model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A", "FlatFault_A"]:
        for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
            mp = os.path.join(root, fam, os.path.basename(sp).replace("seis", "vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs

NORM = {'bn': nn.BatchNorm2d}

class ConvB(nn.Module):
    def __init__(self, i, o, k=3, s=1, p=1, n='bn'):
        super().__init__()
        layers = [nn.Conv2d(i, o, k, s, p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2, True))
        self.seq = nn.Sequential(*layers)
    def forward(self, x): return self.seq(x)

class ConvB_Tanh(nn.Module):
    def __init__(self, i, o, k=3, s=1, p=1, n='bn'):
        super().__init__()
        layers = [nn.Conv2d(i, o, k, s, p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.Tanh())
        self.seq = nn.Sequential(*layers)
    def forward(self, x): return self.seq(x)

class DeconvB(nn.Module):
    def __init__(self, i, o, k=2, s=2, p=0, n='bn'):
        super().__init__()
        layers = [nn.ConvTranspose2d(i, o, k, s, p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2, True))
        self.seq = nn.Sequential(*layers)
    def forward(self, x): return self.seq(x)

class InversionNet(nn.Module):
    def __init__(self, d1=32, d2=64, d3=128, d4=256, d5=512, ratio=1.0):
        super().__init__()
        self.c1 = ConvB(5, d1, k=(7,1), s=(2,1), p=(3,0))
        self.c2_1 = ConvB(d1, d2, k=(3,1), s=(2,1), p=(1,0))
        self.c2_2 = ConvB(d2, d2, k=(3,1), p=(1,0))
        self.c3_1 = ConvB(d2, d2, k=(3,1), s=(2,1), p=(1,0))
        self.c3_2 = ConvB(d2, d2, k=(3,1), p=(1,0))
        self.c4_1 = ConvB(d2, d3, k=(3,1), s=(2,1), p=(1,0))
        self.c4_2 = ConvB(d3, d3, k=(3,1), p=(1,0))
        self.c5_1 = ConvB(d3, d3, s=2)
        self.c5_2 = ConvB(d3, d3)
        self.c6_1 = ConvB(d3, d4, s=2)
        self.c6_2 = ConvB(d4, d4)
        self.c7_1 = ConvB(d4, d4, s=2)
        self.c7_2 = ConvB(d4, d4)
        self.c8 = ConvB(d4, d5, k=(8, ceil(70*ratio/8)), p=0)
        self.d1_1 = DeconvB(d5, d5, k=5)
        self.d1_2 = ConvB(d5, d5)
        self.d2_1 = DeconvB(d5, d4, k=4, p=1)
        self.d2_2 = ConvB(d4, d4)
        self.d3_1 = DeconvB(d4, d3, k=4, p=1)
        self.d3_2 = ConvB(d3, d3)
        self.d4_1 = DeconvB(d3, d2, k=4, p=1)
        self.d4_2 = ConvB(d2, d2)
        self.d5_1 = DeconvB(d2, d1, k=4, p=1)
        self.d5_2 = ConvB(d1, d1)
        self.out = ConvB_Tanh(d1, 1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2_1(x); x = self.c2_2(x)
        x = self.c3_1(x); x = self.c3_2(x)
        x = self.c4_1(x); x = self.c4_2(x)
        x = self.c5_1(x); x = self.c5_2(x)
        x = self.c6_1(x); x = self.c6_2(x)
        x = self.c7_1(x); x = self.c7_2(x)
        x = self.c8(x)
        x = self.d1_1(x); x = self.d1_2(x)
        x = self.d2_1(x); x = self.d2_2(x)
        x = self.d3_1(x); x = self.d3_2(x)
        x = self.d4_1(x); x = self.d4_2(x)
        x = self.d5_1(x); x = self.d5_2(x)
        x = F.pad(x, [-5, -5, -5, -5])
        return self.out(x)

def denorm(t): return t * 1500.0 + 1500.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = gather_pairs(args.data_root)
    dataset = SmallOpenFWI(pairs)
    test_len = len(dataset) // 10
    test_set = torch.utils.data.Subset(dataset, range(len(dataset) - test_len, len(dataset)))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False)

    model = InversionNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
    model.eval()

    mae, rmse, ssim_sum, l2rel = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for s, v in test_loader:
            s, v = s.to(device), v.to(device)
            p = model(s)
            dp, dv = denorm(p), denorm(v)
            mae += torch.mean(torch.abs(dp - dv)).item() * s.size(0)
            rmse += torch.sqrt(torch.mean((dp - dv) ** 2)).item() * s.size(0)
            ssim_sum += ssim(dp, dv, data_range=3000.0, size_average=True).item() * s.size(0)
            l2rel += torch.norm(dp - dv) / torch.norm(dv)

    n = len(test_loader.dataset)
    print(f"Test MAE: {mae / n:.2f} m/s")
    print(f"Test RMSE: {rmse / n:.2f} m/s")
    print(f"Test SSIM: {ssim_sum / n:.4f}")
    print(f"Relative L2 Error: {l2rel / len(test_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)
