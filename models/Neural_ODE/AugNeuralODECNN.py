import os, glob, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

BATCH = 8
EPOCHS = 150
FM_CH, VEC_CH = 192, 384
AUG_FM_CH, AUG_VEC_CH = 32, 64
LR, WD, CLIP = 3e-4, 5e-5, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = torch.cuda.is_available()

class OpenFWISubset(Dataset):
    def __init__(self, root="."):
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

        seis_list, vel_list = [], []
        for sp, mp in pairs:
            seis = np.load(sp).astype(np.float32)
            vel  = np.load(mp).astype(np.float32)
            if vel.ndim == 3:
                vel = vel[:, None]
            seis_list.append(torch.from_numpy(seis))
            vel_list.append(torch.from_numpy(vel))

        self.seis = torch.cat(seis_list, 0)
        self.vel  = torch.cat(vel_list , 0)

        amp = torch.clamp(self.seis, -60, 60)
        self.seis = torch.log1p(torch.abs(amp)) * torch.sign(amp) / math.log(61)

        self.vmin, self.vmax = 1500.0, 4500.0
        self.vel = (self.vel - self.vmin) / (self.vmax - self.vmin) * 2 - 1

    def __len__(self): return len(self.seis)
    def __getitem__(self, idx): return self.seis[idx], self.vel[idx]

def conv_gn_relu(inp, outp, k=3, s=1, p=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(inp, outp, k, s, p, bias=False),
        nn.GroupNorm(groups, outp),
        nn.GELU())

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)
    def forward(self, x):
        w = F.gelu(self.fc1(F.adaptive_avg_pool2d(x, 1)))
        return x * torch.sigmoid(self.fc2(w))

class ResUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(conv_gn_relu(ch, ch), conv_gn_relu(ch, ch), SE(ch))
    def forward(self, x): return self.block(x) + x

class ODEFm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(conv_gn_relu(ch, ch), conv_gn_relu(ch, ch), nn.Conv2d(ch, ch, 1))
    def forward(self, t, h): return self.net(h)

class ODEVec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim))
    def forward(self, t, h): return self.net(h)

def integrate(h0, func, n_steps=4):
    t = torch.linspace(0, 1, n_steps + 1, device=h0.device, dtype=torch.float32)
    return odeint(func, h0, t, method="rk4")[-1]

class AugNODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_time = nn.Conv1d(5, 32, kernel_size=21, stride=4, padding=10)
        self.enc2d = nn.Sequential(
            conv_gn_relu(32, 64, 3, 2, 1), ResUnit(64),
            conv_gn_relu(64, 128, 3, 2, 1), ResUnit(128),
            conv_gn_relu(128, FM_CH, 3, 2, 1), ResUnit(FM_CH))
        self.fm_ode_func = ODEFm(FM_CH + AUG_FM_CH)
        self.fm_to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(FM_CH, VEC_CH))
        self.vec_ode_func = ODEVec(VEC_CH + AUG_VEC_CH)
        self.dec = nn.Sequential(nn.Linear(VEC_CH, VEC_CH * 2), nn.GELU(), nn.Linear(VEC_CH * 2, 4 * 35 * 35))
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        b = x.size(0)
        x = self.stem_time(x.view(b, 5, 1000 * 70)).view(b, 32, 250, 70)
        fm = self.enc2d(x)
        fm_aug = torch.cat([fm, torch.zeros(b, AUG_FM_CH, fm.size(2), fm.size(3), device=x.device)], 1)
        fm_out = integrate(fm_aug, self.fm_ode_func)[:, :FM_CH]
        vec = self.fm_to_vec(fm_out)
        vec_aug = torch.cat([vec, torch.zeros(b, AUG_VEC_CH, device=x.device)], 1)
        vec_out = integrate(vec_aug, self.vec_ode_func)[:, :VEC_CH]
        out = self.up(self.dec(vec_out).view(b, 4, 35, 35))
        return out.tanh()

def vel_denorm(t):
    return (t + 1) / 2 * (4500.0 - 1500.0) + 1500.0

def train():
    ds = OpenFWISubset(".")
    n_train = int(0.7 * len(ds))
    n_val = int(0.2 * len(ds))
    splits = [n_train, n_val, len(ds) - n_train - n_val]
    train_ds, val_ds, _ = random_split(ds, splits, generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = AugNODE().to(DEVICE)
    ema = torch.optim.swa_utils.AveragedModel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for seis, vel in tqdm(train_dl, leave=False, desc=f"Epoch {epoch}/{EPOCHS}"):
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP):
                pred = model(seis)
                loss_l1 = l1(pred, vel)
                loss_ss = 1 - ssim_fn((pred + 1) / 2, (vel + 1) / 2, data_range=1.0)
                loss = 0.85 * loss_l1 + 0.15 * loss_ss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            total += loss_l1.item() * seis.size(0)
        sched.step()

        model.eval(); ema.eval()
        val_l1, val_denorm = 0.0, 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP):
            for seis, vel in val_dl:
                seis, vel = seis.to(DEVICE), vel.to(DEVICE)
                pred = ema(seis)
                val_l1 += l1(pred, vel).item() * seis.size(0)
                val_denorm += torch.mean(torch.abs(vel_denorm(pred) - vel_denorm(vel))).item() * seis.size(0)

        print(f"Epoch {epoch:3d} Train MAE {total / len(train_dl.dataset):.4f} Val MAE {val_l1 / len(val_dl.dataset):.4f} Val MAE (denorm) {val_denorm / len(val_dl.dataset):.2f} m/s")

        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"aug_node_epoch{epoch}.pth")

def test():
    ds = OpenFWISubset(".")
    n_train = int(0.7 * len(ds))
    n_val = int(0.2 * len(ds))
    _, _, test_ds = random_split(ds, [n_train, n_val, len(ds) - n_train - n_val], generator=torch.Generator().manual_seed(0))
    dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = AugNODE().to(DEVICE)
    ema = torch.optim.swa_utils.AveragedModel(model)
    ckpt = torch.load("aug_node_epoch150.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    model.eval(); ema.eval()

    l1 = nn.L1Loss()
    with torch.no_grad():
        for seis, vel in dl:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            pred = ema(seis)
            print(f"Test MAE: {l1(pred, vel).item():.4f}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.add_parser("train")
    subparsers.add_parser("test")
    args = parser.parse_args()

    if args.mode == "train": train()
    elif args.mode == "test": test()
